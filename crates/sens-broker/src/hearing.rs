use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use async_trait::async_trait;
use sens_core::{CapabilityExecutor, CapabilityOutput};
use sens_protocol::{InvokeRequest, SensError};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};
use tracing::info;

use crate::process_group::{KillOnCloseJob, terminate_tree};

#[derive(Debug, Clone)]
pub struct HearingRuntimeConfig {
    pub python_executable: PathBuf,
    pub speech_root: PathBuf,
    pub worker_script: PathBuf,
}

impl HearingRuntimeConfig {
    pub fn discover() -> anyhow::Result<Self> {
        let speech_root = std::env::var_os("SENS_SPEECH_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let development = PathBuf::from(r"D:\Speech");
                if development.join("speech_app").is_dir() {
                    development
                } else {
                    PathBuf::from("sidecars").join("speech")
                }
            });
        let python_executable = std::env::var_os("SENS_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let bundled = speech_root.join(".venv").join("Scripts").join("python.exe");
                if bundled.is_file() {
                    bundled
                } else {
                    PathBuf::from("python")
                }
            });
        let worker_script = discover_worker_script("hearing-worker.py");
        Ok(Self {
            python_executable,
            speech_root,
            worker_script,
        })
    }
}

fn discover_worker_script(name: &str) -> PathBuf {
    if let Some(root) = std::env::var_os("SENS_SIDECARS_ROOT") {
        return PathBuf::from(root).join(name);
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        let installed = parent.join("sidecars").join(name);
        if installed.is_file() {
            return installed;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("sidecars")
        .join(name)
}

struct HearingProcess {
    _child: Child,
    _job: KillOnCloseJob,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

pub struct HearingExecutor {
    config: HearingRuntimeConfig,
    process: Mutex<Option<HearingProcess>>,
}

impl HearingExecutor {
    pub fn new(config: HearingRuntimeConfig) -> Self {
        Self {
            config,
            process: Mutex::new(None),
        }
    }

    async fn start(&self) -> Result<HearingProcess, SensError> {
        if !self.config.speech_root.join("speech_app").is_dir() {
            return Err(runtime_error(
                "hearing_runtime_missing",
                format!(
                    "Speech was not found at {}",
                    self.config.speech_root.display()
                ),
                "Open Sens diagnostics and repair Hearing.",
            ));
        }
        if !self.config.worker_script.is_file() {
            return Err(runtime_error(
                "hearing_adapter_missing",
                format!(
                    "Hearing adapter was not found at {}",
                    self.config.worker_script.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.python_executable);
        command
            .arg(&self.config.worker_script)
            .env("SENS_SPEECH_ROOT", &self.config.speech_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        crate::process_group::hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
            runtime_error(
                "hearing_start_failed",
                format!("Could not start Hearing: {error}"),
                "Check the bundled Python runtime in Sens diagnostics.",
            )
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            runtime_error(
                "hearing_start_failed",
                "Hearing has no stdin",
                "Repair the Sens installation.",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            runtime_error(
                "hearing_start_failed",
                "Hearing has no stdout",
                "Repair the Sens installation.",
            )
        })?;
        let job = KillOnCloseJob::assign(&child).map_err(|error| {
            runtime_error(
                "hearing_start_failed",
                format!("Could not supervise the Hearing process tree: {error}"),
                "Restart Sens or repair the installation.",
            )
        })?;
        Ok(HearingProcess {
            _child: child,
            _job: job,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    async fn invoke_worker(&self, request: &InvokeRequest) -> Result<Value, SensError> {
        validate_hearing_input(request)?;
        info!(
            request_id = %request.request_id,
            operation = %request.operation,
            timeout_ms = ?request.timeout_ms,
            "Hearing request received"
        );
        let mut guard = self.process.lock().await;
        if guard.is_none() {
            info!(request_id = %request.request_id, "starting Hearing worker");
            *guard = Some(self.start().await?);
            info!(request_id = %request.request_id, "Hearing worker started");
        }
        let process = guard.as_mut().expect("process initialized");
        let payload = json!({
            "requestId": request.request_id,
            "operation": request.operation,
            "input": request.input,
            "noStore": request.no_store,
        });
        let mut encoded = serde_json::to_vec(&payload).map_err(|error| {
            runtime_error(
                "hearing_protocol_error",
                error.to_string(),
                "Retry the operation.",
            )
        })?;
        encoded.push(b'\n');
        if let Err(error) = process.stdin.write_all(&encoded).await {
            *guard = None;
            return Err(runtime_error(
                "hearing_disconnected",
                format!("Hearing worker disconnected: {error}"),
                "Retry; Sens will restart Hearing.",
            ));
        }
        info!(request_id = %request.request_id, "Hearing request written");
        process.stdin.flush().await.map_err(|error| {
            runtime_error(
                "hearing_disconnected",
                format!("Hearing worker could not receive the request: {error}"),
                "Retry; Sens will restart Hearing.",
            )
        })?;

        let wait = Duration::from_millis(request.timeout_ms.unwrap_or(600_000));
        info!(request_id = %request.request_id, wait_ms = wait.as_millis(), "waiting for Hearing response");
        let response = timeout(wait, process.stdout.next_line()).await;
        let line = match response {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "hearing_disconnected",
                    "Hearing worker exited without a response",
                    "Retry; Sens will restart Hearing.",
                ));
            }
            Ok(Err(error)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "hearing_protocol_error",
                    format!("Could not read Hearing response: {error}"),
                    "Retry the operation.",
                ));
            }
            Err(_) => {
                info!(request_id = %request.request_id, "Hearing response timed out; terminating worker tree");
                terminate_tree(&mut process._child).await;
                info!(request_id = %request.request_id, "Hearing worker tree termination requested");
                *guard = None;
                return Err(runtime_error(
                    "hearing_timeout",
                    format!("Hearing exceeded its {} ms timeout", wait.as_millis()),
                    "Retry with a shorter audio file or a longer timeout.",
                ));
            }
        };
        let response: Value = serde_json::from_str(&line).map_err(|error| {
            runtime_error(
                "hearing_protocol_error",
                format!("Hearing returned invalid JSON: {error}"),
                "Retry; if this continues, export Sens diagnostics.",
            )
        })?;
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            let message = response
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Hearing failed without an error message");
            return Err(runtime_error(
                "hearing_failed",
                message,
                "Check the selected Hearing model and input in Sens diagnostics.",
            ));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[async_trait]
impl CapabilityExecutor for HearingExecutor {
    async fn invoke(&self, request: &InvokeRequest) -> Result<CapabilityOutput, SensError> {
        let result = self.invoke_worker(request).await?;
        Ok(CapabilityOutput {
            data: result,
            ..Default::default()
        })
    }
}

fn validate_hearing_input(request: &InvokeRequest) -> Result<(), SensError> {
    if request.operation == "dictation_status" {
        return Ok(());
    }
    let input = request.input.as_object().ok_or_else(|| {
        runtime_error(
            "invalid_input",
            "Hearing input must be a JSON object",
            "Pass the tool arguments defined by Sens.",
        )
    })?;
    let value = input
        .get("audioPath")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            runtime_error(
                "invalid_input",
                "audioPath is required",
                "Pass an existing local audio file.",
            )
        })?;
    if Path::new(value).is_file() {
        Ok(())
    } else {
        Err(runtime_error(
            "file_not_found",
            format!("audioPath does not exist or is not a file: {value}"),
            "Pass an existing local audio file.",
        ))
    }
}

fn runtime_error(
    code: impl Into<String>,
    message: impl Into<String>,
    action: impl Into<String>,
) -> SensError {
    SensError {
        code: code.into(),
        message: message.into(),
        recoverable: true,
        action: Some(action.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hearing_requires_existing_audio_file() {
        let request = InvokeRequest::new(
            "hearing",
            "hear",
            json!({ "audioPath": "Z:/missing/audio.wav" }),
        );
        let error = validate_hearing_input(&request).expect_err("missing file");
        assert_eq!(error.code, "file_not_found");
    }
}
