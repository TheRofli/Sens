use std::{
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use async_trait::async_trait;
use sens_core::{CapabilityExecutor, CapabilityOutput};
use sens_protocol::{ArtifactRef, InvokeRequest, SensError};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex,
    time::timeout,
};

use crate::process_group::{KillOnCloseJob, terminate_tree};

#[derive(Debug, Clone)]
pub struct SightRuntimeConfig {
    pub node_executable: PathBuf,
    pub eye_root: PathBuf,
    pub worker_script: PathBuf,
}

impl SightRuntimeConfig {
    pub fn discover() -> anyhow::Result<Self> {
        let node_executable = std::env::var_os("SENS_NODE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("node"));
        let eye_root = discover_eye_root();
        let worker_script = discover_worker_script("eye-worker.mjs");
        Ok(Self {
            node_executable,
            eye_root,
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

fn discover_eye_root() -> PathBuf {
    if let Some(path) = std::env::var_os("SENS_EYE_ROOT") {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        let candidate = PathBuf::from(home)
            .join(".zcode")
            .join("workspace")
            .join("default")
            .join("eye");
        if candidate.join("src").join("service.mjs").is_file() {
            return candidate;
        }
    }
    PathBuf::from("sidecars").join("eye")
}

struct EyeProcess {
    _child: Child,
    _job: KillOnCloseJob,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

pub struct SightExecutor {
    config: SightRuntimeConfig,
    process: Mutex<Option<EyeProcess>>,
}

impl SightExecutor {
    pub fn new(config: SightRuntimeConfig) -> Self {
        Self {
            config,
            process: Mutex::new(None),
        }
    }

    async fn start(&self) -> Result<EyeProcess, SensError> {
        if !self
            .config
            .eye_root
            .join("src")
            .join("service.mjs")
            .is_file()
        {
            return Err(runtime_error(
                "sight_runtime_missing",
                format!(
                    "Eye service was not found at {}",
                    self.config.eye_root.display()
                ),
                "Open Sens diagnostics and repair Sight.",
            ));
        }
        if !self.config.worker_script.is_file() {
            return Err(runtime_error(
                "sight_adapter_missing",
                format!(
                    "Eye adapter was not found at {}",
                    self.config.worker_script.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.node_executable);
        command
            .arg(&self.config.worker_script)
            .env("SENS_EYE_ROOT", &self.config.eye_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        crate::process_group::hide_console(&mut command);
        let mut child = command
            .spawn()
            .map_err(|error| {
                runtime_error(
                    "sight_start_failed",
                    format!("Could not start Sight: {error}"),
                    "Check the bundled Node runtime in Sens diagnostics.",
                )
            })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            runtime_error(
                "sight_start_failed",
                "Sight has no stdin",
                "Repair the Sens installation.",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            runtime_error(
                "sight_start_failed",
                "Sight has no stdout",
                "Repair the Sens installation.",
            )
        })?;
        let job = KillOnCloseJob::assign(&child).map_err(|error| {
            runtime_error(
                "sight_start_failed",
                format!("Could not supervise the Sight process tree: {error}"),
                "Restart Sens or repair the installation.",
            )
        })?;
        Ok(EyeProcess {
            _child: child,
            _job: job,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    async fn invoke_worker(&self, request: &InvokeRequest) -> Result<Value, SensError> {
        validate_sight_input(request)?;
        let mut guard = self.process.lock().await;
        if guard.is_none() {
            *guard = Some(self.start().await?);
        }
        let process = guard.as_mut().expect("process initialized");
        let payload = json!({
            "requestId": request.request_id,
            "operation": request.operation,
            "input": request.input,
            "noStore": request.no_store,
            "maxCalls": request.max_calls,
        });
        let mut encoded = serde_json::to_vec(&payload).map_err(|error| {
            runtime_error(
                "sight_protocol_error",
                error.to_string(),
                "Retry the operation.",
            )
        })?;
        encoded.push(b'\n');
        if let Err(error) = process.stdin.write_all(&encoded).await {
            *guard = None;
            return Err(runtime_error(
                "sight_disconnected",
                format!("Sight worker disconnected: {error}"),
                "Retry; Sens will restart Sight.",
            ));
        }
        process.stdin.flush().await.map_err(|error| {
            runtime_error(
                "sight_disconnected",
                format!("Sight worker could not receive the request: {error}"),
                "Retry; Sens will restart Sight.",
            )
        })?;

        let wait = Duration::from_millis(request.timeout_ms.unwrap_or(120_000));
        let response = timeout(wait, process.stdout.next_line()).await;
        let line = match response {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_disconnected",
                    "Sight worker exited without a response",
                    "Retry; Sens will restart Sight.",
                ));
            }
            Ok(Err(error)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_protocol_error",
                    format!("Could not read Sight response: {error}"),
                    "Retry the operation.",
                ));
            }
            Err(_) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_timeout",
                    format!("Sight exceeded its {} ms timeout", wait.as_millis()),
                    "Retry with quick detail or a longer timeout.",
                ));
            }
        };
        let response: Value = serde_json::from_str(&line).map_err(|error| {
            runtime_error(
                "sight_protocol_error",
                format!("Sight returned invalid JSON: {error}"),
                "Retry; if this continues, export Sens diagnostics.",
            )
        })?;
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            let message = response
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Sight failed without an error message");
            return Err(runtime_error(
                "sight_failed",
                message,
                "Check the Sight provider and input in Sens diagnostics.",
            ));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[async_trait]
impl CapabilityExecutor for SightExecutor {
    async fn invoke(&self, request: &InvokeRequest) -> Result<CapabilityOutput, SensError> {
        let result = self.invoke_worker(request).await?;
        let artifacts = result
            .get("artifactId")
            .and_then(Value::as_str)
            .map(|id| {
                vec![ArtifactRef {
                    id: id.to_owned(),
                    kind: "eye_result".into(),
                    uri: None,
                }]
            })
            .unwrap_or_default();
        Ok(CapabilityOutput {
            data: result
                .get("data")
                .cloned()
                .unwrap_or_else(|| result.clone()),
            artifacts,
            provenance: Vec::new(),
            usage: result.get("usage").cloned().unwrap_or(Value::Null),
            warnings: Vec::new(),
        })
    }
}

fn validate_sight_input(request: &InvokeRequest) -> Result<(), SensError> {
    let input = request.input.as_object().ok_or_else(|| {
        runtime_error(
            "invalid_input",
            "Sight input must be a JSON object",
            "Pass the tool arguments defined by Sens.",
        )
    })?;
    match request.operation.as_str() {
        "artifact_get" => require_string(input, "artifactId"),
        "compare" => {
            require_existing_file(input, "referencePath")?;
            require_existing_file(input, "candidatePath")
        }
        "locate" => {
            require_source(input)?;
            require_string(input, "target")
        }
        "see" | "read" | "inspect" => require_source(input),
        _ => Ok(()),
    }
}

fn require_source(input: &serde_json::Map<String, Value>) -> Result<(), SensError> {
    if input.get("artifactId").and_then(Value::as_str).is_some() {
        return Ok(());
    }
    require_existing_file(input, "imagePath")
}

fn require_string(input: &serde_json::Map<String, Value>, key: &str) -> Result<(), SensError> {
    if input
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        Ok(())
    } else {
        Err(runtime_error(
            "invalid_input",
            format!("{key} is required"),
            "Pass the required tool argument.",
        ))
    }
}

fn require_existing_file(
    input: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<(), SensError> {
    let value = input.get(key).and_then(Value::as_str).ok_or_else(|| {
        runtime_error(
            "invalid_input",
            format!("{key} is required"),
            "Pass an existing local file.",
        )
    })?;
    if Path::new(value).is_file() {
        Ok(())
    } else {
        Err(runtime_error(
            "file_not_found",
            format!("{key} does not exist or is not a file: {value}"),
            "Pass an existing local file path.",
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
    fn locate_requires_target() {
        let request = InvokeRequest::new(
            "sight",
            "locate",
            json!({ "artifactId": "existing-artifact" }),
        );
        let error = validate_sight_input(&request).expect_err("missing target");
        assert_eq!(error.code, "invalid_input");
    }
}
