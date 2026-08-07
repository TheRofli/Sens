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
use tracing::info;

use crate::process_group::{KillOnCloseJob, terminate_tree};

/// Local vision operations served by the Python sight-worker (no network):
/// see, read, locate, inspect, compare (deterministic pixel diff).
/// artifact_get is served by the cloud Eye (artifact store).
const CLOUD_OPERATIONS: [&str; 1] = ["artifact_get"];

#[derive(Debug, Clone)]
pub struct SightRuntimeConfig {
    // Local deterministic stack (RapidOCR + OpenCV + YOLO + CLIP).
    pub python_executable: PathBuf,
    pub speech_root: PathBuf,
    pub local_worker: PathBuf,
    // Optional cloud Eye (VLM via provider API) for compare/artifact_get.
    pub node_executable: PathBuf,
    pub eye_root: PathBuf,
    pub cloud_worker: PathBuf,
    // Global default VLM semantics pack chosen in the desktop settings
    // (vision.visionPack); per-request MCP arguments still win.
    pub vision_pack: Option<String>,
}

impl SightRuntimeConfig {
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
        let node_executable = std::env::var_os("SENS_NODE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("node"));
        let eye_root = discover_eye_root();
        let local_worker = discover_worker_script("sight-worker.py");
        let cloud_worker = discover_worker_script("eye-worker.mjs");
        let vision_pack = discover_vision_pack(&eye_root);
        Ok(Self {
            python_executable,
            speech_root,
            local_worker,
            node_executable,
            eye_root,
            cloud_worker,
            vision_pack,
        })
    }
}

/// Best-effort read of the desktop settings' vision pack (vision.visionPack in
/// the Eye config). Missing file/field or unknown value -> None (worker stays lite).
fn discover_vision_pack(eye_root: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(eye_root.join("config.json")).ok()?;
    let document: Value = serde_json::from_str(&contents).ok()?;
    let pack = document
        .get("vision")?
        .get("visionPack")?
        .as_str()?
        .to_owned();
    matches!(pack.as_str(), "lite" | "quality" | "quality_large").then_some(pack)
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

struct WorkerProcess {
    _child: Child,
    _job: KillOnCloseJob,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

pub struct SightExecutor {
    config: SightRuntimeConfig,
    local: Mutex<Option<WorkerProcess>>,
    cloud: Mutex<Option<WorkerProcess>>,
}

impl SightExecutor {
    pub fn new(config: SightRuntimeConfig) -> Self {
        Self {
            config,
            local: Mutex::new(None),
            cloud: Mutex::new(None),
        }
    }

    async fn start_local(&self) -> Result<WorkerProcess, SensError> {
        if !self.config.speech_root.join("speech_app").is_dir() {
            return Err(runtime_error(
                "sight_runtime_missing",
                format!(
                    "Speech was not found at {}",
                    self.config.speech_root.display()
                ),
                "Open Sens diagnostics and repair Sight.",
            ));
        }
        if !self.config.local_worker.is_file() {
            return Err(runtime_error(
                "sight_adapter_missing",
                format!(
                    "Sight adapter was not found at {}",
                    self.config.local_worker.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.python_executable);
        command
            .arg(&self.config.local_worker)
            .env("SENS_SPEECH_ROOT", &self.config.speech_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if let Some(pack) = &self.config.vision_pack {
            command.env("SENS_VISION_PACK", pack);
        }
        crate::process_group::hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
            runtime_error(
                "sight_start_failed",
                format!("Could not start Sight: {error}"),
                "Check the bundled Python runtime in Sens diagnostics.",
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
        Ok(WorkerProcess {
            _child: child,
            _job: job,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    async fn start_cloud(&self) -> Result<WorkerProcess, SensError> {
        if !self
            .config
            .eye_root
            .join("src")
            .join("service.mjs")
            .is_file()
        {
            return Err(runtime_error(
                "sight_cloud_unavailable",
                "Cloud Eye was not found on this machine",
                "Install Eye or use the local vision tools (see/read/locate/inspect).",
            ));
        }
        if !self.config.cloud_worker.is_file() {
            return Err(runtime_error(
                "sight_adapter_missing",
                format!(
                    "Eye adapter was not found at {}",
                    self.config.cloud_worker.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.node_executable);
        command
            .arg(&self.config.cloud_worker)
            .env("SENS_EYE_ROOT", &self.config.eye_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        crate::process_group::hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
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
        Ok(WorkerProcess {
            _child: child,
            _job: job,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    async fn invoke_worker(
        &self,
        request: &InvokeRequest,
        cloud: bool,
    ) -> Result<Value, SensError> {
        validate_sight_input(request, cloud)?;
        let runtime = if cloud { "Eye" } else { "Sight" };
        info!(
            request_id = %request.request_id,
            operation = %request.operation,
            cloud,
            timeout_ms = ?request.timeout_ms,
            "{runtime} request received"
        );
        let mut guard = if cloud {
            self.cloud.lock().await
        } else {
            self.local.lock().await
        };
        if guard.is_none() {
            info!(request_id = %request.request_id, "starting {runtime} worker");
            *guard = Some(if cloud {
                self.start_cloud().await?
            } else {
                self.start_local().await?
            });
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
                format!("{runtime} worker disconnected: {error}"),
                "Retry; Sens will restart Sight.",
            ));
        }
        if let Err(error) = process.stdin.flush().await {
            // The write may have landed in the pipe buffer while the worker
            // process was already gone; drop the stale process so the next
            // request respawns a fresh worker instead of writing to a corpse.
            *guard = None;
            return Err(runtime_error(
                "sight_disconnected",
                format!("{runtime} worker could not receive the request: {error}"),
                "Retry; Sens will restart Sight.",
            ));
        }

        // Local stack warms up models on the first request; cloud Eye is slow
        // to respond, so both get a generous default timeout.
        let wait = Duration::from_millis(request.timeout_ms.unwrap_or(600_000));
        info!(request_id = %request.request_id, wait_ms = wait.as_millis(), "waiting for {runtime} response");
        let response = timeout(wait, process.stdout.next_line()).await;
        let line = match response {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_disconnected",
                    format!("{runtime} worker exited without a response"),
                    "Retry; Sens will restart Sight.",
                ));
            }
            Ok(Err(error)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_protocol_error",
                    format!("Could not read {runtime} response: {error}"),
                    "Retry the operation.",
                ));
            }
            Err(_) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_timeout",
                    format!("{runtime} exceeded its {} ms timeout", wait.as_millis()),
                    "Retry; if this continues, export Sens diagnostics.",
                ));
            }
        };
        let response: Value = serde_json::from_str(&line).map_err(|error| {
            runtime_error(
                "sight_protocol_error",
                format!("{runtime} returned invalid JSON: {error}"),
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
        let cloud = is_cloud_operation(&request.operation);
        let result = self.invoke_worker(request, cloud).await?;
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

fn is_cloud_operation(operation: &str) -> bool {
    CLOUD_OPERATIONS.contains(&operation)
}

fn validate_sight_input(request: &InvokeRequest, cloud: bool) -> Result<(), SensError> {
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
        "inspect" => {
            require_source(input)?;
            let has_region = input.get("region").is_some();
            let has_target = input
                .get("target")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty());
            if !has_region && !has_target {
                return Err(runtime_error(
                    "invalid_input",
                    "inspect requires region or target",
                    "Pass a pixel region or a visible text target.",
                ));
            }
            Ok(())
        }
        "see" | "read" => require_source(input),
        "zoom" => {
            require_source(input)?;
            let has_region = input.get("region").is_some();
            let has_som = input.get("somId").is_some();
            if !has_region && !has_som {
                return Err(runtime_error(
                    "invalid_input",
                    "zoom requires region or somId",
                    "Pass a pixel region or a SoM element id.",
                ));
            }
            Ok(())
        }
        "ask" => {
            require_source(input)?;
            require_string(input, "question")
        }
        "element" => {
            require_source(input)?;
            if input.get("id").is_none() {
                return Err(runtime_error(
                    "invalid_input",
                    "element requires id",
                    "Pass a SoM element id from a prior sens_see document.",
                ));
            }
            Ok(())
        }
        "motion" | "capture" => require_string(input, "url"),
        "vision_prompt" | "warm" => Ok(()),
        _ if cloud => Err(runtime_error(
            "operation_not_supported",
            format!("Eye does not support operation {}", request.operation),
            "Use see/read/locate/inspect for local vision.",
        )),
        _ => Err(runtime_error(
            "operation_not_supported",
            format!("Sight does not support operation {}", request.operation),
            "Use one of the supported Sight operations.",
        )),
    }
}

fn require_source(input: &serde_json::Map<String, Value>) -> Result<(), SensError> {
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
    use std::fs;

    fn existing_image() -> PathBuf {
        let path = std::env::temp_dir().join("sens-sight-test.png");
        fs::write(&path, b"fixture").expect("write fixture");
        path
    }

    #[test]
    fn locate_requires_target() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "locate",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing target");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn inspect_requires_region_or_target() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "inspect",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing focus");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn inspect_accepts_region() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "inspect",
            json!({ "imagePath": path.to_string_lossy(), "region": { "x": 0, "y": 0, "width": 10, "height": 10 } }),
        );
        assert!(validate_sight_input(&request, false).is_ok());
    }

    #[test]
    fn zoom_requires_region_or_som_id() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "zoom",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing focus");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn zoom_accepts_som_id() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "zoom",
            json!({ "imagePath": path.to_string_lossy(), "somId": 3 }),
        );
        assert!(validate_sight_input(&request, false).is_ok());
    }

    #[test]
    fn ask_requires_question() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "ask",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing question");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn motion_requires_url() {
        let request = InvokeRequest::new("sight", "motion", json!({}));
        let error = validate_sight_input(&request, false).expect_err("missing url");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn vision_prompt_needs_no_input() {
        let request = InvokeRequest::new("sight", "vision_prompt", json!({}));
        assert!(validate_sight_input(&request, false).is_ok());
    }

    #[test]
    fn cloud_operations_are_recognized() {
        assert!(is_cloud_operation("artifact_get"));
        assert!(!is_cloud_operation("compare"));
        assert!(!is_cloud_operation("see"));
        assert!(!is_cloud_operation("inspect"));
    }
}
