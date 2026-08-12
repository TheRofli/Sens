use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod touch;

pub const PROTOCOL_VERSION: &str = "1.0.0";
pub const PRODUCT_NAME: &str = "Sens";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum AppState {
    Starting,
    Ready,
    Degraded,
    Offline,
    Updating,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityState {
    Unavailable,
    Installing,
    Asleep,
    Starting,
    Ready,
    Busy,
    Error,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionState {
    Disconnected,
    Configured,
    Connecting,
    Connected,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running,
    WaitingInput,
    Succeeded,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    LocalFileRead,
    ProviderNetwork,
    ScreenCapture,
    LiveMicrophone,
    SystemOutput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeDescriptor {
    pub kind: String,
    pub lazy: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityManifest {
    pub id: String,
    pub version: String,
    pub title: String,
    pub description: String,
    pub operations: Vec<String>,
    pub runtime: RuntimeDescriptor,
    pub permissions: Vec<Permission>,
    pub state: CapabilityState,
    pub artifact_types: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct StatusSnapshot {
    pub product: String,
    pub version: String,
    pub protocol_version: String,
    pub state: AppState,
    pub broker_pid: u32,
    pub uptime_ms: u64,
    pub connections: Vec<ConnectionSummary>,
    pub capabilities: Vec<CapabilityManifest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConnectionSummary {
    pub id: String,
    pub title: String,
    pub transport: String,
    pub state: ConnectionState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InvokeRequest {
    pub request_id: String,
    pub capability_id: String,
    pub operation: String,
    #[serde(default)]
    pub input: Value,
    #[serde(default)]
    pub no_store: bool,
    pub timeout_ms: Option<u64>,
    pub max_calls: Option<u32>,
}

impl InvokeRequest {
    pub fn new(
        capability_id: impl Into<String>,
        operation: impl Into<String>,
        input: Value,
    ) -> Self {
        Self {
            request_id: uuid::Uuid::new_v4().to_string(),
            capability_id: capability_id.into(),
            operation: operation.into(),
            input,
            no_store: false,
            timeout_ms: None,
            max_calls: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactRef {
    pub id: String,
    pub kind: String,
    pub uri: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    pub kind: String,
    pub method: String,
    pub confidence: Option<f64>,
    pub source_artifact_id: Option<String>,
    pub region: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct SensError {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    pub action: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InvokeResult {
    pub request_id: String,
    pub capability_id: String,
    pub operation: String,
    pub status: JobState,
    #[serde(default)]
    pub data: Value,
    #[serde(default)]
    pub artifacts: Vec<ArtifactRef>,
    #[serde(default)]
    pub provenance: Vec<Provenance>,
    #[serde(default)]
    pub usage: Value,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub error: Option<SensError>,
    pub elapsed_ms: u64,
}

impl InvokeResult {
    pub fn failed(request: &InvokeRequest, error: SensError, elapsed_ms: u64) -> Self {
        Self {
            request_id: request.request_id.clone(),
            capability_id: request.capability_id.clone(),
            operation: request.operation.clone(),
            status: JobState::Failed,
            data: Value::Null,
            artifacts: Vec::new(),
            provenance: Vec::new(),
            usage: Value::Null,
            warnings: Vec::new(),
            error: Some(error),
            elapsed_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BrokerRequest {
    Status,
    Capabilities,
    Invoke { request: InvokeRequest },
    Ping,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BrokerResponse {
    Status {
        status: StatusSnapshot,
    },
    Capabilities {
        capabilities: Vec<CapabilityManifest>,
    },
    Invoke {
        result: InvokeResult,
    },
    Pong {
        protocol_version: String,
    },
    Error {
        error: SensError,
    },
}

pub fn default_capabilities() -> Vec<CapabilityManifest> {
    vec![
        CapabilityManifest {
            id: "sight".into(),
            version: "1.4.0".into(),
            title: "Sight".into(),
            description:
                "Local CPU vision: OCR, geometry, design tokens, Visual Scene v2, URL capture, and deterministic comparison. Optional Eye is used only for legacy artifacts and video.".into(),
            operations: vec![
                "see",
                "read",
                "locate",
                "inspect",
                "compare",
                "review",
                "web_start",
                "web_review",
                "artifact_get",
                "zoom",
                "ask",
                "element",
                "motion",
                "capture",
                "vision_prompt",
                "warm",
                "watch",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            runtime: RuntimeDescriptor {
                kind: "python_sidecar".into(),
                lazy: true,
            },
            permissions: vec![Permission::LocalFileRead, Permission::ProviderNetwork],
            state: CapabilityState::Asleep,
            artifact_types: vec!["image", "crop", "spec", "diff"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        },
        CapabilityManifest {
            id: "hearing".into(),
            version: "1.0.0".into(),
            title: "Hearing".into(),
            description:
                "Local dictation and side-effect-free audio file transcription through Speech."
                    .into(),
            operations: vec![
                "hear",
                "dictation_status",
                "dictation_start",
                "dictation_settings",
                "dictation_stop",
                "model_status",
                "model_install",
                "fetch",
            ]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            runtime: RuntimeDescriptor {
                kind: "python_sidecar".into(),
                lazy: true,
            },
            permissions: vec![
                Permission::LocalFileRead,
                Permission::LiveMicrophone,
                Permission::SystemOutput,
                Permission::ProviderNetwork,
            ],
            state: CapabilityState::Asleep,
            artifact_types: vec!["audio", "transcript"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        },
        CapabilityManifest {
            id: touch::TOUCH_CAPABILITY_ID.into(),
            version: "1.4.0".into(),
            title: "Touch".into(),
            description:
                "Delegation of self-contained work to cheap worker models (OpenRouter/DeepSeek/OpenAI-compatible) with roles, budgets, context isolation, broker-issued evidence receipts, and deterministic verification. Opt-in; requires a provider key. Workers have no secrets, no direct filesystem, and no network; the broker executes all privileged actions.".into(),
            operations: vec![
                "touch",
                "parallel",
                "opinions",
                "verify",
                "status",
                "cancel",
                "check",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            runtime: RuntimeDescriptor {
                kind: "python_sidecar".into(),
                lazy: true,
            },
            permissions: vec![Permission::LocalFileRead, Permission::ProviderNetwork],
            // Coordinator not registered yet (Slice 0 freezes the contract);
            // the capability is advertised so hosts see the intent honestly.
            state: CapabilityState::Unavailable,
            artifact_types: vec!["worker_result", "evidence_receipt", "patch"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_round_trip_preserves_request() {
        let request = BrokerRequest::Invoke {
            request: InvokeRequest::new(
                "sight",
                "read",
                serde_json::json!({ "imagePath": "fixture.png" }),
            ),
        };
        let encoded = serde_json::to_string(&request).expect("encode");
        let decoded: BrokerRequest = serde_json::from_str(&encoded).expect("decode");
        match decoded {
            BrokerRequest::Invoke { request } => {
                assert_eq!(request.capability_id, "sight");
                assert_eq!(request.operation, "read");
            }
            _ => panic!("wrong request variant"),
        }
    }

    #[test]
    fn default_registry_is_capability_first() {
        let capabilities = default_capabilities();
        assert_eq!(capabilities.len(), 3);
        assert!(capabilities.iter().any(|item| item.id == "sight"));
        assert!(capabilities.iter().any(|item| item.id == "hearing"));
        assert!(
            capabilities
                .iter()
                .any(|item| item.id == touch::TOUCH_CAPABILITY_ID)
        );
        assert!(capabilities.iter().all(|item| item.runtime.lazy));
        let sight = capabilities
            .iter()
            .find(|item| item.id == "sight")
            .expect("sight capability");
        assert!(
            sight
                .operations
                .iter()
                .any(|operation| operation == "review")
        );
        assert!(
            sight
                .operations
                .iter()
                .any(|operation| operation == "web_start")
        );
        assert!(
            sight
                .operations
                .iter()
                .any(|operation| operation == "web_review")
        );
        // Touch advertises every operation from Slice 0 (manifest-first lesson
        // from the watch/fetch gap) even though the coordinator lands in
        // Slice 1.
        let touch = capabilities
            .iter()
            .find(|item| item.id == touch::TOUCH_CAPABILITY_ID)
            .expect("touch capability");
        for operation in [
            "touch", "parallel", "opinions", "verify", "status", "cancel", "check",
        ] {
            assert!(
                touch.operations.iter().any(|item| item == operation),
                "touch manifest must advertise {operation}"
            );
        }
        assert_eq!(touch.state, CapabilityState::Unavailable);
    }
}
