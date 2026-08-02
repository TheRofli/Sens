use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use directories::ProjectDirs;
use sens_protocol::{
    AppState, ArtifactRef, CapabilityManifest, CapabilityState, ConnectionState, ConnectionSummary,
    InvokeRequest, InvokeResult, JobState, PRODUCT_NAME, PROTOCOL_VERSION, Provenance, SensError,
    StatusSnapshot, default_capabilities,
};
use serde_json::Value;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub log_dir: PathBuf,
    pub runtime_dir: PathBuf,
}

impl RuntimePaths {
    pub fn discover() -> Self {
        let project = ProjectDirs::from("dev", "Sens", "Sens")
            .expect("Windows and supported desktop platforms provide a home directory");
        let data_dir = project.data_local_dir().to_path_buf();
        Self {
            config_dir: project.config_dir().to_path_buf(),
            cache_dir: project.cache_dir().to_path_buf(),
            log_dir: data_dir.join("logs"),
            runtime_dir: data_dir.join("runtime"),
            data_dir,
        }
    }

    pub fn ensure(&self) -> std::io::Result<()> {
        for path in [
            &self.config_dir,
            &self.data_dir,
            &self.cache_dir,
            &self.log_dir,
            &self.runtime_dir,
        ] {
            std::fs::create_dir_all(path)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct CapabilityOutput {
    pub data: Value,
    pub artifacts: Vec<ArtifactRef>,
    pub provenance: Vec<Provenance>,
    pub usage: Value,
    pub warnings: Vec<String>,
}

#[async_trait]
pub trait CapabilityExecutor: Send + Sync {
    async fn invoke(&self, request: &InvokeRequest) -> Result<CapabilityOutput, SensError>;
}

struct CoreInner {
    started_at: Instant,
    state: RwLock<AppState>,
    capabilities: RwLock<HashMap<String, CapabilityManifest>>,
    executors: RwLock<HashMap<String, Arc<dyn CapabilityExecutor>>>,
}

#[derive(Clone)]
pub struct SensCore {
    inner: Arc<CoreInner>,
}

impl Default for SensCore {
    fn default() -> Self {
        Self::new()
    }
}

impl SensCore {
    pub fn new() -> Self {
        let capabilities = default_capabilities()
            .into_iter()
            .map(|manifest| (manifest.id.clone(), manifest))
            .collect();
        Self {
            inner: Arc::new(CoreInner {
                started_at: Instant::now(),
                state: RwLock::new(AppState::Starting),
                capabilities: RwLock::new(capabilities),
                executors: RwLock::new(HashMap::new()),
            }),
        }
    }

    pub async fn mark_ready(&self) {
        *self.inner.state.write().await = AppState::Ready;
    }

    pub async fn set_app_state(&self, state: AppState) {
        *self.inner.state.write().await = state;
    }

    pub async fn register_executor(
        &self,
        capability_id: impl Into<String>,
        executor: Arc<dyn CapabilityExecutor>,
    ) -> Result<(), SensError> {
        let capability_id = capability_id.into();
        if !self
            .inner
            .capabilities
            .read()
            .await
            .contains_key(&capability_id)
        {
            return Err(SensError {
                code: "capability_not_found".into(),
                message: format!("Unknown capability: {capability_id}"),
                recoverable: false,
                action: None,
            });
        }
        self.inner
            .executors
            .write()
            .await
            .insert(capability_id, executor);
        Ok(())
    }

    pub async fn set_capability_state(&self, capability_id: &str, state: CapabilityState) {
        if let Some(manifest) = self.inner.capabilities.write().await.get_mut(capability_id) {
            manifest.state = state;
        }
    }

    pub async fn capabilities(&self) -> Vec<CapabilityManifest> {
        let mut values: Vec<_> = self
            .inner
            .capabilities
            .read()
            .await
            .values()
            .cloned()
            .collect();
        values.sort_by(|left, right| left.id.cmp(&right.id));
        values
    }

    pub async fn status(&self) -> StatusSnapshot {
        StatusSnapshot {
            product: PRODUCT_NAME.into(),
            version: env!("CARGO_PKG_VERSION").into(),
            protocol_version: PROTOCOL_VERSION.into(),
            state: *self.inner.state.read().await,
            broker_pid: std::process::id(),
            uptime_ms: self
                .inner
                .started_at
                .elapsed()
                .as_millis()
                .try_into()
                .unwrap_or(u64::MAX),
            connections: vec![ConnectionSummary {
                id: "mcp".into(),
                title: "MCP host".into(),
                transport: "stdio_proxy".into(),
                state: ConnectionState::Configured,
            }],
            capabilities: self.capabilities().await,
        }
    }

    pub async fn invoke(&self, request: InvokeRequest) -> InvokeResult {
        let started_at = Instant::now();
        let manifest = self
            .inner
            .capabilities
            .read()
            .await
            .get(&request.capability_id)
            .cloned();
        let Some(manifest) = manifest else {
            return InvokeResult::failed(
                &request,
                SensError {
                    code: "capability_not_found".into(),
                    message: format!("Unknown capability: {}", request.capability_id),
                    recoverable: false,
                    action: None,
                },
                elapsed_ms(started_at.elapsed()),
            );
        };
        if !manifest
            .operations
            .iter()
            .any(|operation| operation == &request.operation)
        {
            return InvokeResult::failed(
                &request,
                SensError {
                    code: "operation_not_supported".into(),
                    message: format!(
                        "Capability {} does not support operation {}",
                        request.capability_id, request.operation
                    ),
                    recoverable: false,
                    action: None,
                },
                elapsed_ms(started_at.elapsed()),
            );
        }
        if manifest.state == CapabilityState::Disabled {
            return InvokeResult::failed(
                &request,
                SensError {
                    code: "capability_disabled".into(),
                    message: format!("Capability {} is disabled", request.capability_id),
                    recoverable: true,
                    action: Some("Enable the capability in Sens settings.".into()),
                },
                elapsed_ms(started_at.elapsed()),
            );
        }

        let executor = self
            .inner
            .executors
            .read()
            .await
            .get(&request.capability_id)
            .cloned();
        let Some(executor) = executor else {
            return InvokeResult::failed(
                &request,
                SensError {
                    code: "capability_unavailable".into(),
                    message: format!(
                        "Capability {} has no available runtime",
                        request.capability_id
                    ),
                    recoverable: true,
                    action: Some("Open Sens diagnostics and repair the capability runtime.".into()),
                },
                elapsed_ms(started_at.elapsed()),
            );
        };

        self.set_capability_state(&request.capability_id, CapabilityState::Busy)
            .await;
        let result = executor.invoke(&request).await;
        match result {
            Ok(output) => {
                self.set_capability_state(&request.capability_id, CapabilityState::Ready)
                    .await;
                InvokeResult {
                    request_id: request.request_id,
                    capability_id: request.capability_id,
                    operation: request.operation,
                    status: JobState::Succeeded,
                    data: output.data,
                    artifacts: output.artifacts,
                    provenance: output.provenance,
                    usage: output.usage,
                    warnings: output.warnings,
                    error: None,
                    elapsed_ms: elapsed_ms(started_at.elapsed()),
                }
            }
            Err(error) => {
                let next_state = if error.recoverable {
                    CapabilityState::Asleep
                } else {
                    CapabilityState::Error
                };
                self.set_capability_state(&request.capability_id, next_state)
                    .await;
                InvokeResult::failed(&request, error, elapsed_ms(started_at.elapsed()))
            }
        }
    }
}

fn elapsed_ms(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Echo;

    struct RecoverableFailure;

    #[async_trait]
    impl CapabilityExecutor for Echo {
        async fn invoke(&self, request: &InvokeRequest) -> Result<CapabilityOutput, SensError> {
            Ok(CapabilityOutput {
                data: request.input.clone(),
                ..Default::default()
            })
        }
    }

    #[async_trait]
    impl CapabilityExecutor for RecoverableFailure {
        async fn invoke(&self, _request: &InvokeRequest) -> Result<CapabilityOutput, SensError> {
            Err(SensError {
                code: "temporary_failure".into(),
                message: "retry later".into(),
                recoverable: true,
                action: Some("Retry the request.".into()),
            })
        }
    }

    #[tokio::test]
    async fn invokes_registered_capability() {
        let core = SensCore::new();
        core.register_executor("sight", Arc::new(Echo))
            .await
            .expect("register");
        core.mark_ready().await;
        let result = core
            .invoke(InvokeRequest::new(
                "sight",
                "read",
                json!({ "imagePath": "a.png" }),
            ))
            .await;
        assert_eq!(result.status, JobState::Succeeded);
        assert_eq!(result.data["imagePath"], "a.png");
    }

    #[tokio::test]
    async fn rejects_unknown_operation_without_calling_worker() {
        let core = SensCore::new();
        core.register_executor("sight", Arc::new(Echo))
            .await
            .expect("register");
        let result = core
            .invoke(InvokeRequest::new("sight", "invent", Value::Null))
            .await;
        assert_eq!(result.status, JobState::Failed);
        assert_eq!(result.error.expect("error").code, "operation_not_supported");
    }

    #[tokio::test]
    async fn recoverable_worker_failure_returns_capability_to_sleep() {
        let core = SensCore::new();
        core.register_executor("sight", Arc::new(RecoverableFailure))
            .await
            .expect("register");
        let result = core
            .invoke(InvokeRequest::new("sight", "read", json!({})))
            .await;
        assert_eq!(result.status, JobState::Failed);
        let sight = core
            .capabilities()
            .await
            .into_iter()
            .find(|capability| capability.id == "sight")
            .expect("sight");
        assert_eq!(sight.state, CapabilityState::Asleep);
    }
}
