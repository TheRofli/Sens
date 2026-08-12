#[cfg(windows)]
mod windows;

mod hearing;
#[cfg(windows)]
mod process_group;
mod sight;
mod touch;

use std::sync::Arc;

use anyhow::Context;
use sens_core::{RuntimePaths, SensCore};
use sens_protocol::{BrokerRequest, BrokerResponse, PROTOCOL_VERSION, SensError};

pub use hearing::{HearingExecutor, HearingRuntimeConfig};
pub use sight::{SightExecutor, SightRuntimeConfig};
pub use touch::{TouchExecutor, TouchRuntimeConfig};

#[cfg(windows)]
pub use windows::{BrokerClient, BrokerServer, DEFAULT_PIPE_NAME, is_broker_already_running};

pub async fn build_core() -> anyhow::Result<SensCore> {
    let paths = RuntimePaths::discover();
    paths.ensure().context("create Sens runtime directories")?;

    let core = SensCore::new();
    let sight = SightExecutor::new(SightRuntimeConfig::discover()?);
    core.register_executor("sight", Arc::new(sight))
        .await
        .map_err(|error| anyhow::anyhow!(error.message))?;
    let hearing = HearingExecutor::new(HearingRuntimeConfig::discover()?);
    core.register_executor("hearing", Arc::new(hearing))
        .await
        .map_err(|error| anyhow::anyhow!(error.message))?;
    let touch_config = TouchRuntimeConfig::discover()?;
    let touch_enabled = touch_config.enabled;
    let touch = TouchExecutor::new(touch_config);
    core.register_executor(sens_protocol::touch::TOUCH_CAPABILITY_ID, Arc::new(touch))
        .await
        .map_err(|error| anyhow::anyhow!(error.message))?;
    core.set_capability_state(
        sens_protocol::touch::TOUCH_CAPABILITY_ID,
        if touch_enabled {
            sens_protocol::CapabilityState::Asleep
        } else {
            sens_protocol::CapabilityState::Disabled
        },
    )
    .await;
    core.mark_ready().await;
    Ok(core)
}

pub async fn handle_request(core: &SensCore, request: BrokerRequest) -> BrokerResponse {
    match request {
        BrokerRequest::Status => BrokerResponse::Status {
            status: core.status().await,
        },
        BrokerRequest::Capabilities => BrokerResponse::Capabilities {
            capabilities: core.capabilities().await,
        },
        BrokerRequest::Invoke { request } => BrokerResponse::Invoke {
            result: core.invoke(request).await,
        },
        BrokerRequest::Ping => BrokerResponse::Pong {
            protocol_version: PROTOCOL_VERSION.into(),
        },
        // Intercepted in the server loop before dispatch; kept for exhaustive
        // matching so a client that sends Shutdown still gets an answer.
        BrokerRequest::Shutdown => BrokerResponse::Pong {
            protocol_version: PROTOCOL_VERSION.into(),
        },
    }
}

pub fn protocol_error(message: impl Into<String>) -> BrokerResponse {
    BrokerResponse::Error {
        error: SensError {
            code: "broker_protocol_error".into(),
            message: message.into(),
            recoverable: true,
            action: Some("Restart Sens and retry the connection.".into()),
        },
    }
}
