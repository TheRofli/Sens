#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use sens_broker::{BrokerServer, build_core, is_broker_already_running};
use tracing::{error, info};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let core = build_core().await?;
    info!(pid = std::process::id(), "Sens broker ready");
    if let Err(error) = BrokerServer::new(core).run().await {
        if is_broker_already_running(&error) {
            info!("Sens broker is already running");
            return Ok(());
        }
        error!(%error, "Sens broker failed");
        return Err(error);
    }
    Ok(())
}
