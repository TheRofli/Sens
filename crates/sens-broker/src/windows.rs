use std::{io::ErrorKind, process::Stdio, time::Duration};

use anyhow::{Context, bail};
use sens_core::SensCore;
use sens_protocol::{BrokerRequest, BrokerResponse};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::windows::named_pipe::{ClientOptions, NamedPipeServer, ServerOptions},
    process::Command,
    time::sleep,
};
use tracing::warn;

use crate::{handle_request, protocol_error};

pub const DEFAULT_PIPE_NAME: &str = r"\\.\pipe\sens-broker-v1";

pub struct BrokerServer {
    pipe_name: String,
    core: SensCore,
}

impl BrokerServer {
    pub fn new(core: SensCore) -> Self {
        Self {
            pipe_name: DEFAULT_PIPE_NAME.into(),
            core,
        }
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let mut server = ServerOptions::new()
            .first_pipe_instance(true)
            .create(&self.pipe_name)
            .with_context(|| format!("bind single Sens broker at {}", self.pipe_name))?;

        loop {
            server
                .connect()
                .await
                .context("accept broker pipe client")?;
            let connected = server;
            server = ServerOptions::new()
                .create(&self.pipe_name)
                .context("create next broker pipe instance")?;
            let core = self.core.clone();
            tokio::spawn(async move {
                if let Err(error) = serve_connection(connected, core).await {
                    warn!(%error, "broker client disconnected with an error");
                }
            });
        }
    }
}

async fn serve_connection(stream: NamedPipeServer, core: SensCore) -> anyhow::Result<()> {
    let (reader, mut writer) = tokio::io::split(stream);
    let mut lines = BufReader::new(reader).lines();
    while let Some(line) = lines.next_line().await.context("read broker request")? {
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<BrokerRequest>(&line) {
            Ok(request) => handle_request(&core, request).await,
            Err(error) => protocol_error(format!("Invalid broker request: {error}")),
        };
        let mut encoded = serde_json::to_vec(&response).context("encode broker response")?;
        encoded.push(b'\n');
        writer
            .write_all(&encoded)
            .await
            .context("write broker response")?;
        writer.flush().await.context("flush broker response")?;
    }
    Ok(())
}

pub struct BrokerClient {
    pipe_name: String,
}

impl Default for BrokerClient {
    fn default() -> Self {
        Self::new()
    }
}

impl BrokerClient {
    pub fn new() -> Self {
        Self {
            pipe_name: DEFAULT_PIPE_NAME.into(),
        }
    }

    pub async fn request(&self, request: BrokerRequest) -> anyhow::Result<BrokerResponse> {
        let stream = ClientOptions::new()
            .open(&self.pipe_name)
            .with_context(|| format!("connect to Sens broker at {}", self.pipe_name))?;
        let (reader, mut writer) = tokio::io::split(stream);
        let mut encoded = serde_json::to_vec(&request).context("encode broker request")?;
        encoded.push(b'\n');
        writer
            .write_all(&encoded)
            .await
            .context("write broker request")?;
        writer.flush().await.context("flush broker request")?;
        let mut lines = BufReader::new(reader).lines();
        let line = lines
            .next_line()
            .await
            .context("read broker response")?
            .context("Sens broker closed without a response")?;
        serde_json::from_str(&line).context("decode broker response")
    }

    pub async fn ensure_running(&self) -> anyhow::Result<()> {
        if self.request(BrokerRequest::Ping).await.is_ok() {
            return Ok(());
        }
        let current_exe = std::env::current_exe().context("locate sens-mcp executable")?;
        let broker_name = if cfg!(windows) {
            "sens-broker.exe"
        } else {
            "sens-broker"
        };
        let broker_exe = current_exe
            .parent()
            .context("Sens executable has no parent directory")?
            .join(broker_name);
        if !broker_exe.is_file() {
            bail!(
                "Sens broker is not running and {} is missing. Reinstall or rebuild Sens.",
                broker_exe.display()
            );
        }
        let mut command = Command::new(&broker_exe);
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(false);
        crate::process_group::hide_console(&mut command);
        command
            .spawn()
            .with_context(|| format!("start {}", broker_exe.display()))?;

        let mut last_error = None;
        for _ in 0..30 {
            match self.request(BrokerRequest::Ping).await {
                Ok(_) => return Ok(()),
                Err(error) => last_error = Some(error),
            }
            sleep(Duration::from_millis(100)).await;
        }
        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("Sens broker did not become ready")))
    }
}

pub fn is_broker_already_running(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|source| source.downcast_ref::<std::io::Error>())
        .any(|io_error| {
            io_error.kind() == ErrorKind::PermissionDenied || io_error.raw_os_error() == Some(231)
        })
}
