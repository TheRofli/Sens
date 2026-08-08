use sens_broker::BrokerClient;
use sens_protocol::{BrokerRequest, BrokerResponse, InvokeRequest, JobState};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::settings::HearingSettings;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SpeechRuntimeStatus {
    pub running: bool,
    pub managed: bool,
    #[serde(alias = "engine_enabled")]
    pub enabled: bool,
    pub hotkey: String,
    pub model: String,
    #[serde(alias = "model_state")]
    pub model_state: String,
    #[serde(alias = "model_loaded")]
    pub model_loaded: bool,
    #[serde(alias = "model_loading")]
    pub model_loading: bool,
    pub transcribing: bool,
    #[serde(alias = "model_installed")]
    pub model_installed: bool,
    #[serde(alias = "model_size_mb")]
    pub model_size_mb: f64,
    pub installing: bool,
    #[serde(alias = "install_phase")]
    pub install_phase: String,
    #[serde(alias = "install_bytes_present")]
    pub install_bytes_present: u64,
    #[serde(alias = "install_bytes_required")]
    pub install_bytes_required: u64,
    #[serde(alias = "install_error")]
    pub install_error: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct HearingModelStatus {
    pub model: String,
    #[serde(alias = "model_installed")]
    pub model_installed: bool,
    #[serde(alias = "model_size_mb")]
    pub model_size_mb: f64,
    pub installing: bool,
    #[serde(alias = "install_phase")]
    pub install_phase: String,
    #[serde(alias = "install_bytes_present")]
    pub install_bytes_present: u64,
    #[serde(alias = "install_bytes_required")]
    pub install_bytes_required: u64,
    #[serde(alias = "install_error")]
    pub install_error: Option<String>,
}

pub async fn status() -> SpeechRuntimeStatus {
    invoke("dictation_status", json!({}), 5_000)
        .await
        .unwrap_or_else(|error| SpeechRuntimeStatus {
            hotkey: "ctrl+win".into(),
            model_state: "stopped".into(),
            error: Some(error),
            ..Default::default()
        })
}

pub async fn start(settings: &HearingSettings) -> Result<SpeechRuntimeStatus, String> {
    invoke("dictation_start", settings_payload(settings), 15_000).await
}

pub async fn sync_settings(settings: &HearingSettings) -> Result<SpeechRuntimeStatus, String> {
    invoke("dictation_settings", settings_payload(settings), 10_000).await
}

pub async fn model_status(model: &str) -> Result<HearingModelStatus, String> {
    invoke_model_status("model_status", model).await
}

pub async fn install_model(model: &str) -> Result<HearingModelStatus, String> {
    invoke_model_status("model_install", model).await
}

async fn invoke_model_status(operation: &str, model: &str) -> Result<HearingModelStatus, String> {
    let value = invoke_value(operation, json!({"model": model}), 10_000).await?;
    serde_json::from_value(value)
        .map_err(|error| format!("Hearing returned an invalid model status: {error}"))
}

fn settings_payload(settings: &HearingSettings) -> Value {
    json!({
        "model": settings.model,
        "engine_enabled": settings.enabled,
        "device": settings.device,
        "hotkey": settings.hotkey,
        "copy_to_clipboard": settings.copy_to_clipboard,
        "paste_to_active_input": settings.paste_to_active_input,
        "suppress_hotkey": settings.suppress_hotkey,
        "preload_model": settings.preload_model,
        "beam_size": settings.beam_size,
        "postprocess_text": settings.postprocess_text,
        "vad_sensitivity": settings.vad_sensitivity,
        "max_frames": settings.max_frames,
        "frame_size": settings.frame_size,
        "default_every": settings.default_every,
        // Secrets stay inside the authenticated local named pipe and worker
        // stdin. They are never placed in argv, stdout diagnostics, or logs.
        "remote_api_key": settings.api_key,
        "remote_base_url": settings.api_base_url,
        "remote_model_id": settings.api_model_id,
    })
}

async fn invoke(
    operation: &str,
    input: Value,
    timeout_ms: u64,
) -> Result<SpeechRuntimeStatus, String> {
    let value = invoke_value(operation, input, timeout_ms).await?;
    serde_json::from_value(value)
        .map_err(|error| format!("Hearing returned an invalid runtime status: {error}"))
}

async fn invoke_value(operation: &str, input: Value, timeout_ms: u64) -> Result<Value, String> {
    let client = BrokerClient::new();
    client.ensure_running().await.map_err(display_error)?;
    let mut request = InvokeRequest::new("hearing", operation, input);
    request.no_store = true;
    request.timeout_ms = Some(timeout_ms);
    match client
        .request(BrokerRequest::Invoke { request })
        .await
        .map_err(display_error)?
    {
        BrokerResponse::Invoke { result } if result.status == JobState::Succeeded => {
            Ok(result.data)
        }
        BrokerResponse::Invoke { result } => Err(result
            .error
            .map(|error| error.message)
            .unwrap_or_else(|| "Hearing runtime request failed".into())),
        BrokerResponse::Error { error } => Err(error.message),
        _ => Err("Sens broker returned an unexpected Hearing response".into()),
    }
}

fn display_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_status_accepts_worker_snake_case_fields() {
        let status: SpeechRuntimeStatus = serde_json::from_value(json!({
            "running": true,
            "managed": true,
            "engine_enabled": true,
            "hotkey": "ctrl+win",
            "model": "gigaam",
            "model_state": "loaded",
            "model_loaded": true,
            "model_loading": false,
            "transcribing": false,
            "model_installed": true,
            "model_size_mb": 162.3,
            "install_phase": "ready",
            "install_bytes_present": 170197019,
            "install_bytes_required": 170197019,
            "install_error": null
        }))
        .expect("status");

        assert!(status.running);
        assert!(status.managed);
        assert!(status.enabled);
        assert_eq!(status.hotkey, "ctrl+win");
        assert_eq!(status.model, "gigaam");
        assert_eq!(status.model_state, "loaded");
        assert!(status.model_installed);
        assert_eq!(status.install_phase, "ready");
        assert_eq!(status.install_bytes_required, 170_197_019);
    }

    #[test]
    fn model_status_converts_worker_fields_for_the_ui() {
        let status: HearingModelStatus = serde_json::from_value(json!({
            "model": "qwen",
            "model_installed": false,
            "model_size_mb": 838.0,
            "installing": true,
            "install_phase": "downloading",
            "install_bytes_present": 123,
            "install_bytes_required": 878702423,
            "install_error": null
        }))
        .expect("status");

        let ui = serde_json::to_value(status).expect("ui json");
        assert_eq!(ui["modelInstalled"], false);
        assert_eq!(ui["installPhase"], "downloading");
        assert_eq!(ui["installBytesPresent"], 123);
        assert_eq!(ui["installBytesRequired"], 878_702_423_u64);
        assert!(ui.get("model_installed").is_none());
    }

    #[test]
    fn settings_payload_keeps_remote_secret_out_of_debug_strings() {
        let settings = HearingSettings {
            enabled: true,
            model: "remote".into(),
            device: "cpu".into(),
            hotkey: "ctrl+win".into(),
            copy_to_clipboard: true,
            paste_to_active_input: true,
            suppress_hotkey: false,
            preload_model: false,
            beam_size: 5,
            postprocess_text: true,
            vad_sensitivity: 0.02,
            max_frames: 12,
            frame_size: 640,
            default_every: 0.0,
            api_key: "test-secret".into(),
            api_base_url: "https://example.test/v1".into(),
            api_model_id: "test/model".into(),
        };

        let payload = settings_payload(&settings);
        assert_eq!(payload["remote_api_key"], "test-secret");
        assert!(!format!("{settings:?}").contains("test-secret"));
    }
}
