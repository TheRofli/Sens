use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

/// Snapshot returned to the React UI; mirrors the Python /api/status payload.
#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct StatusSnapshot {
    running: bool,
    engine_enabled: bool,
    model_state: String,
    model: String,
    model_label: String,
    model_loaded: bool,
    model_loading: bool,
    model_installed: bool,
    model_size_label: String,
    transcribing: bool,
    device: String,
    backend: String,
    status_text: String,
    // Kept for back-compat with the old Overview which reads these directly.
    history_count: usize,
    speech_root: String,
    model_snapshot: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct SettingsPayload {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    engine_enabled: Option<bool>,
    #[serde(default)]
    copy_to_clipboard: Option<bool>,
    #[serde(default)]
    paste_to_active_input: Option<bool>,
    #[serde(default)]
    preload_model: Option<bool>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    backend: Option<String>,
    #[serde(default)]
    hotkey: Option<String>,
    #[serde(default)]
    beam_size: Option<i64>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    repetition_penalty: Option<f64>,
    #[serde(default)]
    no_repeat_ngram_size: Option<i64>,
    #[serde(default)]
    vad_sensitivity: Option<f64>,
    #[serde(default)]
    compression_ratio_threshold: Option<f64>,
    #[serde(default)]
    log_prob_threshold: Option<f64>,
    #[serde(default)]
    postprocess_text: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    key: String,
    label: String,
    engine: String,
    model_id: String,
    description: String,
    installed: bool,
    size_label: String,
    active: bool,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "camelCase")]
struct HistoryItem {
    id: String,
    text: String,
}

// -----------------------------------------------------------------------------
// Discovery: where is the Python API?
// -----------------------------------------------------------------------------

fn speech_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from(r"D:\Speech"))
}

/// Read the port the Python core wrote to data/api.port.
fn api_base() -> Option<String> {
    let port_path = speech_root().join("data").join("api.port");
    let port_str = fs::read_to_string(port_path).ok()?;
    let port: u16 = port_str.trim().parse().ok()?;
    Some(format!("http://127.0.0.1:{port}"))
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(8))
        .build()
        .expect("failed to build reqwest client")
}

fn api_get<T: serde::de::DeserializeOwned>(path: &str) -> Result<T, String> {
    let base = api_base().ok_or_else(|| "Speech core is not running".to_string())?;
    let client = http_client();
    let response = client
        .get(format!("{base}{path}"))
        .send()
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("core returned status {}", response.status()));
    }
    response.json::<T>().map_err(|e| e.to_string())
}

fn api_post<T: serde::de::DeserializeOwned>(
    path: &str,
    body: &Value,
) -> Result<T, String> {
    let base = api_base().ok_or_else(|| "Speech core is not running".to_string())?;
    let client = http_client();
    let response = client
        .post(format!("{base}{path}"))
        .json(body)
        .send()
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("core returned status {}", response.status()));
    }
    response.json::<T>().map_err(|e| e.to_string())
}

// -----------------------------------------------------------------------------
// Tauri commands
// -----------------------------------------------------------------------------

#[tauri::command]
fn app_snapshot() -> Result<StatusSnapshot, String> {
    // Prefer the live API; fall back to an offline snapshot so the UI still
    // renders when the Python core is not running.
    if api_base().is_some() {
        let mut snapshot: StatusSnapshot = api_get("/api/status")?;
        // Augment with cheap derived fields the UI expects.
        let history: Vec<HistoryItem> = api_get("/api/history?limit=80").unwrap_or_default();
        snapshot.history_count = history.len();
        snapshot.speech_root = speech_root().display().to_string();
        snapshot.model_snapshot = String::new();
        return Ok(snapshot);
    }
    Ok(offline_snapshot())
}

#[tauri::command]
fn get_settings() -> Result<Value, String> {
    api_get::<Value>("/api/settings")
}

#[tauri::command]
fn save_settings(settings: SettingsPayload) -> Result<Value, String> {
    let body = serde_json::to_value(&settings).map_err(|e| e.to_string())?;
    api_post::<Value>("/api/settings", &body)
}

#[tauri::command]
fn get_models() -> Result<Vec<ModelInfo>, String> {
    api_get::<Vec<ModelInfo>>("/api/models")
}

#[tauri::command]
fn select_model(key: String) -> Result<Value, String> {
    api_post::<Value>("/api/model", &serde_json::json!({ "key": key }))
}

#[tauri::command]
fn load_model() -> Result<Value, String> {
    api_post::<Value>("/api/model/load", &serde_json::json!({}))
}

#[tauri::command]
fn unload_model() -> Result<Value, String> {
    api_post::<Value>("/api/model/unload", &serde_json::json!({}))
}

#[tauri::command]
fn install_model(key: String) -> Result<Value, String> {
    api_post::<Value>(
        "/api/model/install",
        &serde_json::json!({ "key": key }),
    )
}

#[tauri::command]
fn recent_history(limit: usize) -> Result<Vec<HistoryItem>, String> {
    api_get::<Vec<HistoryItem>>(&format!("/api/history?limit={limit}"))
}

#[tauri::command]
fn copy_history_item(id: String) -> Result<Value, String> {
    api_post::<Value>("/api/history/copy", &serde_json::json!({ "id": id }))
}

#[tauri::command]
fn copy_last() -> Result<Value, String> {
    api_post::<Value>("/api/action/copy_last", &serde_json::json!({}))
}

// Legacy runtime-action commands kept so existing UI buttons still work. They
// shell out to bin/speech.cmd (Windows) / bin/speech (Unix).
#[tauri::command]
fn speech_status() -> Result<String, String> {
    run_speech(["status"])
}

#[tauri::command]
fn speech_diagnose() -> Result<String, String> {
    run_speech(["diagnose"])
}

#[tauri::command]
fn speech_restart() -> Result<String, String> {
    run_speech(["restart"])
}

#[tauri::command]
fn speech_stop() -> Result<String, String> {
    run_speech(["stop"])
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            app_snapshot,
            get_settings,
            save_settings,
            get_models,
            select_model,
            load_model,
            unload_model,
            install_model,
            recent_history,
            copy_history_item,
            copy_last,
            speech_status,
            speech_diagnose,
            speech_restart,
            speech_stop,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Speech Tauri app");
}

// -----------------------------------------------------------------------------
// Offline fallback + CLI bridge
// -----------------------------------------------------------------------------

fn offline_snapshot() -> StatusSnapshot {
    StatusSnapshot {
        running: false,
        engine_enabled: false,
        model_state: "stopped".to_string(),
        model: "parakeet".to_string(),
        model_label: "Speech core offline".to_string(),
        model_loaded: false,
        model_loading: false,
        model_installed: false,
        model_size_label: "Not installed".to_string(),
        transcribing: false,
        device: "cpu".to_string(),
        backend: "auto".to_string(),
        status_text: "Start Speech with: speech".to_string(),
        history_count: 0,
        speech_root: speech_root().display().to_string(),
        model_snapshot: String::new(),
    }
}

fn run_speech<I, S>(args: I) -> Result<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let root = speech_root();
    let launcher = if cfg!(windows) {
        root.join("bin").join("speech.cmd")
    } else {
        root.join("bin").join("speech")
    };
    let output = if cfg!(windows) {
        Command::new("cmd")
            .arg("/C")
            .arg(&launcher)
            .args(args)
            .output()
            .map_err(|e| e.to_string())?
    } else {
        Command::new(&launcher)
            .args(args)
            .output()
            .map_err(|e| e.to_string())?
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{}{}", stdout, stderr);

    if output.status.success() {
        Ok(combined)
    } else {
        Err(combined)
    }
}
