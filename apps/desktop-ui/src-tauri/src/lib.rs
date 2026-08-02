mod settings;
mod speech_runtime;

use sens_broker::BrokerClient;
use sens_connect::{InstallResult, default_zcode_config_path, install};
use sens_protocol::{BrokerRequest, BrokerResponse, CapabilityManifest, StatusSnapshot};
use serde_json::Value;
use settings::CapabilitySettings;
use speech_runtime::{SpeechRuntime, SpeechRuntimeStatus};
use tauri::{
    AppHandle, Emitter, Manager, State,
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
};
use tauri_plugin_positioner::{Position, WindowExt};

#[tauri::command]
async fn sens_status() -> Result<StatusSnapshot, String> {
    let client = BrokerClient::new();
    client.ensure_running().await.map_err(display_error)?;
    match client
        .request(BrokerRequest::Status)
        .await
        .map_err(display_error)?
    {
        BrokerResponse::Status { status } => Ok(status),
        BrokerResponse::Error { error } => Err(error.message),
        _ => Err("Sens broker returned an unexpected status response".into()),
    }
}

#[tauri::command]
async fn sens_capabilities() -> Result<Vec<CapabilityManifest>, String> {
    let client = BrokerClient::new();
    client.ensure_running().await.map_err(display_error)?;
    match client
        .request(BrokerRequest::Capabilities)
        .await
        .map_err(display_error)?
    {
        BrokerResponse::Capabilities { capabilities } => Ok(capabilities),
        BrokerResponse::Error { error } => Err(error.message),
        _ => Err("Sens broker returned an unexpected capabilities response".into()),
    }
}

#[tauri::command]
fn capability_settings() -> Result<CapabilitySettings, String> {
    settings::load()
}

#[tauri::command]
fn save_capability_settings(
    capability: String,
    settings: Value,
    speech: State<'_, SpeechRuntime>,
) -> Result<CapabilitySettings, String> {
    let saved = settings::save(&capability, settings)?;
    if capability == "hearing" {
        speech.sync_hearing_settings(&saved.hearing)?;
    }
    Ok(saved)
}

#[tauri::command]
fn speech_runtime_status(speech: State<'_, SpeechRuntime>) -> SpeechRuntimeStatus {
    speech.status()
}

#[tauri::command]
fn start_speech_runtime(speech: State<'_, SpeechRuntime>) -> Result<SpeechRuntimeStatus, String> {
    speech.ensure_started()
}

#[tauri::command]
fn quit_app(app: AppHandle, speech: State<'_, SpeechRuntime>) {
    speech.stop();
    app.exit(0);
}

#[tauri::command]
fn connect_client(client: String) -> Result<InstallResult, String> {
    if client != "Z-Code" {
        return Err(format!("Automatic setup for {client} is not available yet"));
    }
    let config = default_zcode_config_path()?;
    let executable = std::env::current_exe().map_err(display_error)?;
    let binary_dir = executable
        .parent()
        .ok_or_else(|| "Sens executable has no parent directory".to_string())?;
    let mcp = binary_dir.join(if cfg!(windows) {
        "sens-mcp.exe"
    } else {
        "sens-mcp"
    });
    let sens_root = executable
        .ancestors()
        .find(|path| path.join("sidecars").is_dir())
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| binary_dir.to_path_buf());
    let eye_root = std::env::var_os("SENS_EYE_ROOT")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("USERPROFILE").map(|home| {
                std::path::PathBuf::from(home)
                    .join(".zcode")
                    .join("workspace")
                    .join("default")
                    .join("eye")
            })
        })
        .ok_or_else(|| "Could not discover Eye; set SENS_EYE_ROOT".to_string())?;
    let speech_root = std::env::var_os("SENS_SPEECH_ROOT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"D:\Speech"));
    install(&config, &mcp, &sens_root, &eye_root, &speech_root)
}

#[tauri::command]
fn show_main(app: AppHandle, view: Option<String>) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "Sens main window is unavailable".to_string())?;
    window.show().map_err(display_error)?;
    let _ = window.unminimize();
    window.set_focus().map_err(display_error)?;
    if let Some(view) = view {
        app.emit_to("main", "sens:navigate", view)
            .map_err(display_error)?;
    }
    if let Some(tray) = app.get_webview_window("tray") {
        let _ = tray.hide();
    }
    Ok(())
}

#[tauri::command]
fn window_action(app: AppHandle, action: String) -> Result<(), String> {
    let window = app
        .get_webview_window("main")
        .ok_or_else(|| "Sens main window is unavailable".to_string())?;
    match action.as_str() {
        "minimize" => window.minimize().map_err(display_error),
        "maximize" => {
            if window.is_maximized().map_err(display_error)? {
                window.unmaximize().map_err(display_error)
            } else {
                window.maximize().map_err(display_error)
            }
        }
        "hide" | "close" => window.hide().map_err(display_error),
        _ => Err(format!("Unsupported window action: {action}")),
    }
}

#[tauri::command]
fn hide_tray(app: AppHandle) -> Result<(), String> {
    app.get_webview_window("tray")
        .ok_or_else(|| "Sens tray window is unavailable".to_string())?
        .hide()
        .map_err(display_error)
}

fn display_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

fn reveal_main(app: &AppHandle, view: &str) {
    let _ = show_main(app.clone(), Some(view.to_string()));
}

fn toggle_tray_window(app: &AppHandle) {
    let Some(window) = app.get_webview_window("tray") else {
        return;
    };
    if window.is_visible().unwrap_or(false) {
        let _ = window.hide();
        return;
    }
    let _ = window.move_window_constrained(Position::TrayCenter);
    let _ = window.show();
    let _ = window.set_focus();
}

fn should_toggle_tray(button: MouseButton, button_state: MouseButtonState) -> bool {
    matches!(button_state, MouseButtonState::Up)
        && matches!(button, MouseButton::Left | MouseButton::Right)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            reveal_main(app, "home");
        }))
        .plugin(tauri_plugin_positioner::init())
        .invoke_handler(tauri::generate_handler![
            sens_status,
            sens_capabilities,
            capability_settings,
            save_capability_settings,
            speech_runtime_status,
            start_speech_runtime,
            connect_client,
            show_main,
            window_action,
            hide_tray,
            quit_app
        ])
        .setup(|app| {
            let speech = SpeechRuntime::discover();
            app.manage(speech.clone());
            std::thread::spawn(move || {
                if speech.ensure_started().is_ok()
                    && let Ok(saved) = settings::load()
                {
                    let _ = speech.sync_hearing_settings(&saved.hearing);
                }
            });

            let tray = TrayIconBuilder::with_id("sens-tray")
                .icon(
                    app.default_window_icon()
                        .cloned()
                        .ok_or("missing Sens icon")?,
                )
                .tooltip("Sens — чувства для моделей")
                .show_menu_on_left_click(false)
                .on_tray_icon_event(|tray, event| {
                    tauri_plugin_positioner::on_tray_event(tray.app_handle(), &event);
                    if let TrayIconEvent::Click {
                        button,
                        button_state,
                        ..
                    } = event
                    {
                        if should_toggle_tray(button, button_state) {
                            toggle_tray_window(tray.app_handle());
                        }
                    }
                })
                .build(app)?;
            app.manage(tray);
            Ok(())
        })
        .on_window_event(|window, event| match event {
            tauri::WindowEvent::CloseRequested { api, .. } if window.label() == "main" => {
                api.prevent_close();
                let _ = window.hide();
            }
            tauri::WindowEvent::Focused(false) if window.label() == "tray" => {
                let _ = window.hide();
            }
            _ => {}
        })
        .run(tauri::generate_context!())
        .expect("error while running Sens");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_tray_panel_opens_on_left_and_right_release() {
        assert!(should_toggle_tray(
            MouseButton::Left,
            MouseButtonState::Up
        ));
        assert!(should_toggle_tray(
            MouseButton::Right,
            MouseButtonState::Up
        ));
        assert!(!should_toggle_tray(
            MouseButton::Right,
            MouseButtonState::Down
        ));
    }
}
