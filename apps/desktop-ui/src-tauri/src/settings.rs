use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilitySettings {
    pub sight: SightSettings,
    pub hearing: HearingSettings,
    pub sight_providers: Vec<SightProviderOption>,
    pub hearing_models: Vec<HearingModelOption>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SightSettings {
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub detail: String,
    pub cache: bool,
    pub max_calls_per_image: u64,
    pub verify: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HearingSettings {
    pub enabled: bool,
    pub model: String,
    pub device: String,
    pub hotkey: String,
    pub copy_to_clipboard: bool,
    pub paste_to_active_input: bool,
    pub suppress_hotkey: bool,
    pub preload_model: bool,
    pub beam_size: u64,
    pub postprocess_text: bool,
    pub vad_sensitivity: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SightProviderOption {
    pub value: String,
    pub label: String,
    pub model: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct HearingModelOption {
    pub value: String,
    pub label: String,
    pub description: String,
}

pub fn load() -> Result<CapabilitySettings, String> {
    load_from_paths(&eye_config_path()?, &speech_settings_path())
}

pub fn save(capability: &str, settings: Value) -> Result<CapabilitySettings, String> {
    let eye_path = eye_config_path()?;
    let speech_path = speech_settings_path();
    match capability {
        "sight" => save_sight_at(
            &eye_path,
            serde_json::from_value(settings)
                .map_err(|error| format!("Некорректные настройки зрения: {error}"))?,
        )?,
        "hearing" => save_hearing_at(
            &speech_path,
            serde_json::from_value(settings)
                .map_err(|error| format!("Некорректные настройки слуха: {error}"))?,
        )?,
        _ => return Err(format!("Неизвестная возможность: {capability}")),
    }
    load_from_paths(&eye_path, &speech_path)
}

fn load_from_paths(eye_path: &Path, speech_path: &Path) -> Result<CapabilitySettings, String> {
    let eye = read_json(eye_path)?;
    let speech = read_json_or_default(speech_path)?;
    let provider = eye
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or("mimo")
        .to_owned();
    let providers = eye
        .get("providers")
        .and_then(Value::as_object)
        .ok_or_else(|| "Eye config is missing providers".to_string())?;
    let sight_providers = providers
        .iter()
        .map(|(value, item)| SightProviderOption {
            value: value.clone(),
            label: provider_label(value),
            model: item
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        })
        .collect::<Vec<_>>();
    let model = providers
        .get(&provider)
        .and_then(|item| item.get("model"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let vision = eye.get("vision").and_then(Value::as_object);

    Ok(CapabilitySettings {
        sight: SightSettings {
            enabled: vision
                .and_then(|item| item.get("enabled"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
            provider,
            model,
            detail: vision
                .and_then(|item| item.get("detail"))
                .and_then(Value::as_str)
                .unwrap_or("normal")
                .to_owned(),
            cache: vision
                .and_then(|item| item.get("cache"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
            max_calls_per_image: vision
                .and_then(|item| item.get("maxCallsPerImage"))
                .and_then(Value::as_u64)
                .unwrap_or(8),
            verify: vision
                .and_then(|item| item.get("verify"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        },
        hearing: HearingSettings {
            enabled: bool_field(&speech, "engine_enabled", true),
            model: string_field(&speech, "model", "parakeet"),
            device: string_field(&speech, "device", "cpu"),
            hotkey: string_field(&speech, "hotkey", "ctrl+win"),
            copy_to_clipboard: bool_field(&speech, "copy_to_clipboard", true),
            paste_to_active_input: bool_field(&speech, "paste_to_active_input", true),
            suppress_hotkey: bool_field(&speech, "suppress_hotkey", false),
            preload_model: bool_field(&speech, "preload_model", true),
            beam_size: u64_field(&speech, "beam_size", 5),
            postprocess_text: bool_field(&speech, "postprocess_text", true),
            vad_sensitivity: speech
                .get("vad_sensitivity")
                .and_then(Value::as_f64)
                .unwrap_or(0.02),
        },
        sight_providers,
        hearing_models: hearing_models(),
    })
}

fn save_sight_at(path: &Path, settings: SightSettings) -> Result<(), String> {
    validate_sight(&settings)?;
    let mut document = read_json(path)?;
    let root = object_mut(&mut document, "Eye config")?;
    let providers = root
        .get_mut("providers")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| "Eye config is missing providers".to_string())?;
    let provider = providers
        .get_mut(&settings.provider)
        .and_then(Value::as_object_mut)
        .ok_or_else(|| format!("Eye provider {} is unavailable", settings.provider))?;
    provider.insert(
        "model".into(),
        Value::String(settings.model.trim().to_owned()),
    );

    root.insert("provider".into(), Value::String(settings.provider));
    let vision = root
        .entry("vision")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "Eye vision settings must be an object".to_string())?;
    vision.insert("enabled".into(), Value::Bool(settings.enabled));
    vision.insert("detail".into(), Value::String(settings.detail));
    vision.insert("cache".into(), Value::Bool(settings.cache));
    vision.insert(
        "maxCallsPerImage".into(),
        Value::Number(settings.max_calls_per_image.into()),
    );
    vision.insert("verify".into(), Value::Bool(settings.verify));
    write_json(path, &document)
}

fn save_hearing_at(path: &Path, settings: HearingSettings) -> Result<(), String> {
    validate_hearing(&settings)?;
    let mut document = read_json_or_default(path)?;
    let root = object_mut(&mut document, "Speech settings")?;
    root.insert("engine_enabled".into(), Value::Bool(settings.enabled));
    root.insert("model".into(), Value::String(settings.model.clone()));
    root.insert(
        "model_id".into(),
        Value::String(hearing_model_id(&settings.model)?.to_owned()),
    );
    root.insert("device".into(), Value::String(settings.device));
    root.insert("hotkey".into(), Value::String(settings.hotkey));
    root.insert(
        "copy_to_clipboard".into(),
        Value::Bool(settings.copy_to_clipboard),
    );
    root.insert(
        "paste_to_active_input".into(),
        Value::Bool(settings.paste_to_active_input),
    );
    root.insert(
        "suppress_hotkey".into(),
        Value::Bool(settings.suppress_hotkey),
    );
    root.insert("preload_model".into(), Value::Bool(settings.preload_model));
    root.insert("beam_size".into(), Value::Number(settings.beam_size.into()));
    root.insert(
        "postprocess_text".into(),
        Value::Bool(settings.postprocess_text),
    );
    root.insert("vad_sensitivity".into(), json!(settings.vad_sensitivity));
    write_json(path, &document)
}

fn validate_sight(settings: &SightSettings) -> Result<(), String> {
    if settings.model.trim().is_empty() {
        return Err("Название vision-модели не может быть пустым".into());
    }
    if !matches!(settings.detail.as_str(), "quick" | "normal" | "deep") {
        return Err("Детализация должна быть quick, normal или deep".into());
    }
    if !(1..=32).contains(&settings.max_calls_per_image) {
        return Err("Лимит вызовов должен быть от 1 до 32".into());
    }
    Ok(())
}

fn validate_hearing(settings: &HearingSettings) -> Result<(), String> {
    hearing_model_id(&settings.model)?;
    let hotkey_parts = settings
        .hotkey
        .split('+')
        .filter(|part| !part.trim().is_empty())
        .count();
    if !(2..=3).contains(&hotkey_parts) {
        return Err("Горячая клавиша должна содержать два или три сочетания".into());
    }
    if !matches!(settings.device.as_str(), "auto" | "cpu" | "cuda") {
        return Err("Устройство должно быть auto, cpu или cuda".into());
    }
    if !(1..=10).contains(&settings.beam_size) {
        return Err("Beam size должен быть от 1 до 10".into());
    }
    if !(0.001..=0.1).contains(&settings.vad_sensitivity) {
        return Err("Чувствительность VAD должна быть от 0.001 до 0.1".into());
    }
    Ok(())
}

fn eye_config_path() -> Result<PathBuf, String> {
    if let Some(root) = std::env::var_os("SENS_EYE_ROOT") {
        return Ok(PathBuf::from(root).join("config.json"));
    }
    let home = std::env::var_os("USERPROFILE")
        .ok_or_else(|| "Не удалось найти Eye: USERPROFILE недоступен".to_string())?;
    Ok(PathBuf::from(home)
        .join(".zcode")
        .join("workspace")
        .join("default")
        .join("eye")
        .join("config.json"))
}

fn speech_settings_path() -> PathBuf {
    std::env::var_os("SENS_SPEECH_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"D:\Speech"))
        .join("data")
        .join("settings.json")
}

fn read_json(path: &Path) -> Result<Value, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("Не удалось прочитать {}: {error}", path.display()))?;
    serde_json::from_str(&contents)
        .map_err(|error| format!("{} содержит некорректный JSON: {error}", path.display()))
}

fn read_json_or_default(path: &Path) -> Result<Value, String> {
    if path.is_file() {
        read_json(path)
    } else {
        Ok(json!({}))
    }
}

fn write_json(path: &Path, document: &Value) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Не удалось создать {}: {error}", parent.display()))?;
    }
    if path.is_file() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_millis();
        let name = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("settings");
        let backup = path.with_file_name(format!("{name}.sens-settings-backup-{stamp}.json"));
        fs::copy(path, &backup)
            .map_err(|error| format!("Не удалось создать резервную копию: {error}"))?;
    }
    let encoded = serde_json::to_vec_pretty(document).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.sens.tmp");
    fs::write(&temporary, encoded)
        .map_err(|error| format!("Не удалось записать {}: {error}", temporary.display()))?;
    replace_file(&temporary, path)
        .map_err(|error| format!("Не удалось обновить {}: {error}", path.display()))
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };

    let source: Vec<u16> = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    let moved = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if moved == 0 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> std::io::Result<()> {
    fs::rename(source, destination)
}

fn object_mut<'a>(value: &'a mut Value, label: &str) -> Result<&'a mut Map<String, Value>, String> {
    value
        .as_object_mut()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn bool_field(document: &Value, key: &str, default: bool) -> bool {
    document
        .get(key)
        .and_then(Value::as_bool)
        .unwrap_or(default)
}

fn string_field(document: &Value, key: &str, default: &str) -> String {
    document
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_owned()
}

fn u64_field(document: &Value, key: &str, default: u64) -> u64 {
    document.get(key).and_then(Value::as_u64).unwrap_or(default)
}

fn provider_label(value: &str) -> String {
    match value {
        "mimo" => "MiMo".into(),
        "openai" => "OpenAI".into(),
        "custom" => "Свой провайдер".into(),
        other => other.to_owned(),
    }
}

fn hearing_models() -> Vec<HearingModelOption> {
    vec![
        HearingModelOption {
            value: "parakeet".into(),
            label: "Parakeet · быстрая".into(),
            description: "600M, мультиязычная, быстрая на CPU".into(),
        },
        HearingModelOption {
            value: "whisper-ru".into(),
            label: "Whisper RU · точная".into(),
            description: "RU + EN code-switching, высокая точность".into(),
        },
        HearingModelOption {
            value: "gigaam".into(),
            label: "GigaAM v3 · русский".into(),
            description: "230M, локальная русская модель с пунктуацией".into(),
        },
    ]
}

fn hearing_model_id(value: &str) -> Result<&'static str, String> {
    match value {
        "parakeet" => Ok("nvidia/parakeet-tdt-0.6b-v3"),
        "whisper-ru" => Ok("coriollon/whisper-large-v3-turbo-russian-codeswitch"),
        "gigaam" => Ok("ai-sage/GigaAM-v3"),
        _ => Err(format!("Неизвестная модель слуха: {value}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sight_save_preserves_provider_secret_and_unrelated_fields() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("config.json");
        fs::write(
            &path,
            r#"{"provider":"mimo","providers":{"mimo":{"apiKey":"secret","model":"old"}},"vision":{"detail":"quick"},"keep":7}"#,
        )
        .expect("fixture");
        save_sight_at(
            &path,
            SightSettings {
                enabled: true,
                provider: "mimo".into(),
                model: "mimo-v2.5".into(),
                detail: "normal".into(),
                cache: true,
                max_calls_per_image: 6,
                verify: true,
            },
        )
        .expect("save");
        let document = read_json(&path).expect("document");
        assert_eq!(document["providers"]["mimo"]["apiKey"], "secret");
        assert_eq!(document["vision"]["maxCallsPerImage"], 6);
        assert_eq!(document["vision"]["verify"], true);
        assert_eq!(document["keep"], 7);
    }

    #[test]
    fn hearing_save_maps_model_and_preserves_unrelated_fields() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("settings.json");
        fs::write(&path, r#"{"model":"parakeet","keep":9}"#).expect("fixture");
        save_hearing_at(
            &path,
            HearingSettings {
                enabled: true,
                model: "gigaam".into(),
                device: "cpu".into(),
                hotkey: "ctrl+win".into(),
                copy_to_clipboard: true,
                paste_to_active_input: true,
                suppress_hotkey: false,
                preload_model: false,
                beam_size: 4,
                postprocess_text: true,
                vad_sensitivity: 0.03,
            },
        )
        .expect("save");
        let document = read_json(&path).expect("document");
        assert_eq!(document["model"], "gigaam");
        assert_eq!(document["model_id"], "ai-sage/GigaAM-v3");
        assert_eq!(document["keep"], 9);
    }

    #[test]
    fn hearing_settings_expose_interactive_dictation_contract() {
        let temp = tempfile::tempdir().expect("tempdir");
        let eye_path = temp.path().join("eye.json");
        let speech_path = temp.path().join("speech.json");
        fs::write(
            &eye_path,
            r#"{"provider":"mimo","providers":{"mimo":{"model":"mimo-v2.5"}}}"#,
        )
        .expect("eye fixture");
        fs::write(
            &speech_path,
            r#"{"hotkey":"ctrl+win","copy_to_clipboard":true,"paste_to_active_input":true,"suppress_hotkey":false}"#,
        )
        .expect("speech fixture");

        let settings = load_from_paths(&eye_path, &speech_path)
            .expect("settings")
            .hearing;
        let value = serde_json::to_value(settings).expect("serialize");

        assert_eq!(value["hotkey"], "ctrl+win");
        assert_eq!(value["copyToClipboard"], true);
        assert_eq!(value["pasteToActiveInput"], true);
        assert_eq!(value["suppressHotkey"], false);
    }
}
