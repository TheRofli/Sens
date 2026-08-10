use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ZCodeConnectionStatus {
    pub configured: bool,
    pub enabled: bool,
    pub command: Option<String>,
    pub legacy_eye_enabled: bool,
    pub config_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct InstallResult {
    pub changed: bool,
    pub config_path: PathBuf,
    pub backup_path: Option<PathBuf>,
    pub status: ZCodeConnectionStatus,
}

pub fn default_zcode_config_path() -> Result<PathBuf, String> {
    let home = std::env::var_os("USERPROFILE").ok_or_else(|| {
        "USERPROFILE is unavailable; pass an explicit Z-Code config path".to_string()
    })?;
    Ok(PathBuf::from(home)
        .join(".zcode")
        .join("cli")
        .join("config.json"))
}

pub fn status(config_path: &Path) -> Result<ZCodeConnectionStatus, String> {
    let document = read_document(config_path)?;
    let servers = document.pointer("/mcp/servers").and_then(Value::as_object);
    let sens = servers.and_then(|items| items.get("sens"));
    let eye = servers.and_then(|items| items.get("eye"));
    Ok(ZCodeConnectionStatus {
        configured: sens.is_some(),
        enabled: sens
            .and_then(|value| value.get("enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        command: sens
            .and_then(|value| value.get("command"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        legacy_eye_enabled: eye
            .and_then(|value| value.get("enabled"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        config_path: config_path.to_path_buf(),
    })
}

pub fn install(
    config_path: &Path,
    sens_mcp: &Path,
    sens_root: &Path,
    eye_root: &Path,
    speech_root: &Path,
) -> Result<InstallResult, String> {
    if !config_path.is_absolute() || !sens_mcp.is_absolute() {
        return Err("Z-Code config and Sens MCP paths must be absolute".into());
    }
    if !sens_mcp.is_file() {
        return Err(format!(
            "Sens MCP executable is missing: {}",
            sens_mcp.display()
        ));
    }
    let mut document = read_document(config_path)?;
    let servers = servers_mut(&mut document)?;
    let desired = json!({
        "type": "stdio",
        "command": sens_mcp,
        "args": [],
        "cwd": sens_root,
        "env": {
            "SENS_EYE_ROOT": eye_root,
            "SENS_SPEECH_ROOT": speech_root,
            "SENS_SIDECARS_ROOT": sens_root.join("sidecars")
        },
        "enabled": true,
        "timeoutMs": 900000
    });
    let already_current = servers.get("sens") == Some(&desired)
        && servers
            .get("eye")
            .and_then(|value| value.get("enabled"))
            .and_then(Value::as_bool)
            != Some(true);
    if already_current {
        return Ok(InstallResult {
            changed: false,
            config_path: config_path.to_path_buf(),
            backup_path: None,
            status: status(config_path)?,
        });
    }
    if let Some(eye) = servers.get_mut("eye").and_then(Value::as_object_mut) {
        eye.insert("enabled".into(), Value::Bool(false));
    }
    servers.insert("sens".into(), desired);
    let backup_path = backup(config_path)?;
    write_document(config_path, &document)?;
    Ok(InstallResult {
        changed: true,
        config_path: config_path.to_path_buf(),
        backup_path: Some(backup_path),
        status: status(config_path)?,
    })
}

pub fn uninstall(config_path: &Path) -> Result<InstallResult, String> {
    let mut document = read_document(config_path)?;
    let servers = servers_mut(&mut document)?;
    if !servers.contains_key("sens") {
        return Ok(InstallResult {
            changed: false,
            config_path: config_path.to_path_buf(),
            backup_path: None,
            status: status(config_path)?,
        });
    }
    servers.remove("sens");
    if let Some(eye) = servers.get_mut("eye").and_then(Value::as_object_mut) {
        eye.insert("enabled".into(), Value::Bool(true));
    }
    let backup_path = backup(config_path)?;
    write_document(config_path, &document)?;
    Ok(InstallResult {
        changed: true,
        config_path: config_path.to_path_buf(),
        backup_path: Some(backup_path),
        status: status(config_path)?,
    })
}

fn read_document(path: &Path) -> Result<Value, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("Could not read {}: {error}", path.display()))?;
    serde_json::from_str(&contents)
        .map_err(|error| format!("{} is not valid JSON: {error}", path.display()))
}

fn servers_mut(document: &mut Value) -> Result<&mut Map<String, Value>, String> {
    let root = document
        .as_object_mut()
        .ok_or_else(|| "Z-Code config root must be an object".to_string())?;
    let mcp = root
        .entry("mcp")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "Z-Code mcp config must be an object".to_string())?;
    mcp.entry("servers")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| "Z-Code mcp.servers must be an object".to_string())
}

fn backup(path: &Path) -> Result<PathBuf, String> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| error.to_string())?
        .as_secs();
    let backup = path.with_file_name(format!("config.sens-backup-{stamp}.json"));
    fs::copy(path, &backup)
        .map_err(|error| format!("Could not back up {}: {error}", path.display()))?;
    Ok(backup)
}

fn write_document(path: &Path, document: &Value) -> Result<(), String> {
    let encoded = serde_json::to_vec_pretty(document).map_err(|error| error.to_string())?;
    let temporary = path.with_extension("json.sens.tmp");
    fs::write(&temporary, encoded)
        .map_err(|error| format!("Could not write {}: {error}", temporary.display()))?;
    replace_file(&temporary, path)
        .map_err(|error| format!("Could not replace {}: {error}", path.display()))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(root: &Path) -> (PathBuf, PathBuf) {
        let config = root.join("config.json");
        fs::write(
            &config,
            r#"{"mcp":{"servers":{"eye":{"command":"node","enabled":true}}},"keep":7}"#,
        )
        .expect("write config");
        let executable = root.join("sens-mcp.exe");
        fs::write(&executable, b"fixture").expect("write executable");
        (config, executable)
    }

    #[test]
    fn install_is_reversible_and_preserves_unrelated_config() {
        let temp = tempfile::tempdir().expect("tempdir");
        let (config, executable) = fixture(temp.path());
        let installed = install(
            &config,
            &executable,
            temp.path(),
            &temp.path().join("eye"),
            &temp.path().join("speech"),
        )
        .expect("install");
        assert!(installed.changed);
        assert!(installed.backup_path.expect("backup").is_file());
        let document = read_document(&config).expect("document");
        assert_eq!(document["keep"], 7);
        assert_eq!(document["mcp"]["servers"]["eye"]["enabled"], false);
        assert_eq!(document["mcp"]["servers"]["sens"]["enabled"], true);
        assert_eq!(document["mcp"]["servers"]["sens"]["timeoutMs"], 900_000);

        let removed = uninstall(&config).expect("uninstall");
        assert!(removed.changed);
        let document = read_document(&config).expect("document");
        assert!(document["mcp"]["servers"].get("sens").is_none());
        assert_eq!(document["mcp"]["servers"]["eye"]["enabled"], true);
    }

    #[test]
    fn repeated_install_is_idempotent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let (config, executable) = fixture(temp.path());
        let arguments = (
            config.as_path(),
            executable.as_path(),
            temp.path(),
            temp.path(),
            temp.path(),
        );
        install(
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
        )
        .expect("first install");
        let second = install(
            arguments.0,
            arguments.1,
            arguments.2,
            arguments.3,
            arguments.4,
        )
        .expect("second install");
        assert!(!second.changed);
    }
}
