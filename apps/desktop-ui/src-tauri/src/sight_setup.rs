use std::{
    path::{Path, PathBuf},
    process::Command,
};

use sens_broker::SightRuntimeConfig;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SightSetupStatus {
    pub pack: String,
    pub title: String,
    pub ready: bool,
    pub runtime_ready: bool,
    pub bytes_required: u64,
    pub bytes_present: u64,
    pub models_root: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct PackSpec {
    title: &'static str,
    text: &'static str,
    text_bytes: u64,
    mmproj: &'static str,
    mmproj_bytes: u64,
}

fn pack_spec(pack: &str) -> Result<PackSpec, String> {
    match pack {
        "lite" => Ok(PackSpec {
            title: "Qwen3-VL 2B",
            text: "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
            text_bytes: 1_107_409_952,
            mmproj: "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
            mmproj_bytes: 445_053_216,
        }),
        "quality" => Ok(PackSpec {
            title: "SmolVLM2 2.2B",
            text: "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
            text_bytes: 1_112_602_656,
            mmproj: "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
            mmproj_bytes: 592_523_200,
        }),
        "quality_large" => Ok(PackSpec {
            title: "Qwen2.5-VL 3B",
            text: "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            text_bytes: 1_929_901_408,
            mmproj: "mmproj-F16.gguf",
            mmproj_bytes: 1_338_428_256,
        }),
        _ => Err(format!("Unknown Sight pack: {pack}")),
    }
}

fn file_size(path: PathBuf) -> u64 {
    path.metadata().map(|metadata| metadata.len()).unwrap_or(0)
}

fn partial_path(path: &Path) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(".part");
    value.into()
}

fn downloaded_size(path: &Path) -> u64 {
    let partial = partial_path(path);
    if partial.is_file() {
        file_size(partial)
    } else {
        file_size(path.to_path_buf())
    }
}

pub fn status(pack: Option<String>) -> Result<SightSetupStatus, String> {
    let config = SightRuntimeConfig::discover().map_err(|error| error.to_string())?;
    let pack = pack.or(config.vision_pack).unwrap_or_else(|| "lite".into());
    let spec = pack_spec(&pack)?;
    let text_path = config.models_root.join(spec.text);
    let mmproj_path = config.models_root.join(spec.mmproj);
    let text_size = file_size(text_path.clone());
    let mmproj_size = file_size(mmproj_path.clone());
    let runtime_ready =
        config.python_executable.is_file() || config.python_executable.as_os_str() == "python";
    Ok(SightSetupStatus {
        pack,
        title: spec.title.into(),
        ready: text_size == spec.text_bytes && mmproj_size == spec.mmproj_bytes,
        runtime_ready,
        bytes_required: spec.text_bytes + spec.mmproj_bytes,
        bytes_present: downloaded_size(&text_path).min(spec.text_bytes)
            + downloaded_size(&mmproj_path).min(spec.mmproj_bytes),
        models_root: config.models_root,
    })
}

pub async fn install(pack: String) -> Result<SightSetupStatus, String> {
    pack_spec(&pack)?;
    let config = SightRuntimeConfig::discover().map_err(|error| error.to_string())?;
    let sens_root = config
        .local_worker
        .parent()
        .and_then(|sidecars| sidecars.parent())
        .ok_or_else(|| "Sight worker has no Sens root".to_string())?;
    let downloader = sens_root.join("scripts").join("download-vision-models.py");
    if !downloader.is_file() {
        return Err(format!(
            "Sight model installer is missing: {}",
            downloader.display()
        ));
    }
    std::fs::create_dir_all(&config.models_root)
        .map_err(|error| format!("Could not create {}: {error}", config.models_root.display()))?;
    let python = config.python_executable;
    let models_root = config.models_root;
    let requested_pack = pack.clone();
    let output = tauri::async_runtime::spawn_blocking(move || {
        let mut command = Command::new(python);
        command
            .arg(downloader)
            .arg("--pack")
            .arg(requested_pack)
            .env("SENS_MODELS_ROOT", models_root)
            .env("PYTHONNOUSERSITE", "1");
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x0800_0000);
        }
        command.output()
    })
    .await
    .map_err(|error| format!("Sight model installer task failed: {error}"))?
    .map_err(|error| format!("Could not start Sight model installer: {error}"))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(if detail.is_empty() {
            format!("Sight model installer exited with {}", output.status)
        } else {
            detail
        });
    }
    status(Some(pack))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_pack_metadata_matches_the_verified_qwen_files() {
        let spec = pack_spec("lite").expect("lite pack");

        assert_eq!(spec.title, "Qwen3-VL 2B");
        assert_eq!(spec.text_bytes + spec.mmproj_bytes, 1_552_463_168);
    }

    #[test]
    fn unknown_pack_is_rejected_before_starting_a_process() {
        assert!(pack_spec("turbo").is_err());
    }

    #[test]
    fn setup_status_prefers_an_active_partial_download() {
        let directory = tempfile::tempdir().expect("temporary model directory");
        let model = directory.path().join("model.gguf");
        std::fs::write(&model, [0_u8; 3]).expect("existing model");
        std::fs::write(partial_path(&model), [0_u8; 7]).expect("partial model");

        assert_eq!(downloaded_size(&model), 7);
    }
}
