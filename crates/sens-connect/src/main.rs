use std::path::PathBuf;

use sens_connect::{default_zcode_config_path, install, status, uninstall};

fn main() {
    if let Err(error) = run() {
        eprintln!("sens-connect: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args_os().skip(1);
    let target = arguments.next().and_then(|value| value.into_string().ok());
    let action = arguments.next().and_then(|value| value.into_string().ok());
    if target.as_deref() != Some("zcode") {
        return Err(
            "usage: sens-connect zcode status|install|uninstall [config] [sens-mcp]".into(),
        );
    }
    let config = arguments
        .next()
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(default_zcode_config_path)?;
    let result = match action.as_deref() {
        Some("status") => serde_json::to_value(status(&config)?),
        Some("uninstall") => serde_json::to_value(uninstall(&config)?),
        Some("install") => {
            let executable = arguments
                .next()
                .map(PathBuf::from)
                .ok_or_else(|| "install requires an absolute sens-mcp path".to_string())?;
            let sens_root = executable
                .ancestors()
                .find(|path| path.join("sidecars").is_dir())
                .map(PathBuf::from)
                .or_else(|| executable.parent().map(PathBuf::from))
                .ok_or_else(|| "sens-mcp path has no parent".to_string())?;
            let eye_root = std::env::var_os("SENS_EYE_ROOT")
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::var_os("USERPROFILE").map(|home| {
                        PathBuf::from(home)
                            .join(".zcode")
                            .join("workspace")
                            .join("default")
                            .join("eye")
                    })
                })
                .ok_or_else(|| "Could not discover Eye; set SENS_EYE_ROOT".to_string())?;
            let speech_root = std::env::var_os("SENS_SPEECH_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| sens_root.join("sidecars").join("speech"));
            serde_json::to_value(install(
                &config,
                &executable,
                &sens_root,
                &eye_root,
                &speech_root,
            )?)
        }
        _ => return Err("action must be status, install, or uninstall".into()),
    }
    .map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::to_string_pretty(&result).map_err(|error| error.to_string())?
    );
    Ok(())
}
