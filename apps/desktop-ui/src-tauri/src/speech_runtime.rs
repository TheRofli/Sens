use std::{
    fs,
    io::{Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::settings::HearingSettings;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechLaunchPlan {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub hide_console: bool,
}

impl SpeechLaunchPlan {
    pub fn discover(speech_root: &Path) -> Result<Self, String> {
        if !speech_root.join("speech_app").is_dir() {
            return Err(format!("Speech не найден в {}", speech_root.display()));
        }
        let scripts = speech_root.join(".venv").join("Scripts");
        let pythonw = scripts.join("pythonw.exe");
        let python = scripts.join("python.exe");
        let program = if pythonw.is_file() {
            pythonw
        } else if python.is_file() {
            python
        } else {
            PathBuf::from("pythonw.exe")
        };
        Ok(Self {
            program,
            args: ["-m", "speech_app", "run", "--managed"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            hide_console: true,
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct SpeechRuntimeStatus {
    pub running: bool,
    pub managed: bool,
    #[serde(rename(deserialize = "engine_enabled"))]
    pub enabled: bool,
    pub hotkey: String,
    pub model: String,
    #[serde(rename(deserialize = "model_state"))]
    pub model_state: String,
    #[serde(rename(deserialize = "model_loaded"))]
    pub model_loaded: bool,
    #[serde(rename(deserialize = "model_loading"))]
    pub model_loading: bool,
    pub transcribing: bool,
    pub error: Option<String>,
}

fn parse_status_response(response: &[u8]) -> Result<SpeechRuntimeStatus, String> {
    let text = std::str::from_utf8(response)
        .map_err(|error| format!("Speech вернул не-UTF-8 ответ: {error}"))?;
    let (head, body) = text
        .split_once("\r\n\r\n")
        .ok_or_else(|| "Speech вернул неполный HTTP-ответ".to_string())?;
    let status = head
        .lines()
        .next()
        .ok_or_else(|| "Speech не вернул HTTP-статус".to_string())?;
    if !status.contains(" 200 ") {
        return Err(format!("Speech API ответил: {status}"));
    }
    serde_json::from_str(body)
        .map_err(|error| format!("Speech вернул некорректный статус: {error}"))
}

#[derive(Default)]
struct SpeechRuntimeInner {
    child: Option<Child>,
    token: String,
    #[cfg(windows)]
    job: Option<SpeechJob>,
}

#[derive(Clone)]
pub struct SpeechRuntime {
    speech_root: PathBuf,
    inner: Arc<Mutex<SpeechRuntimeInner>>,
}

impl SpeechRuntime {
    pub fn discover() -> Self {
        let speech_root = std::env::var_os("SENS_SPEECH_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"D:\Speech"));
        let token = fs::read_to_string(speech_root.join("data").join("sens-managed.token"))
            .unwrap_or_default()
            .trim()
            .to_string();
        Self {
            speech_root,
            inner: Arc::new(Mutex::new(SpeechRuntimeInner {
                child: None,
                token,
                #[cfg(windows)]
                job: None,
            })),
        }
    }

    pub fn ensure_started(&self) -> Result<SpeechRuntimeStatus, String> {
        if let Ok(status) = self.fetch_status() {
            return Ok(status);
        }

        let plan = SpeechLaunchPlan::discover(&self.speech_root)?;
        let token = {
            let inner = self
                .inner
                .lock()
                .map_err(|_| "Speech runtime lock повреждён".to_string())?;
            if inner.token.is_empty() {
                Uuid::new_v4().simple().to_string()
            } else {
                inner.token.clone()
            }
        };
        let data_dir = self.speech_root.join("data");
        fs::create_dir_all(&data_dir)
            .map_err(|error| format!("Не удалось подготовить данные Speech: {error}"))?;
        fs::write(data_dir.join("sens-managed.token"), &token)
            .map_err(|error| format!("Не удалось сохранить токен Speech: {error}"))?;
        let mut command = Command::new(&plan.program);
        command
            .args(&plan.args)
            .current_dir(&self.speech_root)
            .env("SENS_SPEECH_ROOT", &self.speech_root)
            .env("SPEECH_API_TOKEN", &token)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
            format!(
                "Не удалось запустить Speech через {}: {error}",
                plan.program.display()
            )
        })?;
        #[cfg(windows)]
        let job = match SpeechJob::assign(&child) {
            Ok(job) => job,
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!(
                    "Не удалось привязать Speech к жизненному циклу Sens: {error}"
                ));
            }
        };
        {
            let mut inner = self
                .inner
                .lock()
                .map_err(|_| "Speech runtime lock повреждён".to_string())?;
            inner.child = Some(child);
            inner.token = token;
            #[cfg(windows)]
            {
                inner.job = Some(job);
            }
        }

        let mut last_error = "Speech ещё запускается".to_string();
        for _ in 0..50 {
            thread::sleep(Duration::from_millis(100));
            match self.fetch_status() {
                Ok(status) => return Ok(status),
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }

    pub fn status(&self) -> SpeechRuntimeStatus {
        match self.fetch_status() {
            Ok(status) => status,
            Err(error) => SpeechRuntimeStatus {
                hotkey: "ctrl+win".into(),
                model_state: "stopped".into(),
                error: Some(error),
                ..Default::default()
            },
        }
    }

    pub fn sync_hearing_settings(
        &self,
        settings: &HearingSettings,
    ) -> Result<SpeechRuntimeStatus, String> {
        self.ensure_started()?;
        let payload = json!({
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
        });
        self.request_json("POST", "/api/settings", Some(&payload))?;
        if settings.enabled && settings.preload_model {
            let _ = self.request_json("POST", "/api/model/load", Some(&json!({})));
        }
        self.fetch_status()
    }

    pub fn stop(&self) {
        let Ok(mut inner) = self.inner.lock() else {
            return;
        };
        if let Some(mut child) = inner.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        #[cfg(windows)]
        {
            inner.job.take();
        }
    }

    fn fetch_status(&self) -> Result<SpeechRuntimeStatus, String> {
        let response = self.request_raw("GET", "/api/status", None)?;
        parse_status_response(&response)
    }

    fn request_json(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, String> {
        let response = self.request_raw(method, path, body)?;
        let text = std::str::from_utf8(&response)
            .map_err(|error| format!("Speech вернул не-UTF-8 ответ: {error}"))?;
        let (head, body) = text
            .split_once("\r\n\r\n")
            .ok_or_else(|| "Speech вернул неполный HTTP-ответ".to_string())?;
        let status = head.lines().next().unwrap_or_default();
        if !status.contains(" 200 ") && !status.contains(" 202 ") {
            return Err(format!("Speech API ответил: {status}"));
        }
        serde_json::from_str(body)
            .map_err(|error| format!("Speech вернул некорректный JSON: {error}"))
    }

    fn request_raw(
        &self,
        method: &str,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Vec<u8>, String> {
        let port_path = self.speech_root.join("data").join("api.port");
        let port = fs::read_to_string(&port_path)
            .map_err(|_| "Служба диктовки Speech не запущена".to_string())?
            .trim()
            .parse::<u16>()
            .map_err(|_| "Speech записал некорректный API-порт".to_string())?;
        let token = self
            .inner
            .lock()
            .map_err(|_| "Speech runtime lock повреждён".to_string())?
            .token
            .clone();
        let encoded = body
            .map(serde_json::to_vec)
            .transpose()
            .map_err(|error| error.to_string())?
            .unwrap_or_default();
        let mut request = format!(
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\nAccept: application/json\r\n"
        );
        if !token.is_empty() {
            request.push_str(&format!("Authorization: Bearer {token}\r\n"));
        }
        if body.is_some() {
            request.push_str("Content-Type: application/json\r\n");
            request.push_str(&format!("Content-Length: {}\r\n", encoded.len()));
        }
        request.push_str("\r\n");

        let mut stream = TcpStream::connect_timeout(
            &format!("127.0.0.1:{port}")
                .parse()
                .map_err(|error| format!("Некорректный Speech API адрес: {error}"))?,
            Duration::from_millis(500),
        )
        .map_err(|_| "Служба диктовки Speech не отвечает".to_string())?;
        stream
            .set_read_timeout(Some(Duration::from_secs(3)))
            .map_err(|error| error.to_string())?;
        stream
            .write_all(request.as_bytes())
            .and_then(|_| stream.write_all(&encoded))
            .map_err(|error| format!("Не удалось отправить команду Speech: {error}"))?;
        let mut response = Vec::new();
        stream
            .read_to_end(&mut response)
            .map_err(|error| format!("Не удалось прочитать ответ Speech: {error}"))?;
        Ok(response)
    }
}

impl Drop for SpeechRuntime {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.stop();
        }
    }
}

#[cfg(windows)]
fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(windows)]
struct SpeechJob(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
unsafe impl Send for SpeechJob {}

#[cfg(windows)]
impl SpeechJob {
    fn assign(child: &Child) -> std::io::Result<Self> {
        use std::{ffi::c_void, mem::size_of, os::windows::io::AsRawHandle, ptr::null};
        use windows_sys::Win32::{
            Foundation::CloseHandle,
            System::JobObjects::{
                AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
                SetInformationJobObject,
            },
        };

        unsafe {
            let job = CreateJobObjectW(null(), null());
            if job.is_null() {
                return Err(std::io::Error::last_os_error());
            }
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast::<c_void>(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
            {
                let error = std::io::Error::last_os_error();
                CloseHandle(job);
                return Err(error);
            }
            if AssignProcessToJobObject(job, child.as_raw_handle().cast()) == 0 {
                let error = std::io::Error::last_os_error();
                CloseHandle(job);
                return Err(error);
            }
            Ok(Self(job))
        }
    }
}

#[cfg(windows)]
impl Drop for SpeechJob {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

#[cfg(not(windows))]
fn hide_console(_command: &mut Command) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_launch_prefers_pythonw_and_has_no_legacy_tray() {
        let temp = tempfile::tempdir().expect("tempdir");
        let scripts = temp.path().join(".venv").join("Scripts");
        std::fs::create_dir_all(&scripts).expect("scripts");
        let pythonw = scripts.join("pythonw.exe");
        std::fs::write(&pythonw, b"fixture").expect("pythonw");
        std::fs::create_dir_all(temp.path().join("speech_app")).expect("speech app");

        let plan = SpeechLaunchPlan::discover(temp.path()).expect("launch plan");

        assert_eq!(plan.program, pythonw);
        assert_eq!(plan.args, ["-m", "speech_app", "run", "--managed"]);
        assert!(plan.hide_console);
    }

    #[test]
    fn runtime_status_parses_hotkey_and_managed_state() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"running\":true,\"managed\":true,\"engine_enabled\":true,\"hotkey\":\"ctrl+win\",\"model\":\"gigaam\",\"model_state\":\"loaded\",\"model_loaded\":true,\"model_loading\":false,\"transcribing\":false}";

        let status = parse_status_response(response).expect("status");

        assert!(status.running);
        assert!(status.managed);
        assert_eq!(status.hotkey, "ctrl+win");
        assert_eq!(status.model, "gigaam");
    }
}
