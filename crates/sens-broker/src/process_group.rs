use std::{ffi::c_void, io, mem::size_of, process::Stdio, ptr::null, time::Duration};

use tokio::process::{Child, Command};
use tracing::info;
use windows_sys::Win32::{
    Foundation::{CloseHandle, HANDLE},
    System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    },
};

pub(crate) fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

    command.as_std_mut().creation_flags(CREATE_NO_WINDOW);
}

/// Windows job handle that terminates the complete sidecar process tree when
/// the worker is dropped, including children spawned by venv launchers.
pub(crate) struct KillOnCloseJob(HANDLE);

// Kernel job handles may be closed from any thread.
unsafe impl Send for KillOnCloseJob {}

impl KillOnCloseJob {
    pub(crate) fn assign(child: &Child) -> io::Result<Self> {
        // SAFETY: all pointers are either null or point to initialized structs
        // for the duration of each Win32 call. Ownership of `job` is retained
        // by `KillOnCloseJob` and closed exactly once in Drop.
        unsafe {
            let job = CreateJobObjectW(null(), null());
            if job.is_null() {
                return Err(io::Error::last_os_error());
            }
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let configured = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&raw const info).cast::<c_void>(),
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if configured == 0 {
                let error = io::Error::last_os_error();
                CloseHandle(job);
                return Err(error);
            }
            let process = child
                .raw_handle()
                .ok_or_else(|| io::Error::other("sidecar has no process handle"))?
                as HANDLE;
            if AssignProcessToJobObject(job, process) == 0 {
                let error = io::Error::last_os_error();
                CloseHandle(job);
                return Err(error);
            }
            Ok(Self(job))
        }
    }
}

impl Drop for KillOnCloseJob {
    fn drop(&mut self) {
        // SAFETY: the handle is valid and owned by this wrapper.
        unsafe {
            CloseHandle(self.0);
        }
    }
}

pub(crate) async fn terminate_tree(child: &mut Child) {
    if let Some(pid) = child.id() {
        info!(pid, "terminating sidecar process tree");
        let mut command = Command::new("taskkill.exe");
        command
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        hide_console(&mut command);
        let outcome = tokio::time::timeout(Duration::from_secs(2), command.status()).await;
        info!(
            pid,
            timed_out = outcome.is_err(),
            "taskkill attempt completed"
        );
    }
    let _ = child.start_kill();
}
