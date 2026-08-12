use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};

use async_trait::async_trait;
use sens_core::{CapabilityExecutor, CapabilityOutput};
use sens_protocol::{ArtifactRef, InvokeRequest, Provenance, SensError};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, MutexGuard},
    time::timeout,
};
use tracing::info;

use crate::process_group::{KillOnCloseJob, terminate_tree};

/// Local vision operations served by the Python sight-worker (no network):
/// see, read, locate, inspect, compare (deterministic pixel diff).
/// artifact_get is served by the cloud Eye (artifact store).
const CLOUD_OPERATIONS: [&str; 2] = ["artifact_get", "watch"];
const WEB_SESSION_TTL: Duration = Duration::from_secs(2 * 60 * 60);
const MAX_WEB_SESSIONS: usize = 8;
const WEB_START_DEFAULT_MAX_CALLS: u32 = 4;
const WEB_START_ANALYSIS_TIMEOUT_MS: u64 = 30 * 60 * 1_000;

#[derive(Debug, Clone)]
pub struct SightRuntimeConfig {
    // Local deterministic stack (RapidOCR + OpenCV) plus an optional CPU VLM.
    pub python_executable: PathBuf,
    pub local_worker: PathBuf,
    pub models_root: PathBuf,
    // Optional Eye compatibility worker for legacy artifacts and video.
    pub node_executable: PathBuf,
    pub eye_root: PathBuf,
    pub cloud_worker: PathBuf,
    // Global default VLM semantics pack chosen in the desktop settings
    // (vision.visionPack); per-request MCP arguments still win.
    pub vision_pack: Option<String>,
}

impl SightRuntimeConfig {
    pub fn discover() -> anyhow::Result<Self> {
        let sens_root = discover_sens_root();
        let python_executable = discover_python_executable(&sens_root);
        let models_root = std::env::var_os("SENS_MODELS_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|| sens_core::RuntimePaths::discover().data_dir.join("models"));
        let node_executable = std::env::var_os("SENS_NODE")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("node"));
        let eye_root = discover_eye_root();
        let local_worker = discover_worker_script("sight-worker.py");
        let cloud_worker = discover_worker_script("eye-worker.mjs");
        let vision_pack = discover_vision_pack(&eye_root);
        Ok(Self {
            python_executable,
            local_worker,
            models_root,
            node_executable,
            eye_root,
            cloud_worker,
            vision_pack,
        })
    }
}

pub(crate) fn discover_sens_root() -> PathBuf {
    if let Some(root) = std::env::var_os("SENS_ROOT") {
        return PathBuf::from(root);
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(root) = executable
            .ancestors()
            .find(|path| path.join("sidecars").is_dir())
    {
        return root.to_path_buf();
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

pub(crate) fn discover_python_executable(sens_root: &Path) -> PathBuf {
    choose_python_executable(std::env::var_os("SENS_PYTHON"), sens_root)
}

pub(crate) fn choose_python_executable(
    configured: Option<std::ffi::OsString>,
    sens_root: &Path,
) -> PathBuf {
    if let Some(path) = configured {
        return PathBuf::from(path);
    }
    let packaged = sens_root.join("runtime").join("python").join("python.exe");
    if packaged.is_file() {
        return packaged;
    }
    PathBuf::from("python")
}

/// Best-effort read of the desktop settings' vision pack (vision.visionPack in
/// the Eye config). Missing file/field or unknown value -> None (worker stays lite).
fn discover_vision_pack(eye_root: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(eye_root.join("config.json")).ok()?;
    let document: Value = serde_json::from_str(&contents).ok()?;
    let pack = document
        .get("vision")?
        .get("visionPack")?
        .as_str()?
        .to_owned();
    matches!(pack.as_str(), "lite" | "quality" | "quality_large").then_some(pack)
}

pub(crate) fn discover_worker_script(name: &str) -> PathBuf {
    if let Some(root) = std::env::var_os("SENS_SIDECARS_ROOT") {
        return PathBuf::from(root).join(name);
    }
    if let Ok(executable) = std::env::current_exe()
        && let Some(parent) = executable.parent()
    {
        let installed = parent.join("sidecars").join(name);
        if installed.is_file() {
            return installed;
        }
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("sidecars")
        .join(name)
}

pub(crate) fn discover_eye_root() -> PathBuf {
    if let Some(path) = std::env::var_os("SENS_EYE_ROOT") {
        return PathBuf::from(path);
    }
    if let Some(home) = std::env::var_os("USERPROFILE") {
        let candidate = PathBuf::from(home)
            .join(".zcode")
            .join("workspace")
            .join("default")
            .join("eye");
        if candidate.join("src").join("service.mjs").is_file() {
            return candidate;
        }
    }
    PathBuf::from("sidecars").join("eye")
}

struct WorkerProcess {
    _child: Child,
    _job: KillOnCloseJob,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

pub struct SightExecutor {
    config: SightRuntimeConfig,
    local: Mutex<Option<WorkerProcess>>,
    cloud: Mutex<Option<WorkerProcess>>,
    review_history: Mutex<HashMap<String, ReviewHistory>>,
    web_sessions: Mutex<HashMap<String, WebSession>>,
}

#[derive(Debug, Clone)]
struct ReviewHistory {
    review_count: u32,
    champion_score: f64,
    champion_quality: ReviewQuality,
    champion_screenshot: Option<String>,
    non_improving_reviews: u32,
}

#[derive(Debug, Clone)]
struct ReviewQuality {
    can_complete: bool,
    check_count: u32,
    failed_checks: u32,
    max_normalized_violation: f64,
    total_normalized_violation: f64,
    similarity: f64,
}

impl ReviewQuality {
    fn from_result(result: &Value, similarity: f64) -> Self {
        let checks = result
            .pointer("/visual/acceptance/checks")
            .and_then(Value::as_array);
        let mut quality = Self {
            can_complete: result
                .get("canComplete")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            check_count: 0,
            failed_checks: 0,
            max_normalized_violation: 0.0,
            total_normalized_violation: 0.0,
            similarity,
        };
        for check in checks.into_iter().flatten() {
            quality.check_count += 1;
            if check.get("passed").and_then(Value::as_bool) != Some(false) {
                continue;
            }
            quality.failed_checks += 1;
            let actual = check.get("actual").and_then(Value::as_f64);
            let threshold = check.get("threshold").and_then(Value::as_f64);
            let name = check
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let operator = check
                .get("operator")
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    if name.ends_with("_minimum") {
                        ">="
                    } else {
                        "<="
                    }
                });
            let violation = match (actual, threshold, operator) {
                (Some(actual), Some(threshold), ">=") => {
                    (threshold - actual).max(0.0) / threshold.abs().max(0.000_001)
                }
                (Some(actual), Some(threshold), _) => {
                    (actual - threshold).max(0.0) / threshold.abs().max(0.000_001)
                }
                _ => 1.0,
            };
            quality.max_normalized_violation = quality.max_normalized_violation.max(violation);
            quality.total_normalized_violation += violation;
        }
        quality
    }
}

fn compare_review_quality(current: &ReviewQuality, champion: &ReviewQuality) -> i8 {
    const EPSILON: f64 = 0.000_001;
    if current.can_complete != champion.can_complete {
        return if current.can_complete { 1 } else { -1 };
    }
    if current.check_count > 0 && champion.check_count > 0 {
        if current.failed_checks != champion.failed_checks {
            return if current.failed_checks < champion.failed_checks {
                1
            } else {
                -1
            };
        }
        if (current.max_normalized_violation - champion.max_normalized_violation).abs() > EPSILON {
            return if current.max_normalized_violation < champion.max_normalized_violation {
                1
            } else {
                -1
            };
        }
        if (current.total_normalized_violation - champion.total_normalized_violation).abs()
            > EPSILON
        {
            return if current.total_normalized_violation < champion.total_normalized_violation {
                1
            } else {
                -1
            };
        }
    }
    if current.similarity > champion.similarity + 0.000_1 {
        1
    } else if current.similarity + 0.000_1 < champion.similarity {
        -1
    } else {
        0
    }
}

#[derive(Debug, Clone, Default)]
struct CandidateSnapshot {
    screenshot: Option<String>,
    sha256: Option<String>,
}

#[derive(Debug, Clone)]
struct WebSession {
    source_url: String,
    reference_path: String,
    reference_sha256: String,
    contract_path: Option<String>,
    capture_settings: Value,
    candidate_url: Option<String>,
    previous_candidate: CandidateSnapshot,
    review_count: u32,
    last_active: Instant,
}

fn web_input_object(request: &InvokeRequest) -> Result<&serde_json::Map<String, Value>, SensError> {
    request.input.as_object().ok_or_else(|| {
        runtime_error(
            "invalid_input",
            "Sight input must be a JSON object",
            "Pass the tool arguments defined by Sens.",
        )
    })
}

fn build_web_start_capture_request(request: &InvokeRequest) -> Result<InvokeRequest, SensError> {
    let input = web_input_object(request)?;
    require_string(input, "sourceUrl")?;
    let source_url = input
        .get("sourceUrl")
        .and_then(Value::as_str)
        .expect("required sourceUrl");
    let mut capture_input = serde_json::Map::new();
    capture_input.insert("url".to_owned(), Value::String(source_url.to_owned()));
    capture_input.insert(
        "networkPolicy".to_owned(),
        Value::String("public".to_owned()),
    );
    capture_input.insert("fullPage".to_owned(), Value::Bool(false));
    capture_input.insert("scrollSteps".to_owned(), Value::from(0));
    for key in [
        "viewport",
        "dpr",
        "theme",
        "locale",
        "waitUntil",
        "timeoutMs",
        "settleMs",
    ] {
        if let Some(value) = input.get(key) {
            capture_input.insert(key.to_owned(), value.clone());
        }
    }
    let mut capture = InvokeRequest::new("sight", "capture", Value::Object(capture_input));
    capture.no_store = false;
    capture.timeout_ms = request.timeout_ms;
    Ok(capture)
}

fn build_web_start_see_request(
    request: &InvokeRequest,
    reference_path: &str,
    source_raster_assets: Option<&Value>,
    source_vector_assets: Option<&Value>,
    source_text_nodes: Option<&Value>,
    source_font_assets: Option<&Value>,
) -> Result<InvokeRequest, SensError> {
    let input = web_input_object(request)?;
    require_string(input, "prompt")?;
    require_string(input, "assetOutputDir")?;
    let mut see_input = json!({
        "imagePath": reference_path,
        "prompt": input.get("prompt").cloned().unwrap_or(Value::Null),
        "assetOutputDir": input.get("assetOutputDir").cloned().unwrap_or(Value::Null),
        "profile": "reconstruct",
        "targetKind": "web",
        "response": "brief",
        "resolveFocus": true,
    });
    let see_object = see_input
        .as_object_mut()
        .expect("web see input is always an object");
    for key in ["fast", "quality", "pack"] {
        if let Some(value) = input.get(key) {
            see_object.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(assets) = sanitize_source_raster_assets(source_raster_assets) {
        see_object.insert("sourceRasterAssets".to_owned(), assets);
    }
    if let Some(assets) = sanitize_source_vector_assets(source_vector_assets) {
        see_object.insert("sourceVectorAssets".to_owned(), assets);
    }
    if let Some(nodes) = sanitize_source_text_nodes(source_text_nodes) {
        see_object.insert("sourceTextNodes".to_owned(), nodes);
    }
    if let Some(assets) = sanitize_source_font_assets(source_font_assets) {
        see_object.insert("sourceFontAssets".to_owned(), assets);
    }
    let mut see = InvokeRequest::new("sight", "see", see_input);
    see.no_store = false;
    see.max_calls = Some(request.max_calls.unwrap_or(WEB_START_DEFAULT_MAX_CALLS));
    see.timeout_ms = Some(request.timeout_ms.unwrap_or(WEB_START_ANALYSIS_TIMEOUT_MS));
    Ok(see)
}

fn sanitize_source_text_nodes(value: Option<&Value>) -> Option<Value> {
    const MAX_SOURCE_TEXT_NODES: usize = 600;
    const STYLE_FIELDS: [&str; 9] = [
        "color",
        "fontFamily",
        "fontSize",
        "fontWeight",
        "fontStyle",
        "lineHeight",
        "letterSpacing",
        "textTransform",
        "textAlign",
    ];
    let entries = value?.as_array()?;
    let sanitized = entries
        .iter()
        .take(MAX_SOURCE_TEXT_NODES)
        .filter_map(|entry| {
            let object = entry.as_object()?;
            let text = object.get("text").and_then(Value::as_str)?.trim();
            let box_value = object.get("box")?;
            if text.is_empty()
                || text.len() > 500
                || box_value.as_array()?.len() != 4
                || object.get("visible").and_then(Value::as_bool) == Some(false)
            {
                return None;
            }
            let mut style = serde_json::Map::new();
            if let Some(source_style) = object.get("style").and_then(Value::as_object) {
                for field in STYLE_FIELDS {
                    if let Some(value) = source_style.get(field).and_then(Value::as_str) {
                        style.insert(
                            field.to_owned(),
                            Value::String(value.chars().take(256).collect()),
                        );
                    }
                }
            }
            let mut sanitized_entry = json!({
                "text": text,
                "box": box_value,
                "visible": true,
                "style": style,
            });
            let word_boxes = object
                .get("wordBoxes")
                .and_then(Value::as_array)
                .map(|entries| {
                    entries
                        .iter()
                        .take(128)
                        .filter_map(|entry| {
                            let object = entry.as_object()?;
                            let text = object.get("text").and_then(Value::as_str)?.trim();
                            let coordinates = object.get("box").and_then(Value::as_array)?;
                            if text.is_empty()
                                || text.len() > 128
                                || coordinates.len() != 4
                                || coordinates.iter().any(|value| {
                                    value
                                        .as_f64()
                                        .is_none_or(|coordinate| coordinate.abs() > 1_000_000.0)
                                })
                            {
                                return None;
                            }
                            Some(json!({
                                "text": text,
                                "box": coordinates,
                            }))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if !word_boxes.is_empty() {
                sanitized_entry
                    .as_object_mut()
                    .expect("sanitized source text node is always an object")
                    .insert("wordBoxes".to_owned(), Value::Array(word_boxes));
            }
            Some(sanitized_entry)
        })
        .collect::<Vec<_>>();
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

fn sanitize_source_font_assets(value: Option<&Value>) -> Option<Value> {
    const MAX_SOURCE_FONT_ASSETS: usize = 16;
    const FIELDS: [&str; 11] = [
        "family",
        "weight",
        "style",
        "stretch",
        "path",
        "sha256",
        "sizeBytes",
        "mediaType",
        "format",
        "source",
        "method",
    ];
    let entries = value?.as_array()?;
    let sanitized = entries
        .iter()
        .take(MAX_SOURCE_FONT_ASSETS)
        .filter_map(|entry| {
            let object = entry.as_object()?;
            let family = object.get("family").and_then(Value::as_str)?.trim();
            let path = object.get("path").and_then(Value::as_str)?.trim();
            let sha256 = object.get("sha256").and_then(Value::as_str)?.trim();
            let media_type = object.get("mediaType").and_then(Value::as_str)?;
            if family.is_empty()
                || family.len() > 128
                || path.is_empty()
                || sha256.len() != 64
                || !media_type.starts_with("font/") && !media_type.contains("font")
            {
                return None;
            }
            let mut selected = serde_json::Map::new();
            for field in FIELDS {
                if let Some(field_value) = object.get(field) {
                    selected.insert(field.to_owned(), field_value.clone());
                }
            }
            Some(Value::Object(selected))
        })
        .collect::<Vec<_>>();
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

fn sanitize_source_vector_assets(value: Option<&Value>) -> Option<Value> {
    const MAX_SOURCE_VECTOR_ASSETS: usize = 12;
    const FIELDS: [&str; 11] = [
        "vectorIndex",
        "domIndex",
        "path",
        "sha256",
        "sizeBytes",
        "mediaType",
        "box",
        "visible",
        "viewportCoverage",
        "source",
        "method",
    ];
    let entries = value?.as_array()?;
    let sanitized = entries
        .iter()
        .take(MAX_SOURCE_VECTOR_ASSETS)
        .filter_map(|entry| {
            let object = entry.as_object()?;
            if object.get("path").and_then(Value::as_str)?.is_empty()
                || object.get("sha256").and_then(Value::as_str)?.is_empty()
                || object.get("mediaType").and_then(Value::as_str)? != "image/svg+xml"
                || object.get("box").and_then(Value::as_array)?.len() != 4
            {
                return None;
            }
            let mut selected = serde_json::Map::new();
            for field in FIELDS {
                if let Some(field_value) = object.get(field) {
                    selected.insert(field.to_owned(), field_value.clone());
                }
            }
            Some(Value::Object(selected))
        })
        .collect::<Vec<_>>();
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

fn sanitize_source_raster_assets(value: Option<&Value>) -> Option<Value> {
    const MAX_SOURCE_RASTER_ASSETS: usize = 12;
    const FIELDS: [&str; 15] = [
        "rasterIndex",
        "domIndex",
        "kind",
        "path",
        "sha256",
        "sizeBytes",
        "mediaType",
        "box",
        "visible",
        "objectFit",
        "backgroundSize",
        "backdropColor",
        "overlappingLiveTextCount",
        "source",
        "method",
    ];
    let entries = value?.as_array()?;
    let sanitized = entries
        .iter()
        .take(MAX_SOURCE_RASTER_ASSETS)
        .filter_map(|entry| {
            let object = entry.as_object()?;
            if object.get("path").and_then(Value::as_str)?.is_empty()
                || object.get("sha256").and_then(Value::as_str)?.is_empty()
                || object.get("mediaType").and_then(Value::as_str)?.is_empty()
                || object.get("box").and_then(Value::as_array)?.len() != 4
            {
                return None;
            }
            let mut selected = serde_json::Map::new();
            for field in FIELDS {
                if let Some(field_value) = object.get(field) {
                    selected.insert(field.to_owned(), field_value.clone());
                }
            }
            Some(Value::Object(selected))
        })
        .collect::<Vec<_>>();
    (!sanitized.is_empty()).then_some(Value::Array(sanitized))
}

fn build_web_review_request(
    request: &InvokeRequest,
    reference_path: &str,
    contract_path: Option<&str>,
    capture_settings: &Value,
    candidate_url: &str,
) -> Result<InvokeRequest, SensError> {
    if candidate_url.trim().is_empty() {
        return Err(runtime_error(
            "invalid_input",
            "candidateUrl is required for the first review",
            "Pass the running local candidate URL returned by your preview server.",
        ));
    }
    let mut review_input = serde_json::Map::new();
    review_input.insert(
        "referencePath".to_owned(),
        Value::String(reference_path.to_owned()),
    );
    review_input.insert("url".to_owned(), Value::String(candidate_url.to_owned()));
    review_input.insert(
        "networkPolicy".to_owned(),
        Value::String("candidate".to_owned()),
    );
    review_input.insert("fullPage".to_owned(), Value::Bool(false));
    review_input.insert(
        "final".to_owned(),
        Value::Bool(
            request
                .input
                .get("final")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        ),
    );
    if let Some(path) = contract_path {
        review_input.insert("contractPath".to_owned(), Value::String(path.to_owned()));
    }
    if let Some(settings) = capture_settings.as_object() {
        for key in [
            "viewport",
            "dpr",
            "theme",
            "locale",
            "waitUntil",
            "timeoutMs",
            "settleMs",
        ] {
            if let Some(value) = settings.get(key) {
                review_input.insert(key.to_owned(), value.clone());
            }
        }
    }
    let mut review = InvokeRequest::new("sight", "review", Value::Object(review_input));
    review.no_store = false;
    review.max_calls = request.max_calls;
    review.timeout_ms = request.timeout_ms;
    Ok(review)
}

fn decorate_web_review_result(
    result: &mut Value,
    session_id: &str,
    source_sha256: &str,
    previous: Option<&CandidateSnapshot>,
    final_requested: bool,
    review_request_id: &str,
) -> CandidateSnapshot {
    let current = CandidateSnapshot {
        screenshot: result
            .pointer("/capture/screenshot")
            .and_then(Value::as_str)
            .map(str::to_owned),
        sha256: result
            .pointer("/capture/screenshotSha256")
            .and_then(Value::as_str)
            .map(str::to_owned),
    };
    let before = previous.cloned().unwrap_or_default();
    result["webSession"] = json!({
        "sessionId": session_id,
        "sourceFrozen": true,
        "freshCapture": true,
        "finalRequested": final_requested,
    });
    result["reviewRequestId"] = Value::String(review_request_id.to_owned());
    result["beforeCapture"] = json!({
        "screenshot": before.screenshot,
        "sha256": before.sha256,
        "source": "observed",
        "method": "previous-session-candidate-capture",
    });
    result["afterCapture"] = json!({
        "screenshot": current.screenshot,
        "sha256": current.sha256,
        "source": "observed",
        "method": "fresh-playwright-candidate-capture",
    });

    let blocking_reasons_empty = result
        .get("blockingReasons")
        .and_then(Value::as_array)
        .is_some_and(Vec::is_empty);
    let passed = result.get("visualPass").and_then(Value::as_bool) == Some(true)
        && result.get("webPass").and_then(Value::as_bool) == Some(true)
        && result.get("canComplete").and_then(Value::as_bool) == Some(true)
        && blocking_reasons_empty;
    let stopped =
        result.get("requiredAction").and_then(Value::as_str) == Some("stop-and-return-champion");
    if stopped {
        if let Some(object) = result.as_object_mut() {
            object.remove("repairHints");
        }
        result["workflow"] = json!({
            "state": "stopped-repair-budget-exhausted",
            "nextTool": Value::Null,
            "nextAction": "stop-and-return-champion",
            "prohibitedNextActions": ["modify-candidate", "apply-repair-hints", "run-another-review"],
            "source": "broker-owned-iteration-gate",
        });
        if let Some(policy) = result
            .get_mut("iterationPolicy")
            .and_then(Value::as_object_mut)
        {
            policy.insert("mayContinue".to_owned(), Value::Bool(false));
            policy.insert(
                "requiredAction".to_owned(),
                Value::String("stop-and-return-champion".to_owned()),
            );
            policy.insert(
                "instruction".to_owned(),
                Value::String(
                    "The bounded repair budget is exhausted. Return the broker-owned champion and do not modify or review it again in this session."
                        .to_owned(),
                ),
            );
        }
    }
    if passed {
        if let Some(object) = result.as_object_mut() {
            object.remove("repairHints");
        }
        let completed = final_requested && current.sha256.is_some();
        let required_action = if completed {
            "complete"
        } else {
            "request-fresh-final-review"
        };
        result["requiredAction"] = Value::String(required_action.to_owned());
        result["workflow"] = if completed {
            json!({
                "state": "complete",
                "nextTool": Value::Null,
                "nextAction": "complete",
                "prohibitedNextActions": ["modify-candidate", "apply-repair-hints"],
                "source": "broker-owned-completion-gate",
            })
        } else {
            json!({
                "state": "ready-for-fresh-final-review",
                "nextTool": "sens_web_review",
                "nextAction": "call-sens-web-review-with-final-true-without-modifying-candidate",
                "prohibitedNextActions": ["modify-candidate", "apply-repair-hints"],
                "source": "broker-owned-completion-gate",
            })
        };
        if let Some(policy) = result
            .get_mut("iterationPolicy")
            .and_then(Value::as_object_mut)
        {
            policy.insert("mayContinue".to_owned(), Value::Bool(false));
            policy.insert(
                "requiredAction".to_owned(),
                Value::String(required_action.to_owned()),
            );
            policy.insert(
                "instruction".to_owned(),
                Value::String(if completed {
                    "The fresh final review passed. Do not modify the candidate or run another review."
                        .to_owned()
                } else {
                    "No blocking repair remains. Do not modify the candidate; call sens_web_review once with final=true."
                        .to_owned()
                }),
            );
        }
        if completed {
            let candidate_sha256 = current
                .sha256
                .as_deref()
                .expect("completed web review has a candidate hash");
            result["completionReceipt"] = json!({
                "receiptId": format!("{session_id}:{review_request_id}"),
                "sessionId": session_id,
                "reviewRequestId": review_request_id,
                "sourceCaptureSha256": source_sha256,
                "candidateCaptureSha256": candidate_sha256,
                "freshCapture": true,
                "visualPass": true,
                "webPass": true,
                "canComplete": true,
                "source": "measured",
                "method": "broker-owned-fresh-final-web-review",
            });
        }
    }
    current
}

fn project_json_fields(value: &Value, fields: &[&str]) -> Value {
    let mut projected = serde_json::Map::new();
    if let Some(object) = value.as_object() {
        for field in fields {
            if let Some(item) = object.get(*field) {
                projected.insert((*field).to_owned(), item.clone());
            }
        }
    }
    Value::Object(projected)
}

fn compact_web_review_result(result: &Value) -> Value {
    const TOP_LEVEL_FIELDS: &[&str] = &[
        "schemaVersion",
        "completionScope",
        "reviewRequestId",
        "visualPass",
        "webPass",
        "canComplete",
        "verdict",
        "requiredAction",
        "blockingReasons",
        "repairHints",
        "workflow",
        "reference",
        "candidate",
        "capture",
        "noStore",
        "iterationPolicy",
        "webSession",
        "beforeCapture",
        "afterCapture",
        "sourceCapture",
        "urlWorkflow",
        "completionReceipt",
        "observed",
        "measured",
        "inferred",
    ];
    const VISUAL_FIELDS: &[&str] = &[
        "width",
        "height",
        "completionScope",
        "visualPass",
        "dimensions",
        "verdict",
        "canComplete",
        "blockingReasons",
        "requiredAction",
        "acceptance",
        "similarityScore",
        "metrics",
        "provenance",
        "mismatchRatio",
        "meanDelta",
    ];
    const WEB_FIELDS: &[&str] = &[
        "webPass",
        "textCoverage",
        "symbolArtCoverage",
        "controlCoverage",
        "structuralLineCoverage",
        "blockingReasons",
        "observed",
        "measured",
        "inferred",
    ];

    let mut compact = project_json_fields(result, TOP_LEVEL_FIELDS);
    let Some(compact_object) = compact.as_object_mut() else {
        return compact;
    };

    if let Some(visual) = result.get("visual") {
        let mut visual_projection = project_json_fields(visual, VISUAL_FIELDS);
        if let (Some(source_regions), Some(projected)) = (
            visual.get("hotRegions").and_then(Value::as_array),
            visual_projection.as_object_mut(),
        ) {
            projected.insert(
                "hotRegions".to_owned(),
                Value::Array(source_regions.iter().take(6).cloned().collect()),
            );
        }
        compact_object.insert("visual".to_owned(), visual_projection);
    }

    if let Some(web) = result.get("web") {
        let mut web_projection = project_json_fields(web, WEB_FIELDS);
        if let (Some(raster_audit), Some(projected)) =
            (web.get("rasterAudit"), web_projection.as_object_mut())
        {
            projected.insert(
                "rasterAudit".to_owned(),
                project_json_fields(raster_audit, &["observedCount", "allowedCount"]),
            );
        }
        compact_object.insert("web".to_owned(), web_projection);
    }

    compact
}

fn safe_web_session_path_component(session_id: &str) -> String {
    let component: String = session_id
        .chars()
        .take(80)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if component.is_empty() {
        "session".to_owned()
    } else {
        component
    }
}

fn persist_compact_web_review_result_in(
    root: &Path,
    session_id: &str,
    review_count: u32,
    result: &mut Value,
) -> Result<PathBuf, SensError> {
    let session_root = root.join(safe_web_session_path_component(session_id));
    std::fs::create_dir_all(&session_root).map_err(|error| {
        runtime_error(
            "review_report_persist_failed",
            format!("Could not create the URL reconstruction review-report directory: {error}"),
            "Check that the Sens cache directory is writable, then repeat the review.",
        )
    })?;
    let path = session_root.join(format!("review-{review_count:03}.json"));
    result["reviewReport"] = json!({
        "path": path.to_string_lossy(),
        "sessionId": session_id,
        "reviewCount": review_count,
        "reviewRequestId": result.get("reviewRequestId").cloned().unwrap_or(Value::Null),
        "source": "measured",
        "method": "broker-persisted-compact-web-review",
        "instruction": "If host context is compacted, read this report instead of reconstructing prior repairHints from memory.",
    });
    let bytes = serde_json::to_vec_pretty(result).map_err(|error| {
        runtime_error(
            "review_report_persist_failed",
            format!("Could not serialize the compact web review report: {error}"),
            "Retry the review; if this continues, export Sens diagnostics.",
        )
    })?;
    std::fs::write(&path, bytes).map_err(|error| {
        runtime_error(
            "review_report_persist_failed",
            format!("Could not persist the compact web review report: {error}"),
            "Check that the Sens cache directory is writable, then repeat the review.",
        )
    })?;
    Ok(path)
}

fn persist_compact_web_review_result(
    session_id: &str,
    review_count: u32,
    result: &mut Value,
) -> Result<PathBuf, SensError> {
    let root = sens_core::RuntimePaths::discover()
        .cache_dir
        .join("sight")
        .join("web-sessions");
    persist_compact_web_review_result_in(&root, session_id, review_count, result)
}

fn prune_web_sessions_for_insert(sessions: &mut HashMap<String, WebSession>, now: Instant) {
    sessions.retain(|_, session| now.duration_since(session.last_active) <= WEB_SESSION_TTL);
    while sessions.len() >= MAX_WEB_SESSIONS {
        let Some(oldest_id) = sessions
            .iter()
            .min_by_key(|(_, session)| session.last_active)
            .map(|(id, _)| id.clone())
        else {
            break;
        };
        sessions.remove(&oldest_id);
    }
}

impl SightExecutor {
    pub fn new(config: SightRuntimeConfig) -> Self {
        Self {
            config,
            local: Mutex::new(None),
            cloud: Mutex::new(None),
            review_history: Mutex::new(HashMap::new()),
            web_sessions: Mutex::new(HashMap::new()),
        }
    }

    async fn invoke_web_start(&self, request: &InvokeRequest) -> Result<Value, SensError> {
        let capture_request = build_web_start_capture_request(request)?;
        let capture = self.invoke_worker(&capture_request, false).await?;
        let reference_path = capture
            .get("screenshot")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                runtime_error(
                    "sight_protocol_error",
                    "Source capture did not return a persisted screenshot",
                    "Retry sens_web_start; if this continues, export Sens diagnostics.",
                )
            })?
            .to_owned();
        let reference_sha256 = capture
            .get("screenshotSha256")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                runtime_error(
                    "sight_protocol_error",
                    "Source capture did not return a screenshot hash",
                    "Retry sens_web_start; if this continues, export Sens diagnostics.",
                )
            })?
            .to_owned();
        let capture_settings = capture
            .get("settings")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let see_request = build_web_start_see_request(
            request,
            &reference_path,
            capture.get("sourceRasterAssets"),
            capture.get("sourceVectorAssets"),
            capture.get("textNodes"),
            capture.get("sourceFontAssets"),
        )?;
        let mut result = self.invoke_worker(&see_request, false).await?;
        let result_object = result.as_object_mut().ok_or_else(|| {
            runtime_error(
                "sight_protocol_error",
                "Sight reconstruction did not return a JSON object",
                "Retry sens_web_start; if this continues, export Sens diagnostics.",
            )
        })?;
        let contract_path = result_object
            .get("contractPath")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let mut artifacts = result_object
            .get("artifacts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if let Some(source_artifacts) = capture.get("artifacts").and_then(Value::as_array) {
            for artifact in source_artifacts {
                let id = artifact.get("id").and_then(Value::as_str);
                if !artifacts.iter().any(|existing| {
                    existing.get("id").and_then(Value::as_str) == id && id.is_some()
                }) {
                    artifacts.push(artifact.clone());
                }
            }
        }
        result_object.insert("artifacts".to_owned(), Value::Array(artifacts));
        let input = web_input_object(request)?;
        let source_url = input
            .get("sourceUrl")
            .and_then(Value::as_str)
            .expect("sourceUrl validated by capture builder")
            .to_owned();
        let session_id = request.request_id.clone();
        let now = Instant::now();
        {
            let mut sessions = self.web_sessions.lock().await;
            prune_web_sessions_for_insert(&mut sessions, now);
            sessions.insert(
                session_id.clone(),
                WebSession {
                    source_url: source_url.clone(),
                    reference_path: reference_path.clone(),
                    reference_sha256: reference_sha256.clone(),
                    contract_path: contract_path.clone(),
                    capture_settings: capture_settings.clone(),
                    candidate_url: None,
                    previous_candidate: CandidateSnapshot::default(),
                    review_count: 0,
                    last_active: now,
                },
            );
        }

        let source_capture = json!({
            "url": source_url,
            "finalUrl": capture.pointer("/source/finalUrl"),
            "screenshot": reference_path,
            "screenshotSha256": reference_sha256,
            "settings": capture_settings,
            "blockedRequests": capture.get("blockedRequests").cloned().unwrap_or_else(|| json!([])),
            "source": "observed",
            "method": "frozen-public-playwright-capture",
        });
        result_object.insert(
            "webSession".to_owned(),
            json!({
                "sessionId": session_id,
                "state": "source-frozen",
                "sourceFrozen": true,
                "reviewCount": 0,
            }),
        );
        result_object.insert("sourceCapture".to_owned(), source_capture);
        result_object.insert(
            "urlWorkflow".to_owned(),
            json!({
                "nextTool": "sens_web_review",
                "candidateUrlRequired": true,
                "rule": "Serve or copy the returned starter, then review its candidate URL. Do not recapture the source URL.",
            }),
        );
        result_object.insert(
            "requiredAction".to_owned(),
            Value::String("serve-starter-then-call-sens-web-review".to_owned()),
        );
        Ok(result)
    }

    async fn invoke_web_review(&self, request: &InvokeRequest) -> Result<Value, SensError> {
        let input = web_input_object(request)?;
        require_string(input, "sessionId")?;
        let session_id = input
            .get("sessionId")
            .and_then(Value::as_str)
            .expect("required sessionId")
            .to_owned();
        let candidate_from_request = input
            .get("candidateUrl")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_owned);
        let final_requested = input.get("final").and_then(Value::as_bool).unwrap_or(false);
        let now = Instant::now();
        let session = {
            let mut sessions = self.web_sessions.lock().await;
            let expired = sessions
                .get(&session_id)
                .is_some_and(|value| now.duration_since(value.last_active) > WEB_SESSION_TTL);
            if expired {
                sessions.remove(&session_id);
            }
            let current = sessions.get_mut(&session_id).ok_or_else(|| {
                runtime_error(
                    "web_session_not_found",
                    format!("URL reconstruction session was not found or expired: {session_id}"),
                    "Call sens_web_start again to freeze a new source session.",
                )
            })?;
            current.last_active = now;
            current.clone()
        };
        let candidate_url = candidate_from_request
            .or_else(|| session.candidate_url.clone())
            .ok_or_else(|| {
                runtime_error(
                    "invalid_input",
                    "candidateUrl is required for the first review",
                    "Pass the URL of the running local candidate preview.",
                )
            })?;
        let review_request = build_web_review_request(
            request,
            &session.reference_path,
            session.contract_path.as_deref(),
            &session.capture_settings,
            &candidate_url,
        )?;
        let mut result = self.invoke_worker(&review_request, false).await?;
        {
            let mut history = self.review_history.lock().await;
            apply_review_champion_policy(&mut history, &review_request, &mut result);
        }
        let snapshot = decorate_web_review_result(
            &mut result,
            &session_id,
            &session.reference_sha256,
            Some(&session.previous_candidate),
            final_requested,
            &review_request.request_id,
        );
        let review_count = session.review_count.saturating_add(1);
        result["sourceCapture"] = json!({
            "url": session.source_url,
            "screenshot": session.reference_path,
            "screenshotSha256": session.reference_sha256,
            "sourceFrozen": true,
            "source": "observed",
            "method": "frozen-public-playwright-capture",
        });
        result["webSession"]["reviewCount"] = Value::from(review_count);
        let workflow_terminal = result.get("completionReceipt").is_some()
            || result
                .pointer("/workflow/nextTool")
                .is_some_and(Value::is_null);
        result["urlWorkflow"] = json!({
            "nextTool": if workflow_terminal { Value::Null } else { Value::String("sens_web_review".to_owned()) },
            "candidateUrl": candidate_url,
            "sourceRecaptureAllowed": false,
            "freshFinalReceiptRequired": true,
        });
        {
            let mut sessions = self.web_sessions.lock().await;
            if let Some(current) = sessions.get_mut(&session_id) {
                current.candidate_url = Some(candidate_url);
                current.previous_candidate = snapshot;
                current.review_count = review_count;
                current.last_active = Instant::now();
            }
        }
        let mut compact = compact_web_review_result(&result);
        persist_compact_web_review_result(&session_id, review_count, &mut compact)?;
        Ok(compact)
    }

    async fn start_local(&self) -> Result<WorkerProcess, SensError> {
        if self.config.python_executable.components().count() > 1
            && !self.config.python_executable.is_file()
        {
            return Err(runtime_error(
                "sight_runtime_missing",
                format!(
                    "Sight Python runtime was not found at {}",
                    self.config.python_executable.display()
                ),
                "Open Sens diagnostics and repair Sight.",
            ));
        }
        if !self.config.local_worker.is_file() {
            return Err(runtime_error(
                "sight_adapter_missing",
                format!(
                    "Sight adapter was not found at {}",
                    self.config.local_worker.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.python_executable);
        command
            .arg(&self.config.local_worker)
            .env("SENS_MODELS_ROOT", &self.config.models_root)
            .env("PYTHONNOUSERSITE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        if let Some(pack) = &self.config.vision_pack {
            command.env("SENS_VISION_PACK", pack);
        }
        crate::process_group::hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
            runtime_error(
                "sight_start_failed",
                format!("Could not start Sight: {error}"),
                "Check the bundled Python runtime in Sens diagnostics.",
            )
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            runtime_error(
                "sight_start_failed",
                "Sight has no stdin",
                "Repair the Sens installation.",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            runtime_error(
                "sight_start_failed",
                "Sight has no stdout",
                "Repair the Sens installation.",
            )
        })?;
        let job = KillOnCloseJob::assign(&child).map_err(|error| {
            runtime_error(
                "sight_start_failed",
                format!("Could not supervise the Sight process tree: {error}"),
                "Restart Sens or repair the installation.",
            )
        })?;
        Ok(WorkerProcess {
            _child: child,
            _job: job,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    async fn start_cloud(&self) -> Result<WorkerProcess, SensError> {
        if !self
            .config
            .eye_root
            .join("src")
            .join("service.mjs")
            .is_file()
        {
            return Err(runtime_error(
                "sight_cloud_unavailable",
                "Cloud Eye was not found on this machine",
                "Install Eye or use the local vision tools (see/read/locate/inspect).",
            ));
        }
        if !self.config.cloud_worker.is_file() {
            return Err(runtime_error(
                "sight_adapter_missing",
                format!(
                    "Eye adapter was not found at {}",
                    self.config.cloud_worker.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.node_executable);
        command
            .arg(&self.config.cloud_worker)
            .env("SENS_EYE_ROOT", &self.config.eye_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        crate::process_group::hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
            runtime_error(
                "sight_start_failed",
                format!("Could not start Sight: {error}"),
                "Check the bundled Node runtime in Sens diagnostics.",
            )
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            runtime_error(
                "sight_start_failed",
                "Sight has no stdin",
                "Repair the Sens installation.",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            runtime_error(
                "sight_start_failed",
                "Sight has no stdout",
                "Repair the Sens installation.",
            )
        })?;
        let job = KillOnCloseJob::assign(&child).map_err(|error| {
            runtime_error(
                "sight_start_failed",
                format!("Could not supervise the Sight process tree: {error}"),
                "Restart Sens or repair the installation.",
            )
        })?;
        Ok(WorkerProcess {
            _child: child,
            _job: job,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    async fn invoke_worker(
        &self,
        request: &InvokeRequest,
        cloud: bool,
    ) -> Result<Value, SensError> {
        validate_sight_input(request, cloud)?;
        let runtime = if cloud { "Eye" } else { "Sight" };
        info!(
            request_id = %request.request_id,
            operation = %request.operation,
            cloud,
            timeout_ms = ?request.timeout_ms,
            "{runtime} request received"
        );
        let mut guard = self.acquire_worker(cloud).await?;
        if guard.is_none() {
            info!(request_id = %request.request_id, "starting {runtime} worker");
            *guard = Some(if cloud {
                self.start_cloud().await?
            } else {
                self.start_local().await?
            });
        }
        let process = guard.as_mut().expect("process initialized");
        let payload = json!({
            "requestId": request.request_id,
            "operation": request.operation,
            "input": request.input,
            "noStore": request.no_store,
            "maxCalls": request.max_calls,
        });
        let mut encoded = serde_json::to_vec(&payload).map_err(|error| {
            runtime_error(
                "sight_protocol_error",
                error.to_string(),
                "Retry the operation.",
            )
        })?;
        encoded.push(b'\n');
        if let Err(error) = process.stdin.write_all(&encoded).await {
            *guard = None;
            return Err(runtime_error(
                "sight_disconnected",
                format!("{runtime} worker disconnected: {error}"),
                "Retry; Sens will restart Sight.",
            ));
        }
        if let Err(error) = process.stdin.flush().await {
            // The write may have landed in the pipe buffer while the worker
            // process was already gone; drop the stale process so the next
            // request respawns a fresh worker instead of writing to a corpse.
            *guard = None;
            return Err(runtime_error(
                "sight_disconnected",
                format!("{runtime} worker could not receive the request: {error}"),
                "Retry; Sens will restart Sight.",
            ));
        }

        // Local stack warms up models on the first request; cloud Eye is slow
        // to respond, so both get a generous default timeout.
        let wait = Duration::from_millis(request.timeout_ms.unwrap_or(600_000));
        info!(request_id = %request.request_id, wait_ms = wait.as_millis(), "waiting for {runtime} response");
        let response = timeout(wait, process.stdout.next_line()).await;
        let line = match response {
            Ok(Ok(Some(line))) => line,
            Ok(Ok(None)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_disconnected",
                    format!("{runtime} worker exited without a response"),
                    "Retry; Sens will restart Sight.",
                ));
            }
            Ok(Err(error)) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_protocol_error",
                    format!("Could not read {runtime} response: {error}"),
                    "Retry the operation.",
                ));
            }
            Err(_) => {
                terminate_tree(&mut process._child).await;
                *guard = None;
                return Err(runtime_error(
                    "sight_timeout",
                    format!("{runtime} exceeded its {} ms timeout", wait.as_millis()),
                    "Retry; if this continues, export Sens diagnostics.",
                ));
            }
        };
        let response: Value = serde_json::from_str(&line).map_err(|error| {
            runtime_error(
                "sight_protocol_error",
                format!("{runtime} returned invalid JSON: {error}"),
                "Retry; if this continues, export Sens diagnostics.",
            )
        })?;
        if response.get("ok").and_then(Value::as_bool) != Some(true) {
            let message = response
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("Sight failed without an error message");
            return Err(runtime_error(
                "sight_failed",
                message,
                "Check the Sight provider and input in Sens diagnostics.",
            ));
        }
        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    async fn acquire_worker(
        &self,
        cloud: bool,
    ) -> Result<MutexGuard<'_, Option<WorkerProcess>>, SensError> {
        if cloud {
            Ok(self.cloud.lock().await)
        } else {
            self.local.try_lock().map_err(|_| {
                runtime_error(
                    "sight_busy",
                    "The local CPU Sight worker is already processing another request",
                    "Run local CPU vision and focus calls serially, then retry after the current request completes.",
                )
            })
        }
    }
}

#[async_trait]
impl CapabilityExecutor for SightExecutor {
    async fn invoke(&self, request: &InvokeRequest) -> Result<CapabilityOutput, SensError> {
        let cloud = is_cloud_operation(&request.operation);
        let mut result = match request.operation.as_str() {
            "web_start" => self.invoke_web_start(request).await?,
            "web_review" => self.invoke_web_review(request).await?,
            _ => self.invoke_worker(request, cloud).await?,
        };
        if request.operation == "review" {
            let mut history = self.review_history.lock().await;
            apply_review_champion_policy(&mut history, request, &mut result);
        }
        let (artifacts, provenance, warnings) = worker_metadata(&result, &request.operation);
        Ok(CapabilityOutput {
            data: result
                .get("data")
                .cloned()
                .unwrap_or_else(|| result.clone()),
            artifacts,
            provenance,
            usage: result.get("usage").cloned().unwrap_or(Value::Null),
            warnings,
        })
    }
}

fn apply_review_champion_policy(
    histories: &mut HashMap<String, ReviewHistory>,
    request: &InvokeRequest,
    result: &mut Value,
) {
    let Some(score) = result
        .pointer("/visual/similarityScore")
        .and_then(Value::as_f64)
    else {
        return;
    };
    let reference = request
        .input
        .get("referencePath")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let url = request
        .input
        .get("url")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if reference.is_empty() || url.is_empty() {
        return;
    }
    let key = format!("{reference}\0{url}");
    let screenshot = result
        .pointer("/capture/screenshot")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let passed = result
        .get("canComplete")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let final_requested = request
        .input
        .get("final")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let quality = ReviewQuality::from_result(result, score);
    let history = histories.entry(key).or_insert_with(|| ReviewHistory {
        review_count: 0,
        champion_score: score,
        champion_quality: quality.clone(),
        champion_screenshot: screenshot.clone(),
        non_improving_reviews: 0,
    });
    history.review_count += 1;

    let quality_comparison = compare_review_quality(&quality, &history.champion_quality);
    let decision;
    if quality_comparison > 0 {
        history.champion_score = score;
        history.champion_quality = quality.clone();
        history.champion_screenshot = screenshot;
        history.non_improving_reviews = 0;
        decision = "new-champion";
    } else if quality_comparison < 0 {
        if !final_requested {
            history.non_improving_reviews += 1;
        }
        decision = "regressed-rollback-required";
    } else {
        if history.review_count > 1 && !passed && !final_requested {
            history.non_improving_reviews += 1;
        }
        decision = if history.review_count == 1 {
            "initial-champion"
        } else {
            "unchanged"
        };
    }

    let exhausted = history.non_improving_reviews >= 3 && !passed;
    let regression = quality_comparison < 0;
    let may_continue = !regression && !exhausted;
    let required_action = if passed {
        "complete"
    } else if regression {
        "rollback-to-champion"
    } else if exhausted {
        "stop-and-return-champion"
    } else {
        "repair-only-returned-hints"
    };

    result["iterationPolicy"] = json!({
        "source": "broker-owned-runtime-state",
        "reviewCount": history.review_count,
        "decision": decision,
        "currentScore": score,
        "championScore": history.champion_score,
        "scoreDeltaFromChampion": (score - history.champion_score),
        "championScreenshot": history.champion_screenshot,
        "currentFailedChecks": quality.failed_checks,
        "championFailedChecks": history.champion_quality.failed_checks,
        "currentMaxNormalizedViolation": quality.max_normalized_violation,
        "championMaxNormalizedViolation": history.champion_quality.max_normalized_violation,
        "currentTotalNormalizedViolation": quality.total_normalized_violation,
        "championTotalNormalizedViolation": history.champion_quality.total_normalized_violation,
        "nonImprovingReviews": history.non_improving_reviews,
        "maxNonImprovingReviews": 3,
        "mayContinue": may_continue,
        "requiredAction": required_action,
        "instruction": "Keep a source-code snapshot for every champion. Apply only measured repairHints, never manual pixel scans. If score regresses, restore the champion code before any further repair; the champion screenshot is evidence and must never become a web asset."
    });

    if regression || exhausted {
        result["canComplete"] = Value::Bool(false);
        result["verdict"] = Value::String("fail".to_owned());
        result["requiredAction"] = Value::String(required_action.to_owned());
        let reason = if regression {
            json!({
                "code": "regression-from-champion",
                "detail": "The current candidate scores below the broker-owned champion. Restore the champion source before any new repair.",
                "epistemic": "measured",
                "evidence": {
                    "currentScore": score,
                    "championScore": history.champion_score,
                    "scoreDelta": score - history.champion_score,
                    "currentFailedChecks": quality.failed_checks,
                    "championFailedChecks": history.champion_quality.failed_checks,
                    "currentMaxNormalizedViolation": quality.max_normalized_violation,
                    "championMaxNormalizedViolation": history.champion_quality.max_normalized_violation
                }
            })
        } else {
            json!({
                "code": "repair-budget-exhausted",
                "detail": "Three consecutive reviews failed to improve the champion. Stop the unbounded edit loop and return the champion for a new bounded run.",
                "epistemic": "measured",
                "evidence": {
                    "championScore": history.champion_score,
                    "nonImprovingReviews": history.non_improving_reviews
                }
            })
        };
        if let Some(reasons) = result
            .get_mut("blockingReasons")
            .and_then(Value::as_array_mut)
        {
            reasons.insert(0, reason);
        }
    }
}

fn worker_metadata(
    result: &Value,
    operation: &str,
) -> (Vec<ArtifactRef>, Vec<Provenance>, Vec<String>) {
    let data = result.get("data").unwrap_or(result);
    let doc = data.get("doc");
    let source_id = doc
        .and_then(|value| value.pointer("/source/id"))
        .and_then(Value::as_str)
        .map(str::to_owned);

    let mut artifacts = Vec::new();
    if let Some(id) = result.get("artifactId").and_then(Value::as_str) {
        artifacts.push(ArtifactRef {
            id: id.to_owned(),
            kind: "eye_result".into(),
            uri: None,
        });
    }
    for items in [
        doc.and_then(|value| value.get("artifacts"))
            .and_then(Value::as_array),
        data.get("artifacts").and_then(Value::as_array),
    ]
    .into_iter()
    .flatten()
    {
        for item in items {
            let Some(id) = item.get("id").and_then(Value::as_str) else {
                continue;
            };
            if artifacts.iter().any(|artifact| artifact.id == id) {
                continue;
            }
            artifacts.push(ArtifactRef {
                id: id.to_owned(),
                kind: item
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("sight_artifact")
                    .to_owned(),
                uri: item.get("uri").and_then(Value::as_str).map(str::to_owned),
            });
        }
    }

    let mut provenance = Vec::new();
    if let Some(claims) = doc
        .and_then(|value| value.get("claims"))
        .and_then(Value::as_array)
    {
        for claim in claims {
            let (Some(kind), Some(method)) = (
                claim.get("epistemic").and_then(Value::as_str),
                claim.get("method").and_then(Value::as_str),
            ) else {
                continue;
            };
            if provenance
                .iter()
                .any(|item: &Provenance| item.kind == kind && item.method == method)
            {
                continue;
            }
            provenance.push(Provenance {
                kind: kind.to_owned(),
                method: method.to_owned(),
                confidence: claim.get("confidence").and_then(Value::as_f64),
                source_artifact_id: source_id.clone(),
                region: claim.get("regionNorm").cloned(),
            });
        }
    }
    collect_embedded_provenance(data, &source_id, &mut provenance);
    if provenance.is_empty() {
        let (kind, method) = default_operation_provenance(operation);
        provenance.push(Provenance {
            kind: kind.into(),
            method: method.into(),
            confidence: None,
            source_artifact_id: source_id,
            region: None,
        });
    }

    let mut warnings = Vec::new();
    if let Some(items) = doc
        .and_then(|value| value.get("warnings"))
        .and_then(Value::as_array)
    {
        for item in items {
            let message = item.get("message").and_then(Value::as_str);
            let code = item.get("code").and_then(Value::as_str);
            match (code, message) {
                (Some(code), Some(message)) => warnings.push(format!("{code}: {message}")),
                (None, Some(message)) => warnings.push(message.to_owned()),
                _ => {}
            }
        }
    }
    if let Some(items) = result.get("warnings").and_then(Value::as_array) {
        warnings.extend(items.iter().filter_map(Value::as_str).map(str::to_owned));
    }

    (artifacts, provenance, warnings)
}

fn collect_embedded_provenance(
    value: &Value,
    source_id: &Option<String>,
    output: &mut Vec<Provenance>,
) {
    match value {
        Value::Object(object) => {
            if let (Some(kind), Some(method)) = (
                object.get("source").and_then(Value::as_str),
                object.get("method").and_then(Value::as_str),
            ) && matches!(kind, "observed" | "measured" | "inferred")
                && !output
                    .iter()
                    .any(|item| item.kind == kind && item.method == method)
            {
                output.push(Provenance {
                    kind: kind.to_owned(),
                    method: method.to_owned(),
                    confidence: object.get("confidence").and_then(Value::as_f64),
                    source_artifact_id: source_id.clone(),
                    region: object.get("box").cloned(),
                });
            }
            for child in object.values() {
                collect_embedded_provenance(child, source_id, output);
            }
        }
        Value::Array(items) => {
            for child in items {
                collect_embedded_provenance(child, source_id, output);
            }
        }
        _ => {}
    }
}

fn default_operation_provenance(operation: &str) -> (&'static str, &'static str) {
    match operation {
        "read" | "locate" => ("inferred", "rapidocr"),
        "ask" => ("inferred", "local-vlm"),
        "compare" => ("measured", "opencv-hsv-pixel-diff"),
        "review" => ("measured", "playwright-dom-plus-opencv-web-review"),
        "web_start" => ("observed", "frozen-public-playwright-capture"),
        "web_review" => ("measured", "broker-owned-fresh-candidate-web-review"),
        "capture" => ("observed", "playwright-dom-capture"),
        "motion" => ("measured", "playwright-css-frame-diff"),
        "see" | "zoom" | "inspect" | "element" => ("measured", "sight-deterministic-pipeline"),
        _ => ("observed", "sens-worker"),
    }
}

fn is_cloud_operation(operation: &str) -> bool {
    CLOUD_OPERATIONS.contains(&operation)
}

fn validate_sight_input(request: &InvokeRequest, cloud: bool) -> Result<(), SensError> {
    let input = request.input.as_object().ok_or_else(|| {
        runtime_error(
            "invalid_input",
            "Sight input must be a JSON object",
            "Pass the tool arguments defined by Sens.",
        )
    })?;
    match request.operation.as_str() {
        "artifact_get" => require_string(input, "artifactId"),
        "watch" => require_existing_file(input, "videoPath"),
        "compare" => {
            require_existing_file(input, "referencePath")?;
            require_existing_file(input, "candidatePath")
        }
        "review" => {
            require_existing_file(input, "referencePath")?;
            if input.get("contractPath").is_some() {
                require_existing_file(input, "contractPath")?;
            }
            require_string(input, "url")
        }
        "locate" => {
            require_source(input)?;
            require_string(input, "target")
        }
        "inspect" => {
            require_source(input)?;
            let has_region = input.get("region").is_some();
            let has_target = input
                .get("target")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.trim().is_empty());
            if !has_region && !has_target {
                return Err(runtime_error(
                    "invalid_input",
                    "inspect requires region or target",
                    "Pass a pixel region or a visible text target.",
                ));
            }
            Ok(())
        }
        "see" | "read" => require_source(input),
        "zoom" => {
            require_source(input)?;
            let has_region = input.get("region").is_some();
            let has_som = input.get("somId").is_some();
            if !has_region && !has_som {
                return Err(runtime_error(
                    "invalid_input",
                    "zoom requires region or somId",
                    "Pass a pixel region or a SoM element id.",
                ));
            }
            Ok(())
        }
        "ask" => {
            require_source(input)?;
            require_string(input, "question")
        }
        "element" => {
            require_source(input)?;
            if input.get("id").is_none() {
                return Err(runtime_error(
                    "invalid_input",
                    "element requires id",
                    "Pass a SoM element id from a prior sens_see document.",
                ));
            }
            Ok(())
        }
        "motion" | "capture" => require_string(input, "url"),
        "vision_prompt" | "warm" => Ok(()),
        _ if cloud => Err(runtime_error(
            "operation_not_supported",
            format!("Eye does not support operation {}", request.operation),
            "Use see/read/locate/inspect for local vision.",
        )),
        _ => Err(runtime_error(
            "operation_not_supported",
            format!("Sight does not support operation {}", request.operation),
            "Use one of the supported Sight operations.",
        )),
    }
}

fn require_source(input: &serde_json::Map<String, Value>) -> Result<(), SensError> {
    require_existing_file(input, "imagePath")
}

fn require_string(input: &serde_json::Map<String, Value>, key: &str) -> Result<(), SensError> {
    if input
        .get(key)
        .and_then(Value::as_str)
        .is_some_and(|value| !value.trim().is_empty())
    {
        Ok(())
    } else {
        Err(runtime_error(
            "invalid_input",
            format!("{key} is required"),
            "Pass the required tool argument.",
        ))
    }
}

fn require_existing_file(
    input: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<(), SensError> {
    let value = input.get(key).and_then(Value::as_str).ok_or_else(|| {
        runtime_error(
            "invalid_input",
            format!("{key} is required"),
            "Pass an existing local file.",
        )
    })?;
    if Path::new(value).is_file() {
        Ok(())
    } else {
        Err(runtime_error(
            "file_not_found",
            format!("{key} does not exist or is not a file: {value}"),
            "Pass an existing local file path.",
        ))
    }
}

fn runtime_error(
    code: impl Into<String>,
    message: impl Into<String>,
    action: impl Into<String>,
) -> SensError {
    SensError {
        code: code.into(),
        message: message.into(),
        recoverable: true,
        action: Some(action.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn existing_image() -> PathBuf {
        let path = std::env::temp_dir().join("sens-sight-test.png");
        fs::write(&path, b"fixture").expect("write fixture");
        path
    }

    #[test]
    fn packaged_sight_python_wins_without_a_speech_checkout() {
        let root = std::env::temp_dir().join(format!("sens-runtime-test-{}", std::process::id()));
        let packaged = root.join("runtime").join("python").join("python.exe");
        fs::create_dir_all(packaged.parent().expect("parent")).expect("create runtime");
        fs::write(&packaged, b"fixture").expect("write python fixture");
        let selected = choose_python_executable(None, &root);

        assert_eq!(selected, packaged);
        fs::remove_dir_all(root).expect("remove runtime fixture");
    }

    #[test]
    fn configured_python_has_highest_priority() {
        let selected = choose_python_executable(
            Some(std::ffi::OsString::from(r"C:\SensPython\python.exe")),
            Path::new(r"C:\Sens"),
        );

        assert_eq!(selected, PathBuf::from(r"C:\SensPython\python.exe"));
    }

    #[test]
    fn locate_requires_target() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "locate",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing target");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn inspect_requires_region_or_target() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "inspect",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing focus");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn inspect_accepts_region() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "inspect",
            json!({ "imagePath": path.to_string_lossy(), "region": { "x": 0, "y": 0, "width": 10, "height": 10 } }),
        );
        assert!(validate_sight_input(&request, false).is_ok());
    }

    #[test]
    fn zoom_requires_region_or_som_id() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "zoom",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing focus");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn zoom_accepts_som_id() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "zoom",
            json!({ "imagePath": path.to_string_lossy(), "somId": 3 }),
        );
        assert!(validate_sight_input(&request, false).is_ok());
    }

    #[test]
    fn ask_requires_question() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "ask",
            json!({ "imagePath": path.to_string_lossy() }),
        );
        let error = validate_sight_input(&request, false).expect_err("missing question");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn motion_requires_url() {
        let request = InvokeRequest::new("sight", "motion", json!({}));
        let error = validate_sight_input(&request, false).expect_err("missing url");
        assert_eq!(error.code, "invalid_input");
    }

    #[test]
    fn review_requires_reference_and_url() {
        let path = existing_image();
        let valid = InvokeRequest::new(
            "sight",
            "review",
            json!({
                "referencePath": path.to_string_lossy(),
                "url": "http://localhost:8123/index.html"
            }),
        );
        assert!(validate_sight_input(&valid, false).is_ok());

        let missing_url = InvokeRequest::new(
            "sight",
            "review",
            json!({ "referencePath": path.to_string_lossy() }),
        );
        assert_eq!(
            validate_sight_input(&missing_url, false)
                .expect_err("missing review url")
                .code,
            "invalid_input"
        );
    }

    #[test]
    fn web_start_builders_freeze_public_source_and_prepare_web_reconstruction() {
        let request = InvokeRequest::new(
            "sight",
            "web_start",
            json!({
                "sourceUrl": "https://example.com/design",
                "prompt": "Recreate this site exactly",
                "assetOutputDir": r"C:\project\assets",
                "viewport": {"width": 1366, "height": 768},
                "dpr": 1.0,
                "theme": "dark",
                "fast": true,
                "pack": "lite"
            }),
        );

        let capture = build_web_start_capture_request(&request).expect("capture request");
        assert_eq!(capture.operation, "capture");
        assert_eq!(
            capture.input.get("networkPolicy").and_then(Value::as_str),
            Some("public")
        );
        assert_eq!(
            capture.input.get("url").and_then(Value::as_str),
            Some("https://example.com/design")
        );
        assert_eq!(
            capture.input.get("fullPage").and_then(Value::as_bool),
            Some(false)
        );

        let source_raster_assets = json!([{
            "path": r"C:\cache\hero.avif",
            "sha256": "abc123",
            "mediaType": "image/avif",
            "box": [-95, -274, 1542, 1464],
            "backdropColor": "rgb(220, 238, 255)"
        }]);
        let source_vector_assets = json!([{
            "vectorIndex": 0,
            "domIndex": 42,
            "path": r"C:\cache\letter-s.svg",
            "sha256": "def456",
            "sizeBytes": 1234,
            "mediaType": "image/svg+xml",
            "box": [349, 141, 498, 636],
            "visible": true,
            "source": "observed",
            "method": "sanitized-live-dom-svg"
        }]);
        let source_text_nodes = json!([{
            "text": "Gateway",
            "box": [62, 327, 448, 426],
            "wordBoxes": [{
                "text": "Gateway",
                "box": [76, 327, 439, 423]
            }],
            "visible": true,
            "style": {
                "fontFamily": "Whyte Inktrap, sans-serif",
                "fontWeight": "500",
                "fontStyle": "normal"
            }
        }]);
        let font_digest = "f".repeat(64);
        let source_font_assets = json!([{
            "family": "Whyte Inktrap",
            "weight": "500",
            "style": "normal",
            "path": r"C:\cache\whyte-medium.woff2",
            "sha256": font_digest,
            "sizeBytes": 4567,
            "mediaType": "font/woff2",
            "format": "woff2",
            "source": "observed",
            "method": "playwright-loaded-font-response"
        }]);
        let see = build_web_start_see_request(
            &request,
            r"C:\cache\reference.png",
            Some(&source_raster_assets),
            Some(&source_vector_assets),
            Some(&source_text_nodes),
            Some(&source_font_assets),
        )
        .expect("see request");
        assert_eq!(see.operation, "see");
        assert_eq!(
            see.input.get("imagePath").and_then(Value::as_str),
            Some(r"C:\cache\reference.png")
        );
        assert_eq!(
            see.input.get("profile").and_then(Value::as_str),
            Some("reconstruct")
        );
        assert_eq!(
            see.input.get("targetKind").and_then(Value::as_str),
            Some("web")
        );
        assert_eq!(
            see.input.get("response").and_then(Value::as_str),
            Some("brief")
        );
        assert_eq!(
            see.input.get("resolveFocus").and_then(Value::as_bool),
            Some(true)
        );
        assert_eq!(
            see.input.get("sourceRasterAssets"),
            Some(&source_raster_assets)
        );
        assert_eq!(
            see.input.get("sourceVectorAssets"),
            Some(&source_vector_assets)
        );
        assert_eq!(see.input.get("sourceTextNodes"), Some(&source_text_nodes));
        assert_eq!(see.input.get("sourceFontAssets"), Some(&source_font_assets));
        assert_eq!(see.max_calls, Some(4));
        assert_eq!(see.timeout_ms, Some(30 * 60 * 1_000));
    }

    #[test]
    fn web_start_see_builder_preserves_an_explicit_smaller_budget_and_timeout() {
        let mut request = InvokeRequest::new(
            "sight",
            "web_start",
            json!({
                "sourceUrl": "https://example.com/design",
                "prompt": "Recreate this site exactly",
                "assetOutputDir": r"C:\project\assets"
            }),
        );
        request.max_calls = Some(2);
        request.timeout_ms = Some(420_000);

        let see = build_web_start_see_request(
            &request,
            r"C:\cache\reference.png",
            None,
            None,
            None,
            None,
        )
        .expect("see request");

        assert_eq!(see.max_calls, Some(2));
        assert_eq!(see.timeout_ms, Some(420_000));
    }

    #[test]
    fn web_review_builder_reuses_frozen_reference_and_capture_settings() {
        let request = InvokeRequest::new(
            "sight",
            "web_review",
            json!({"sessionId": "session-1", "final": true}),
        );
        let review = build_web_review_request(
            &request,
            r"C:\cache\reference.png",
            Some(r"C:\cache\contract.json"),
            &json!({
                "viewport": {"width": 1366, "height": 768},
                "dpr": 1.0,
                "theme": "dark",
                "locale": "en-US",
                "waitUntil": "networkidle",
                "timeoutMs": 30000,
                "settleMs": 250
            }),
            "http://localhost:8123/index.html",
        )
        .expect("review request");

        assert_eq!(review.operation, "review");
        assert_eq!(
            review.input.get("referencePath").and_then(Value::as_str),
            Some(r"C:\cache\reference.png")
        );
        assert_eq!(
            review.input.get("contractPath").and_then(Value::as_str),
            Some(r"C:\cache\contract.json")
        );
        assert_eq!(
            review.input.get("url").and_then(Value::as_str),
            Some("http://localhost:8123/index.html")
        );
        assert_eq!(
            review.input.get("networkPolicy").and_then(Value::as_str),
            Some("candidate")
        );
        assert_eq!(
            review
                .input
                .pointer("/viewport/width")
                .and_then(Value::as_u64),
            Some(1366)
        );
        assert!(!review.no_store);
    }

    #[test]
    fn web_review_result_is_compact_without_losing_repair_contract() {
        let repeated_match = json!({
            "referenceElementId": 17,
            "referenceText": "DOWNLOAD",
            "candidateBoxSource": [727, 93, 757, 104],
            "referenceBoxSource": [726, 92, 758, 105],
            "typography": {"fontFamily": "Sens Inter Tight", "fontSize": 14},
            "payload": "x".repeat(512),
        });
        let raw = json!({
            "schemaVersion": "2.0.0",
            "completionScope": "visual+web",
            "verdict": "fail",
            "visualPass": false,
            "webPass": true,
            "canComplete": false,
            "requiredAction": "repair-visual",
            "blockingReasons": [{"code": "visual-threshold-failed"}],
            "reviewRequestId": "review-request-compact",
            "repairHints": {
                "source": "measured-reference-vs-candidate",
                "visual": [{"kind": "largest-visual-hot-region", "referenceBoxSource": [10, 20, 900, 220]}],
                "text": [{"action": "Adjust this DOM node's CSS"}],
                "controls": [],
                "structure": [],
            },
            "workflow": {"state": "repair-from-returned-hints", "nextTool": "sens_review"},
            "visual": {
                "similarityScore": 0.8783,
                "verdict": "fail",
                "dimensions": {"exactMatch": true},
                "acceptance": {"policy": "sens-reconstruction-v1", "checks": [{"name": "similarity", "passed": false}]},
                "metrics": {"pixel": {"score": 0.91}, "foreground": {"score": 0.89}, "text": {"score": 0.63}, "layout": {"score": 0.94}},
                "hotRegions": (0..20).map(|index| json!({"box": [index, 0, 100, 100], "areaRatio": 0.1})).collect::<Vec<_>>(),
                "zones": (0..20).map(|index| json!({"box": [index, 0, 100, 100]})).collect::<Vec<_>>(),
                "requiredAction": {"kind": "repair-largest-hot-region-from-existing-contract"},
            },
            "web": {
                "webPass": true,
                "textCoverage": {"referenceCount": 12, "liveCount": 12, "selectableCount": 12},
                "symbolArtCoverage": {"referenceCount": 0, "exactSelectableCount": 0},
                "controlCoverage": {"referenceCount": 3, "semanticCount": 3},
                "structuralLineCoverage": {"referenceCount": 2, "matchedCount": 2},
                "textMatches": vec![repeated_match.clone(); 40],
                "symbolArtMatches": vec![repeated_match.clone(); 40],
                "controlMatches": vec![repeated_match.clone(); 40],
                "structuralLineMatches": vec![repeated_match.clone(); 40],
                "rasterAudit": {"observedCount": 1, "allowedCount": 1, "elements": vec![repeated_match.clone(); 40]},
                "observed": {"domTextNodeCount": 12},
                "measured": {"canvas": [1440, 900]},
                "inferred": [],
            },
            "reference": {"path": "reference.png", "size": {"width": 1440, "height": 900}},
            "candidate": {"url": "http://127.0.0.1:8165/index.html"},
            "capture": {"screenshot": "candidate.png", "screenshotSha256": "candidate-sha", "textNodeCount": 12},
            "artifacts": vec![repeated_match.clone(); 40],
            "iterationPolicy": {"decision": "initial-champion", "requiredAction": "repair-only-returned-hints"},
            "webSession": {"sessionId": "session-1", "reviewCount": 1},
            "beforeCapture": {"sha256": null},
            "afterCapture": {"sha256": "candidate-sha"},
            "sourceCapture": {"screenshotSha256": "source-sha"},
            "urlWorkflow": {"nextTool": "sens_web_review"},
        });

        let compact = compact_web_review_result(&raw);

        assert_eq!(compact.get("verdict").and_then(Value::as_str), Some("fail"));
        assert_eq!(
            compact.get("reviewRequestId").and_then(Value::as_str),
            Some("review-request-compact")
        );
        assert_eq!(
            compact
                .pointer("/visual/similarityScore")
                .and_then(Value::as_f64),
            Some(0.8783)
        );
        assert_eq!(
            compact
                .pointer("/visual/hotRegions")
                .and_then(Value::as_array)
                .map(Vec::len),
            Some(6)
        );
        assert_eq!(
            compact
                .pointer("/web/textCoverage/referenceCount")
                .and_then(Value::as_u64),
            Some(12)
        );
        assert_eq!(
            compact
                .pointer("/web/rasterAudit/allowedCount")
                .and_then(Value::as_u64),
            Some(1)
        );
        assert!(compact.pointer("/web/textMatches").is_none());
        assert!(compact.pointer("/web/rasterAudit/elements").is_none());
        assert!(compact.pointer("/visual/zones").is_none());
        assert!(compact.get("artifacts").is_none());
        assert!(compact.get("repairHints").is_some());
        assert!(compact.get("iterationPolicy").is_some());
        assert!(compact.get("urlWorkflow").is_some());

        let raw_size = serde_json::to_vec(&raw)
            .expect("serialize raw review")
            .len();
        let compact_size = serde_json::to_vec(&compact)
            .expect("serialize compact review")
            .len();
        assert!(compact_size * 8 < raw_size, "{compact_size} vs {raw_size}");
    }

    #[test]
    fn broker_persists_compact_review_report_for_context_recovery() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "sens-broker-review-report-{}-{unique}",
            std::process::id()
        ));
        let mut review = json!({
            "verdict": "fail",
            "reviewRequestId": "review-request-2",
            "repairHints": {"text": [{"action": "Move the label by one pixel"}]},
            "iterationPolicy": {"requiredAction": "repair-only-returned-hints"},
        });

        let path =
            persist_compact_web_review_result_in(&root, "session/../../escape", 2, &mut review)
                .expect("persist review report");

        assert!(path.starts_with(&root));
        assert_eq!(
            path.file_name().and_then(|value| value.to_str()),
            Some("review-002.json")
        );
        assert_eq!(
            review
                .pointer("/reviewReport/reviewCount")
                .and_then(Value::as_u64),
            Some(2)
        );
        assert_eq!(
            review
                .pointer("/reviewReport/reviewRequestId")
                .and_then(Value::as_str),
            Some("review-request-2")
        );
        assert_eq!(
            review.pointer("/reviewReport/path").and_then(Value::as_str),
            path.to_str()
        );
        let persisted: Value =
            serde_json::from_slice(&std::fs::read(&path).expect("read persisted review report"))
                .expect("parse persisted review report");
        assert_eq!(persisted, review);
        assert!(!path.to_string_lossy().contains(".."));

        std::fs::remove_dir_all(&root).expect("remove review report fixture");
    }

    #[test]
    fn web_review_issues_receipt_only_for_a_fresh_requested_final_pass() {
        let passing = || {
            json!({
                "visualPass": true,
                "webPass": true,
                "canComplete": true,
                "blockingReasons": [],
                "repairHints": {
                    "text": [{"action": "Move an already passing label by one pixel"}],
                    "visual": [{"kind": "nonblocking-hot-region"}]
                },
                "workflow": {"state": "repair-from-returned-hints"},
                "iterationPolicy": {
                    "mayContinue": true,
                    "requiredAction": "repair-only-returned-hints"
                },
                "capture": {
                    "screenshot": r"C:\cache\candidate.png",
                    "screenshotSha256": "candidate-sha"
                }
            })
        };
        let previous = CandidateSnapshot {
            screenshot: Some(r"C:\cache\before.png".to_owned()),
            sha256: Some("before-sha".to_owned()),
        };

        let mut ordinary_review = passing();
        let ordinary_snapshot = decorate_web_review_result(
            &mut ordinary_review,
            "session-1",
            "source-sha",
            Some(&previous),
            false,
            "review-request-1",
        );
        assert!(ordinary_review.get("completionReceipt").is_none());
        assert!(ordinary_review.get("repairHints").is_none());
        assert_eq!(
            ordinary_review
                .get("reviewRequestId")
                .and_then(Value::as_str),
            Some("review-request-1")
        );
        assert_eq!(
            ordinary_review
                .get("requiredAction")
                .and_then(Value::as_str),
            Some("request-fresh-final-review")
        );
        assert_eq!(
            ordinary_review
                .pointer("/workflow/state")
                .and_then(Value::as_str),
            Some("ready-for-fresh-final-review")
        );
        assert_eq!(
            ordinary_review
                .pointer("/iterationPolicy/mayContinue")
                .and_then(Value::as_bool),
            Some(false)
        );
        assert_eq!(ordinary_snapshot.sha256.as_deref(), Some("candidate-sha"));
        assert_eq!(
            ordinary_review
                .pointer("/beforeCapture/sha256")
                .and_then(Value::as_str),
            Some("before-sha")
        );

        let mut final_review = passing();
        decorate_web_review_result(
            &mut final_review,
            "session-1",
            "source-sha",
            Some(&ordinary_snapshot),
            true,
            "review-request-2",
        );
        assert_eq!(
            final_review
                .pointer("/completionReceipt/reviewRequestId")
                .and_then(Value::as_str),
            Some("review-request-2")
        );
        assert!(final_review.get("repairHints").is_none());
        assert_eq!(
            final_review
                .pointer("/workflow/state")
                .and_then(Value::as_str),
            Some("complete")
        );
        assert_eq!(
            final_review
                .pointer("/completionReceipt/sourceCaptureSha256")
                .and_then(Value::as_str),
            Some("source-sha")
        );
        assert_eq!(
            final_review
                .pointer("/completionReceipt/candidateCaptureSha256")
                .and_then(Value::as_str),
            Some("candidate-sha")
        );

        let mut failing_final = passing();
        failing_final["webPass"] = Value::Bool(false);
        decorate_web_review_result(
            &mut failing_final,
            "session-1",
            "source-sha",
            Some(&ordinary_snapshot),
            true,
            "review-request-3",
        );
        assert!(failing_final.get("completionReceipt").is_none());
    }

    #[test]
    fn web_session_store_expires_stale_sessions_and_reserves_bounded_capacity() {
        fn session(last_active: Instant) -> WebSession {
            WebSession {
                source_url: "https://example.com".to_owned(),
                reference_path: r"C:\cache\reference.png".to_owned(),
                reference_sha256: "source-sha".to_owned(),
                contract_path: None,
                capture_settings: json!({}),
                candidate_url: None,
                previous_candidate: CandidateSnapshot::default(),
                review_count: 0,
                last_active,
            }
        }

        let now = Instant::now();
        let mut sessions = HashMap::new();
        sessions.insert(
            "expired".to_owned(),
            session(now - WEB_SESSION_TTL - Duration::from_secs(1)),
        );
        for index in 0..=MAX_WEB_SESSIONS {
            sessions.insert(
                format!("fresh-{index}"),
                session(
                    now - Duration::from_secs(
                        (MAX_WEB_SESSIONS - index.min(MAX_WEB_SESSIONS)) as u64,
                    ),
                ),
            );
        }

        prune_web_sessions_for_insert(&mut sessions, now);

        assert!(!sessions.contains_key("expired"));
        assert!(!sessions.contains_key("fresh-0"));
        assert_eq!(sessions.len(), MAX_WEB_SESSIONS - 1);
    }

    fn review_result(score: f64) -> Value {
        json!({
            "visual": {"similarityScore": score},
            "capture": {"screenshot": format!("C:/cache/{score}.png")},
            "canComplete": false,
            "verdict": "fail",
            "requiredAction": "repair-visual",
            "blockingReasons": []
        })
    }

    fn review_result_with_checks(score: f64, pixel: f64, hot: f64, bounding: f64) -> Value {
        json!({
            "visual": {
                "similarityScore": score,
                "acceptance": {
                    "checks": [
                        {"name": "similarity_minimum", "actual": score, "threshold": 0.88, "operator": ">=", "passed": score >= 0.88},
                        {"name": "pixel_mismatch_maximum", "actual": pixel, "threshold": 0.12, "operator": "<=", "passed": pixel <= 0.12},
                        {"name": "largest_hot_region_maximum", "actual": hot, "threshold": 0.05, "operator": "<=", "passed": hot <= 0.05},
                        {"name": "largest_material_hot_region_bounding_maximum", "actual": bounding, "threshold": 0.08, "operator": "<=", "passed": bounding <= 0.08}
                    ]
                }
            },
            "capture": {"screenshot": format!("C:/cache/{score}.png")},
            "canComplete": false,
            "verdict": "fail",
            "requiredAction": "repair-visual",
            "blockingReasons": []
        })
    }

    #[test]
    fn review_policy_prefers_completion_distance_over_composite_score() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "review",
            json!({
                "referencePath": path.to_string_lossy(),
                "url": "http://localhost:8126/index.html"
            }),
        );
        let mut histories = HashMap::new();
        let mut initial = review_result_with_checks(0.90, 0.14, 0.07, 0.12);
        apply_review_champion_policy(&mut histories, &request, &mut initial);

        let mut closer = review_result_with_checks(0.89, 0.119, 0.049, 0.09);
        apply_review_champion_policy(&mut histories, &request, &mut closer);

        assert_eq!(
            closer
                .pointer("/iterationPolicy/decision")
                .and_then(Value::as_str),
            Some("new-champion")
        );
        assert_eq!(
            closer
                .pointer("/iterationPolicy/championScore")
                .and_then(Value::as_f64),
            Some(0.89)
        );
        assert_eq!(
            closer
                .pointer("/iterationPolicy/currentFailedChecks")
                .and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn final_review_does_not_consume_another_non_improving_iteration() {
        let path = existing_image();
        let mut ordinary = InvokeRequest::new(
            "sight",
            "review",
            json!({
                "referencePath": path.to_string_lossy(),
                "url": "http://localhost:8127/index.html",
                "final": false
            }),
        );
        let mut histories = HashMap::new();
        let mut first = review_result(0.75);
        apply_review_champion_policy(&mut histories, &ordinary, &mut first);
        let mut second = review_result(0.75);
        apply_review_champion_policy(&mut histories, &ordinary, &mut second);
        assert_eq!(
            second
                .pointer("/iterationPolicy/nonImprovingReviews")
                .and_then(Value::as_u64),
            Some(1)
        );

        ordinary.input["final"] = Value::Bool(true);
        let mut final_result = review_result(0.75);
        apply_review_champion_policy(&mut histories, &ordinary, &mut final_result);
        assert_eq!(
            final_result
                .pointer("/iterationPolicy/nonImprovingReviews")
                .and_then(Value::as_u64),
            Some(1)
        );
    }

    #[test]
    fn exhausted_review_is_a_terminal_workflow_without_repair_hints() {
        let mut result = json!({
            "visualPass": false,
            "webPass": true,
            "canComplete": false,
            "requiredAction": "stop-and-return-champion",
            "blockingReasons": [{"code": "repair-budget-exhausted"}],
            "repairHints": {"visual": [{"kind": "largest-visual-hot-region"}]},
            "workflow": {"state": "repair-from-returned-hints", "nextTool": "sens_review"},
            "iterationPolicy": {"mayContinue": false, "requiredAction": "stop-and-return-champion"},
            "capture": {"screenshot": "C:/cache/current.png", "screenshotSha256": "candidate-sha"}
        });

        decorate_web_review_result(
            &mut result,
            "session-1",
            "source-sha",
            None,
            true,
            "review-request-terminal",
        );

        assert!(result.get("repairHints").is_none());
        assert_eq!(
            result.pointer("/workflow/state").and_then(Value::as_str),
            Some("stopped-repair-budget-exhausted")
        );
        assert!(
            result
                .pointer("/workflow/nextTool")
                .is_some_and(Value::is_null)
        );
    }

    #[test]
    fn review_policy_preserves_champion_and_blocks_regressions() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "review",
            json!({
                "referencePath": path.to_string_lossy(),
                "url": "http://localhost:8123/index.html"
            }),
        );
        let mut histories = HashMap::new();
        let mut initial = review_result(0.70);
        apply_review_champion_policy(&mut histories, &request, &mut initial);
        assert_eq!(
            initial
                .pointer("/iterationPolicy/decision")
                .and_then(Value::as_str),
            Some("initial-champion")
        );

        let mut improved = review_result(0.78);
        apply_review_champion_policy(&mut histories, &request, &mut improved);
        assert_eq!(
            improved
                .pointer("/iterationPolicy/decision")
                .and_then(Value::as_str),
            Some("new-champion")
        );

        let mut regressed = review_result(0.74);
        apply_review_champion_policy(&mut histories, &request, &mut regressed);
        assert_eq!(
            regressed
                .pointer("/iterationPolicy/championScore")
                .and_then(Value::as_f64),
            Some(0.78)
        );
        assert_eq!(
            regressed.get("requiredAction").and_then(Value::as_str),
            Some("rollback-to-champion")
        );
        assert_eq!(
            regressed
                .pointer("/blockingReasons/0/code")
                .and_then(Value::as_str),
            Some("regression-from-champion")
        );
    }

    #[test]
    fn review_policy_stops_three_non_improving_reviews() {
        let path = existing_image();
        let request = InvokeRequest::new(
            "sight",
            "review",
            json!({
                "referencePath": path.to_string_lossy(),
                "url": "http://localhost:8124/index.html"
            }),
        );
        let mut histories = HashMap::new();
        for _ in 0..3 {
            let mut result = review_result(0.75);
            apply_review_champion_policy(&mut histories, &request, &mut result);
        }
        let mut exhausted = review_result(0.75);
        apply_review_champion_policy(&mut histories, &request, &mut exhausted);

        assert_eq!(
            exhausted.get("requiredAction").and_then(Value::as_str),
            Some("stop-and-return-champion")
        );
        assert_eq!(
            exhausted
                .pointer("/blockingReasons/0/code")
                .and_then(Value::as_str),
            Some("repair-budget-exhausted")
        );
        assert_eq!(
            exhausted
                .pointer("/iterationPolicy/mayContinue")
                .and_then(Value::as_bool),
            Some(false)
        );
    }

    #[tokio::test]
    async fn concurrent_local_request_receives_recoverable_busy_error() {
        let executor = SightExecutor::new(SightRuntimeConfig {
            python_executable: PathBuf::new(),
            local_worker: PathBuf::new(),
            models_root: PathBuf::new(),
            node_executable: PathBuf::new(),
            eye_root: PathBuf::new(),
            cloud_worker: PathBuf::new(),
            vision_pack: None,
        });
        let _held = executor.local.lock().await;

        match executor.acquire_worker(false).await {
            Err(error) => {
                assert_eq!(error.code, "sight_busy");
                assert!(error.recoverable);
                assert!(
                    error
                        .action
                        .as_deref()
                        .is_some_and(|action| action.contains("serial"))
                );
            }
            Ok(_) => panic!("a concurrent local request must not enter a hidden queue"),
        }
    }

    #[test]
    fn vision_prompt_needs_no_input() {
        let request = InvokeRequest::new("sight", "vision_prompt", json!({}));
        assert!(validate_sight_input(&request, false).is_ok());
    }

    #[test]
    fn cloud_operations_are_recognized() {
        assert!(is_cloud_operation("artifact_get"));
        assert!(is_cloud_operation("watch"));
        assert!(!is_cloud_operation("compare"));
        assert!(!is_cloud_operation("see"));
        assert!(!is_cloud_operation("inspect"));
    }

    #[test]
    fn scene_metadata_reaches_the_shared_result_envelope() {
        let result = json!({
            "doc": {
                "source": {"id": "sha256-128:abc"},
                "claims": [{
                    "epistemic": "inferred",
                    "method": "rapidocr",
                    "confidence": 0.91,
                    "regionNorm": [1, 2, 3, 4]
                }],
                "artifacts": [{
                    "id": "som:abc",
                    "kind": "set-of-marks",
                    "uri": "C:/cache/som.png"
                }],
                "warnings": [{
                    "code": "semantics_unavailable",
                    "message": "No local VLM is loaded."
                }]
            }
        });

        let (artifacts, provenance, warnings) = worker_metadata(&result, "see");

        assert_eq!(artifacts[0].id, "som:abc");
        assert_eq!(provenance[0].kind, "inferred");
        assert_eq!(provenance[0].method, "rapidocr");
        assert_eq!(
            provenance[0].source_artifact_id.as_deref(),
            Some("sha256-128:abc")
        );
        assert_eq!(
            warnings,
            vec!["semantics_unavailable: No local VLM is loaded."]
        );
    }

    #[test]
    fn web_start_root_artifacts_reach_the_shared_result_envelope() {
        let result = json!({
            "artifacts": [{
                "id": "sha256:reference",
                "kind": "web-screenshot",
                "uri": "C:/cache/reference.png"
            }],
            "sourceCapture": {
                "source": "observed",
                "method": "frozen-public-playwright-capture"
            }
        });

        let (artifacts, provenance, _warnings) = worker_metadata(&result, "web_start");

        assert_eq!(artifacts.len(), 1);
        assert_eq!(artifacts[0].id, "sha256:reference");
        assert!(provenance.iter().any(|item| {
            item.kind == "observed" && item.method == "frozen-public-playwright-capture"
        }));
    }
}
