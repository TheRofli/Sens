//! Sens 1.4.0 Touch capability executor (broker-owned).
//!
//! Design v1.1 (external review 2026-08-13):
//! - Provider transport is broker-owned: the API key lives ONLY in this
//!   struct; HTTPS to the provider is made here; the Python worker never
//!   sees the key and has no direct filesystem or network access.
//! - "Worker requests, Broker permits and executes": every privileged tool
//!   (read/glob/grep/web_fetch/web_search/sandbox write) is executed here.
//! - Evidence receipts are issued at the moment of the real read/fetch;
//!   workers reference receipts by id and cannot cite what they never saw.
//! - Two-axis semantics: `claim_status` (inferred/verified) vs machine
//!   `evidence_status`; semantic conclusions stay inferred.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, USER_AGENT};
use sens_core::{CapabilityExecutor, CapabilityOutput, RuntimePaths};
use sens_protocol::touch::{
    ClaimStatus, ConsentRequest, EvidenceKind, EvidenceReceipt, EvidenceStatus, OutputFormat,
    Predicate, PredicateResult, TOUCH_PROTOCOL_VERSION, TaskContext, TouchBudget,
    TouchCancelRequest, TouchCheckRequest, TouchConsent, TouchJobStatus, TouchOpinionsRequest,
    TouchParallelRequest, TouchProgress, TouchProgressEvent, TouchRequest, TouchRole,
    TouchStatusRequest, TouchStatusResponse, TouchTaskPacket, TouchUsage, TouchVerifyRequest,
    WorkerResult,
};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::RwLock,
    time::timeout,
};
use tracing::info;

use crate::process_group::{KillOnCloseJob, hide_console};
use crate::sight::{
    discover_eye_root, discover_python_executable, discover_sens_root, discover_worker_script,
};

const MAX_FILE_READ_BYTES: u64 = 256 * 1024;
const MAX_GREP_FILE_BYTES: u64 = 1024 * 1024;
const MAX_GREP_MATCHES: usize = 200;
const MAX_GLOB_RESULTS: usize = 500;
const MAX_WEB_FETCH_BYTES: u64 = 2 * 1024 * 1024;
const WEB_FETCH_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_STORED_JOBS: usize = 32;
/// Default price table (USD per 1M tokens) for pessimistic cost estimates
/// when the user config does not set explicit prices. Pinned at Slice 0
/// verification (docs/touch/slice0-verification.md).
const PRICE_PER_1M_IN: f64 = 0.08;
const PRICE_PER_1M_OUT: f64 = 0.252;

fn format_usd(dollars: f64) -> String {
    format!("${dollars:.4}")
}

fn micro_usd(dollars: f64) -> u64 {
    (dollars * 1_000_000.0).round().max(0.0) as u64
}

fn now_rfc3339() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;
    // Compact RFC3339 with milliseconds (no chrono dependency).
    let secs = millis / 1000;
    let days = secs / 86_400;
    let (year, month, day) = civil_from_days(days);
    let rem = secs % 86_400;
    let (hour, minute, second) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{:03}Z",
        millis % 1000
    )
}

/// Days-to-civil conversion (Howard Hinnant's algorithm).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn runtime_error(
    code: &str,
    message: impl Into<String>,
    action: impl Into<String>,
) -> sens_protocol::SensError {
    sens_protocol::SensError {
        code: code.into(),
        message: message.into(),
        recoverable: true,
        action: Some(action.into()),
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Provider configuration. `Debug` masks the API key.
#[derive(Clone)]
pub struct ProviderConfig {
    pub kind: String,
    pub base_url: String,
    pub model: String,
    /// Broker-owned provider credential. Masked in Debug and never logged.
    pub api_key: String,
    pub price_per_1m_in: Option<f64>,
    pub price_per_1m_out: Option<f64>,
}

impl std::fmt::Debug for ProviderConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("kind", &self.kind)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<masked>")
            .field("price_per_1m_in", &self.price_per_1m_in)
            .field("price_per_1m_out", &self.price_per_1m_out)
            .finish()
    }
}

impl ProviderConfig {
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

#[derive(Debug, Clone)]
pub struct WebSearchConfig {
    pub provider: String,
    pub api_key: String,
}

impl WebSearchConfig {
    pub fn api_key(&self) -> &str {
        &self.api_key
    }
}

#[derive(Debug, Clone)]
pub struct TouchLimits {
    pub max_workers_per_turn: u32,
    pub max_parallel: u32,
    pub max_depth: u32,
    pub max_active_jobs: u32,
    pub max_candidates: u32,
}

impl Default for TouchLimits {
    fn default() -> Self {
        Self {
            max_workers_per_turn: 4,
            max_parallel: 3,
            max_depth: 1,
            max_active_jobs: 2,
            max_candidates: 3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpendLimits {
    pub max_per_task_usd: f64,
    pub max_per_day_usd: f64,
    pub confirm_above_usd: f64,
}

impl Default for SpendLimits {
    fn default() -> Self {
        Self {
            max_per_task_usd: 0.50,
            max_per_day_usd: 5.00,
            confirm_above_usd: 0.20,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SandboxConfig {
    pub root: Option<PathBuf>,
    pub max_size_mb: u64,
    pub ttl_minutes: u64,
    pub copy_dependencies: Vec<String>,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            root: None,
            max_size_mb: 50,
            ttl_minutes: 60,
            copy_dependencies: vec![
                "package.json".into(),
                "Cargo.toml".into(),
                "tsconfig.json".into(),
            ],
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobsConfig {
    pub result_ttl_minutes: u64,
    pub max_stored: usize,
}

impl Default for JobsConfig {
    fn default() -> Self {
        Self {
            result_ttl_minutes: 60,
            max_stored: DEFAULT_MAX_STORED_JOBS,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TouchRuntimeConfig {
    pub enabled: bool,
    pub python_executable: PathBuf,
    pub worker_script: PathBuf,
    pub provider: ProviderConfig,
    pub web_search: WebSearchConfig,
    pub limits: TouchLimits,
    pub worker: TouchBudget,
    pub spend: SpendLimits,
    pub sandbox: SandboxConfig,
    pub jobs: JobsConfig,
}

impl TouchRuntimeConfig {
    pub fn discover() -> anyhow::Result<Self> {
        let sens_root = discover_sens_root();
        let python_executable = discover_python_executable(&sens_root);
        let worker_script = discover_worker_script("touch-worker.py");
        let eye_root = discover_eye_root();
        let mut config = Self {
            enabled: false,
            python_executable,
            worker_script,
            provider: ProviderConfig {
                kind: "openrouter".into(),
                base_url: "https://openrouter.ai/api/v1".into(),
                model: "deepseek/deepseek-v4-flash-0731".into(),
                api_key: String::new(),
                price_per_1m_in: None,
                price_per_1m_out: None,
            },
            web_search: WebSearchConfig {
                provider: "tavily".into(),
                api_key: String::new(),
            },
            limits: TouchLimits::default(),
            worker: TouchBudget::default(),
            spend: SpendLimits::default(),
            sandbox: SandboxConfig::default(),
            jobs: JobsConfig::default(),
        };
        let contents = std::fs::read_to_string(eye_root.join("config.json")).unwrap_or_default();
        let document: Value = serde_json::from_str(&contents).unwrap_or_else(|_| json!({}));
        let section = document.get("touch").cloned().unwrap_or_else(|| json!({}));

        config.enabled = section
            .get("enabled")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        if let Some(provider) = section.get("provider") {
            config.provider.kind = provider
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("openrouter")
                .to_owned();
            config.provider.base_url = provider
                .get("baseUrl")
                .and_then(Value::as_str)
                .unwrap_or("https://openrouter.ai/api/v1")
                .to_owned();
            config.provider.model = provider
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("deepseek/deepseek-v4-flash-0731")
                .to_owned();
            config.provider.api_key = provider
                .get("apiKey")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            config.provider.price_per_1m_in = provider.get("pricePer1mIn").and_then(Value::as_f64);
            config.provider.price_per_1m_out =
                provider.get("pricePer1mOut").and_then(Value::as_f64);
        }
        if let Some(web_search) = section.get("webSearch") {
            config.web_search.provider = web_search
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or("tavily")
                .to_owned();
            config.web_search.api_key = web_search
                .get("apiKey")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
        }
        if let Some(limits) = section.get("limits") {
            config.limits.max_workers_per_turn = limits
                .get("maxWorkersPerTurn")
                .and_then(Value::as_u64)
                .unwrap_or(4) as u32;
            config.limits.max_parallel = limits
                .get("maxParallel")
                .and_then(Value::as_u64)
                .unwrap_or(3) as u32;
            config.limits.max_depth =
                limits.get("maxDepth").and_then(Value::as_u64).unwrap_or(1) as u32;
            config.limits.max_active_jobs = limits
                .get("maxActiveJobs")
                .and_then(Value::as_u64)
                .unwrap_or(2) as u32;
            config.limits.max_candidates = limits
                .get("maxCandidates")
                .and_then(Value::as_u64)
                .unwrap_or(3) as u32;
        }
        if let Some(worker) = section.get("worker") {
            config.worker.max_steps =
                worker.get("maxSteps").and_then(Value::as_u64).unwrap_or(15) as u32;
            config.worker.max_total_input_tokens = worker
                .get("maxTotalInputTokens")
                .and_then(Value::as_u64)
                .unwrap_or(50_000) as u32;
            config.worker.max_total_output_tokens = worker
                .get("maxTotalOutputTokens")
                .and_then(Value::as_u64)
                .unwrap_or(6_000) as u32;
            config.worker.max_context_tokens = worker
                .get("maxContextTokens")
                .and_then(Value::as_u64)
                .unwrap_or(24_000) as u32;
            config.worker.max_tool_result_tokens = worker
                .get("maxToolResultTokens")
                .and_then(Value::as_u64)
                .unwrap_or(6_000) as u32;
            config.worker.max_single_tool_result_tokens = worker
                .get("maxSingleToolResultTokens")
                .and_then(Value::as_u64)
                .unwrap_or(2_500) as u32;
            config.worker.timeout_s = worker
                .get("timeoutS")
                .and_then(Value::as_u64)
                .unwrap_or(180);
            config.worker.max_spend_usd = worker.get("maxSpendUsd").and_then(Value::as_f64);
        }
        if let Some(spend) = section.get("spend") {
            config.spend.max_per_task_usd = spend
                .get("maxPerTaskUsd")
                .and_then(Value::as_f64)
                .unwrap_or(0.50);
            config.spend.max_per_day_usd = spend
                .get("maxPerDayUsd")
                .and_then(Value::as_f64)
                .unwrap_or(5.00);
            config.spend.confirm_above_usd = spend
                .get("confirmAboveUsd")
                .and_then(Value::as_f64)
                .unwrap_or(0.20);
        }
        if let Some(sandbox) = section.get("sandbox") {
            config.sandbox.root = sandbox
                .get("root")
                .and_then(Value::as_str)
                .map(PathBuf::from);
            config.sandbox.max_size_mb = sandbox
                .get("maxSizeMb")
                .and_then(Value::as_u64)
                .unwrap_or(50);
            config.sandbox.ttl_minutes = sandbox
                .get("ttlMinutes")
                .and_then(Value::as_u64)
                .unwrap_or(60);
            config.sandbox.copy_dependencies = sandbox
                .get("copyDependencies")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect()
                })
                .unwrap_or_else(|| {
                    vec![
                        "package.json".into(),
                        "Cargo.toml".into(),
                        "tsconfig.json".into(),
                    ]
                });
        }
        if let Some(jobs) = section.get("jobs") {
            config.jobs.result_ttl_minutes = jobs
                .get("resultTtlMinutes")
                .and_then(Value::as_u64)
                .unwrap_or(60);
            config.jobs.max_stored =
                jobs.get("maxStored").and_then(Value::as_u64).unwrap_or(32) as usize;
        }
        Ok(config)
    }
}

// ---------------------------------------------------------------------------
// Job model
// ---------------------------------------------------------------------------

struct WorkerProcess {
    _child: Child,
    _job: KillOnCloseJob,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

struct Job {
    id: String,
    role: TouchRole,
    packet: TouchTaskPacket,
    status: TouchJobStatus,
    receipts: Vec<EvidenceReceipt>,
    cumulative_usage: TouchUsage,
    cost_estimate_usd: f64,
    consent_request: Option<ConsentRequest>,
    progress: TouchProgress,
    result: Option<Value>,
    error: Option<String>,
    created_at: Instant,
    ttl: Duration,
    no_store: bool,
    sandbox: Option<PathBuf>,
    sandbox_originals: HashMap<String, (String, Vec<u8>)>,
}

impl Clone for Job {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            role: self.role,
            packet: self.packet.clone(),
            status: self.status,
            receipts: self.receipts.clone(),
            cumulative_usage: self.cumulative_usage.clone(),
            cost_estimate_usd: self.cost_estimate_usd,
            consent_request: self.consent_request.clone(),
            progress: self.progress.clone(),
            result: self.result.clone(),
            error: self.error.clone(),
            created_at: self.created_at,
            ttl: self.ttl,
            no_store: self.no_store,
            sandbox: self.sandbox.clone(),
            sandbox_originals: self.sandbox_originals.clone(),
        }
    }
}

impl Job {
    fn new(
        id: String,
        packet: TouchTaskPacket,
        no_store: bool,
        ttl: Duration,
        confirm_above_usd: f64,
        pessimistic_cost: f64,
        needs_consent: bool,
    ) -> Self {
        let status = if needs_consent {
            TouchJobStatus::AwaitingConsent
        } else {
            TouchJobStatus::Queued
        };
        let consent_request = if needs_consent {
            Some(ConsentRequest {
                cost_estimate_usd: pessimistic_cost,
                confirm_above_usd,
            })
        } else {
            None
        };
        Self {
            id,
            role: packet.role,
            packet,
            status,
            receipts: Vec::new(),
            cumulative_usage: TouchUsage {
                steps: 0,
                total_input_tokens: 0,
                total_output_tokens: 0,
                cost_estimate_usd: 0.0,
                latency_ms: 0,
            },
            cost_estimate_usd: 0.0,
            consent_request,
            progress: TouchProgress {
                step: 0,
                max_steps: 0,
                events: Vec::new(),
                elapsed_s: 0.0,
                cost_estimate_usd: 0.0,
            },
            result: None,
            error: None,
            created_at: Instant::now(),
            ttl,
            no_store,
            sandbox: None,
            sandbox_originals: HashMap::new(),
        }
    }

    fn to_status_response(&self) -> TouchStatusResponse {
        TouchStatusResponse {
            job_id: self.id.clone(),
            status: self.status,
            consent_request: self.consent_request.clone(),
            progress: Some(self.progress.clone()),
            result: self.result.clone(),
            error: self.error.clone(),
        }
    }
}

struct StoreInner {
    jobs: HashMap<String, Job>,
}

#[derive(Clone)]
struct JobStore {
    inner: Arc<RwLock<StoreInner>>,
    ttl: Duration,
    max_stored: usize,
}

impl JobStore {
    fn new(ttl: Duration, max_stored: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StoreInner {
                jobs: HashMap::new(),
            })),
            ttl,
            max_stored,
        }
    }

    async fn insert(&self, job: Job) {
        let mut guard = self.inner.write().await;
        guard.jobs.insert(job.id.clone(), job);
        self.sweep_locked(&mut guard);
    }

    async fn get(&self, id: &str) -> Option<Job> {
        let guard = self.inner.read().await;
        guard.jobs.get(id).cloned()
    }

    async fn update<F>(&self, id: &str, mutator: F)
    where
        F: FnOnce(&mut Job),
    {
        let mut guard = self.inner.write().await;
        if let Some(job) = guard.jobs.get_mut(id) {
            mutator(job);
        }
    }

    async fn snapshot(&self) -> Vec<Job> {
        let guard = self.inner.read().await;
        let mut jobs: Vec<_> = guard.jobs.values().cloned().collect();
        jobs.sort_by_key(|job| job.created_at);
        jobs
    }

    fn sweep_locked(&self, guard: &mut tokio::sync::RwLockWriteGuard<'_, StoreInner>) {
        let now = Instant::now();
        guard.jobs.retain(|_, job| {
            let expired = job.status == TouchJobStatus::Complete
                || job.status == TouchJobStatus::Failed
                || job.status == TouchJobStatus::Cancelled;
            let age_ok = !expired || now.duration_since(job.created_at) < job.ttl;
            // Keep queued/running jobs regardless of age; they are live.
            age_ok
                || matches!(
                    job.status,
                    TouchJobStatus::Queued
                        | TouchJobStatus::AwaitingConsent
                        | TouchJobStatus::Running
                        | TouchJobStatus::Partial
                )
        });
        let mut live: Vec<_> = guard
            .jobs
            .iter()
            .filter(|(_, job)| {
                matches!(
                    job.status,
                    TouchJobStatus::Queued
                        | TouchJobStatus::Running
                        | TouchJobStatus::AwaitingConsent
                )
            })
            .map(|(id, _)| id.clone())
            .collect();
        let keep = guard.jobs.len().saturating_sub(live.len());
        if keep > self.max_stored {
            // Evict oldest finished results beyond the cap (FIFO).
            let mut finished: Vec<_> = guard
                .jobs
                .iter()
                .filter(|(_, job)| {
                    matches!(
                        job.status,
                        TouchJobStatus::Complete
                            | TouchJobStatus::Failed
                            | TouchJobStatus::Cancelled
                    )
                })
                .map(|(id, job)| (job.created_at, id.clone()))
                .collect();
            finished.sort();
            let excess = finished.len().saturating_sub(self.max_stored);
            for (_, id) in finished.into_iter().take(excess) {
                guard.jobs.remove(&id);
                live.retain(|item| item != &id);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

pub struct TouchExecutor {
    config: TouchRuntimeConfig,
    client: reqwest::Client,
    jobs: JobStore,
    spend_today_micro_usd: AtomicU64,
}

impl std::fmt::Debug for TouchExecutor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TouchExecutor")
            .field("config", &self.config)
            .finish()
    }
}

impl TouchExecutor {
    pub fn new(config: TouchRuntimeConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .expect("reqwest client builds");
        let ttl = Duration::from_secs(config.jobs.result_ttl_minutes.max(1));
        let max_stored = config.jobs.max_stored;
        Self {
            config,
            client,
            jobs: JobStore::new(ttl, max_stored),
            spend_today_micro_usd: AtomicU64::new(0),
        }
    }

    fn price_in(&self) -> f64 {
        self.config
            .provider
            .price_per_1m_in
            .unwrap_or(PRICE_PER_1M_IN)
    }

    fn price_out(&self) -> f64 {
        self.config
            .provider
            .price_per_1m_out
            .unwrap_or(PRICE_PER_1M_OUT)
    }

    /// Pessimistic cost estimate for a job before it starts.
    fn pessimistic_cost(&self, budget: &TouchBudget) -> f64 {
        let input =
            budget.max_steps as f64 * budget.max_context_tokens as f64 * self.price_in() / 1e6;
        let output = budget.max_total_output_tokens as f64 * self.price_out() / 1e6;
        input + output
    }

    fn needs_consent(&self, budget: &TouchBudget, consent: TouchConsent) -> (bool, f64) {
        let estimate = self.pessimistic_cost(budget);
        let needs = consent == TouchConsent::Auto && estimate > self.config.spend.confirm_above_usd;
        (needs, estimate)
    }

    async fn dispatch(
        &self,
        request: &sens_protocol::InvokeRequest,
    ) -> Result<Value, sens_protocol::SensError> {
        if !self.config.enabled {
            return Err(runtime_error(
                "touch_disabled",
                "Touch is disabled in the Sens config (touch.enabled).",
                "Enable Touch in the Sens config to use worker delegation.",
            ));
        }
        match request.operation.as_str() {
            "touch" => {
                let input: TouchRequest = parse_input(request)?;
                self.create_job(input, request.no_store).await
            }
            "parallel" => {
                let input: TouchParallelRequest = parse_input(request)?;
                self.create_parallel(input, request.no_store).await
            }
            "opinions" => {
                let input: TouchOpinionsRequest = parse_input(request)?;
                self.create_opinions(input, request.no_store).await
            }
            "verify" => {
                let input: TouchVerifyRequest = parse_input(request)?;
                self.create_verify(input, request.no_store).await
            }
            "status" => {
                let input: TouchStatusRequest = parse_input(request)?;
                self.status(input).await
            }
            "cancel" => {
                let input: TouchCancelRequest = parse_input(request)?;
                self.cancel(input).await
            }
            "check" => {
                let input: TouchCheckRequest = parse_input(request)?;
                self.check(input).await
            }
            other => Err(runtime_error(
                "operation_not_supported",
                format!("Touch does not support operation {other}"),
                "Use one of touch, parallel, opinions, verify, status, cancel, check.",
            )),
        }
    }

    // -- Job creation -------------------------------------------------------

    fn build_packet(
        &self,
        id: &str,
        input: &TouchRequest,
        context: Option<TaskContext>,
    ) -> TouchTaskPacket {
        TouchTaskPacket {
            packet_id: format!("pkt_{}", id.trim_start_matches("tch_")),
            job_id: id.to_owned(),
            role: input.role,
            objective: input.objective.clone(),
            scope: input.scope.clone(),
            constraints: input.constraints.clone(),
            deliverable: input.deliverable.clone(),
            max_findings: None,
            output_format: input.output_format,
            budget: input.budget.clone(),
            context,
        }
    }

    async fn create_job(
        &self,
        input: TouchRequest,
        no_store: bool,
    ) -> Result<Value, sens_protocol::SensError> {
        if input.scope.is_empty() {
            return Err(runtime_error(
                "touch_invalid_scope",
                "scope must contain at least one entry (path globs and/or \"web\")",
                "Add a scope such as [\"src/**\"] or [\"web\"].",
            ));
        }
        let id = format!("tch_{}", uuid::Uuid::new_v4().simple());
        let context = TaskContext {
            os: Some("windows".to_owned()),
            cwd: None,
            repo: None,
        };
        let packet = self.build_packet(&id, &input, Some(context));
        let (needs_consent, estimate) = self.needs_consent(&packet.budget, input.consent);
        let spent_today = self.spend_today_micro_usd.load(Ordering::Relaxed) as f64 / 1e6;
        if spent_today + estimate > self.config.spend.max_per_day_usd {
            return Err(runtime_error(
                "touch_budget_limited",
                format!(
                    "daily spend limit reached ({} + {} > {})",
                    format_usd(spent_today),
                    format_usd(estimate),
                    format_usd(self.config.spend.max_per_day_usd)
                ),
                "Wait for the daily window to reset or raise touch.spend.maxPerDayUsd.",
            ));
        }
        let job = Job::new(
            id.clone(),
            packet,
            no_store,
            self.jobs.ttl,
            self.config.spend.confirm_above_usd,
            estimate,
            needs_consent,
        );
        self.jobs.insert(job).await;
        if !needs_consent {
            self.schedule_next();
        }
        Ok(json!({ "jobId": id, "status": "created", "awaitingConsent": needs_consent }))
    }

    async fn create_parallel(
        &self,
        input: TouchParallelRequest,
        no_store: bool,
    ) -> Result<Value, sens_protocol::SensError> {
        let max = self.config.limits.max_workers_per_turn as usize;
        if input.jobs.is_empty() || input.jobs.len() > max {
            return Err(runtime_error(
                "touch_invalid_parallel",
                format!("parallel requires 1..={max} jobs"),
                "Reduce the number of jobs.",
            ));
        }
        let mut ids = Vec::new();
        for job_input in input.jobs {
            let created = self.create_job(job_input, no_store).await?;
            ids.push(created.get("jobId").cloned().unwrap_or(Value::Null));
        }
        Ok(json!({ "jobs": ids, "count": ids.len() }))
    }

    async fn create_opinions(
        &self,
        input: TouchOpinionsRequest,
        no_store: bool,
    ) -> Result<Value, sens_protocol::SensError> {
        let perspectives: Vec<String> = match &input.perspectives {
            Some(named) if !named.is_empty() => named.clone(),
            _ => {
                let count = input
                    .perspectives_count
                    .unwrap_or(3)
                    .clamp(1, self.config.limits.max_candidates);
                sens_protocol::touch::default_perspectives(input.role)
                    .into_iter()
                    .take(count as usize)
                    .collect()
            }
        };
        if perspectives.is_empty() {
            return Err(runtime_error(
                "touch_invalid_opinions",
                "opinions requires perspectives or perspectivesCount >= 1",
                "Pass perspectives or perspectivesCount.",
            ));
        }
        let mut ids = Vec::new();
        for perspective in &perspectives {
            let objective = format!("{}\n\nPerspective: {perspective}", input.objective);
            let job_input = TouchRequest {
                role: input.role,
                objective,
                scope: vec!["web".to_owned()],
                constraints: vec![
                    "independent opinion; do not assume other opinions exist".to_owned(),
                ],
                deliverable: Some("opinion".to_owned()),
                output_format: OutputFormat::Auto,
                budget: input.budget.clone(),
                consent: input.consent,
            };
            let created = self.create_job(job_input, no_store).await?;
            ids.push(created.get("jobId").cloned().unwrap_or(Value::Null));
        }
        Ok(json!({ "jobs": ids, "count": ids.len(), "perspectives": perspectives }))
    }

    async fn create_verify(
        &self,
        input: TouchVerifyRequest,
        no_store: bool,
    ) -> Result<Value, sens_protocol::SensError> {
        let role = input.role;
        let objective = format!(
            "Critically verify the following candidate.\nCriteria: {}\n\nCANDIDATE:\n{}",
            input.criteria.join("; "),
            input.candidate
        );
        let job_input = TouchRequest {
            role,
            objective,
            scope: input.scope.clone(),
            constraints: vec!["verification role; do not modify anything".to_owned()],
            deliverable: Some("verification_report".to_owned()),
            output_format: OutputFormat::Auto,
            budget: TouchBudget::default(),
            consent: input.consent,
        };
        self.create_job(job_input, no_store).await
    }

    // -- Status / cancel ----------------------------------------------------

    async fn status(&self, input: TouchStatusRequest) -> Result<Value, sens_protocol::SensError> {
        let job = self.jobs.get(&input.job_id).await.ok_or_else(|| {
            runtime_error(
                "touch_job_not_found",
                format!("Unknown job {}", input.job_id),
                "Check the job id; finished results expire after the configured TTL.",
            )
        })?;
        if input.consent && job.status == TouchJobStatus::AwaitingConsent {
            self.schedule_next();
            let updated = self.jobs.get(&input.job_id).await.unwrap_or(job);
            return serde_json::to_value(updated.to_status_response()).map_err(|error| {
                runtime_error("touch_protocol_error", error.to_string(), "Retry.")
            });
        }
        serde_json::to_value(job.to_status_response())
            .map_err(|error| runtime_error("touch_protocol_error", error.to_string(), "Retry."))
    }

    async fn cancel(&self, input: TouchCancelRequest) -> Result<Value, sens_protocol::SensError> {
        let job = self.jobs.get(&input.job_id).await.ok_or_else(|| {
            runtime_error(
                "touch_job_not_found",
                format!("Unknown job {}", input.job_id),
                "Check the job id.",
            )
        })?;
        match job.status {
            TouchJobStatus::Complete | TouchJobStatus::Failed | TouchJobStatus::Cancelled => {
                return Ok(json!({ "jobId": input.job_id, "status": "complete" }));
            }
            _ => {}
        }
        self.jobs
            .update(&input.job_id, |job| {
                job.status = TouchJobStatus::Cancelled;
                job.error = Some("cancelled by client".to_owned());
            })
            .await;
        self.schedule_next();
        Ok(json!({ "jobId": input.job_id, "status": "cancelled" }))
    }

    // -- Scheduler ----------------------------------------------------------

    async fn active_count(&self) -> usize {
        self.jobs
            .snapshot()
            .await
            .iter()
            .filter(|job| job.status == TouchJobStatus::Running)
            .count()
    }

    /// Try to start one job if capacity allows; otherwise leave it queued.
    async fn try_start(&self, id: &str) {
        let job = self.jobs.get(id).await;
        let Some(job) = job else { return };
        if job.status != TouchJobStatus::Queued && job.status != TouchJobStatus::AwaitingConsent {
            return;
        }
        if self.active_count().await >= self.config.limits.max_active_jobs as usize {
            if job.status == TouchJobStatus::AwaitingConsent {
                // Consent confirmed: move to queue; drain_queue picks it up.
                self.jobs
                    .update(id, |job| job.status = TouchJobStatus::Queued)
                    .await;
            }
            return;
        }
        self.jobs
            .update(id, |job| job.status = TouchJobStatus::Running)
            .await;
        let snapshot = self.jobs.get(id).await.expect("job exists");
        let executor = self.clone_for_job();
        tokio::spawn(async move {
            executor.run_job(snapshot).await;
        });
    }

    /// Start queued jobs in FIFO order while capacity allows. Detached from
    /// any running job future so the await graph never recurses.
    async fn drain_queue(&self) {
        loop {
            let snapshot = self.jobs.snapshot().await;
            let next = snapshot
                .iter()
                .filter(|job| {
                    job.status == TouchJobStatus::Queued
                        || job.status == TouchJobStatus::AwaitingConsent
                })
                .min_by_key(|job| job.created_at)
                .map(|job| job.id.clone());
            let Some(id) = next else { break };
            if self.active_count().await >= self.config.limits.max_active_jobs as usize {
                break;
            }
            self.try_start(&id).await;
        }
    }

    fn schedule_next(&self) {
        let executor = self.clone_for_job();
        tokio::spawn(async move {
            executor.drain_queue().await;
        });
    }

    fn clone_for_job(&self) -> Arc<TouchExecutor> {
        Arc::new(TouchExecutor {
            config: self.config.clone(),
            client: self.client.clone(),
            jobs: self.jobs.clone(),
            spend_today_micro_usd: AtomicU64::new(0),
        })
    }

    // -- Worker runner ------------------------------------------------------

    async fn run_job(&self, mut job: Job) {
        let started = Instant::now();
        let timeout_s = job.packet.budget.timeout_s.max(1);
        let id = job.id.clone();
        let result = self
            .run_worker_exchange(&id, &mut job, Duration::from_secs(timeout_s))
            .await;
        let elapsed = started.elapsed().as_millis() as u64;
        match result {
            Ok(()) => {
                self.jobs
                    .update(&id, |job| {
                        job.status = TouchJobStatus::Complete;
                        job.cumulative_usage.latency_ms = elapsed;
                        job.progress.elapsed_s = elapsed as f64 / 1000.0;
                    })
                    .await;
            }
            Err(error) => {
                let code = error.code.clone();
                self.jobs
                    .update(&id, |job| {
                        job.status = if code == "touch_worker_cancelled" {
                            TouchJobStatus::Cancelled
                        } else if job.status == TouchJobStatus::Partial {
                            TouchJobStatus::Partial
                        } else {
                            TouchJobStatus::Failed
                        };
                        job.error = Some(error.message.clone());
                        job.cumulative_usage.latency_ms = elapsed;
                        job.progress.elapsed_s = elapsed as f64 / 1000.0;
                    })
                    .await;
            }
        }
        self.cleanup_sandbox(&id).await;
        self.schedule_next();
    }

    async fn run_worker_exchange(
        &self,
        id: &str,
        job: &mut Job,
        overall: Duration,
    ) -> Result<(), sens_protocol::SensError> {
        let mut worker = self.spawn_worker(job).await?;
        // The packet the worker sees; the provider key stays in the broker.
        let start_payload = json!({
            "type": "start",
            "packet": job.packet,
            "model": self.config.provider.model,
            "budget": job.packet.budget,
        });
        self.worker_send(&mut worker, &start_payload).await?;

        loop {
            let line = timeout(overall, worker.stdout.next_line())
                .await
                .map_err(|_| {
                    runtime_error(
                        "touch_timeout",
                        "Touch job exceeded its time budget",
                        "Increase worker.timeoutS or split the task.",
                    )
                })?;
            let line = match line {
                Ok(Some(line)) => line,
                Ok(None) => {
                    return Err(runtime_error(
                        "touch_worker_exited",
                        "Touch worker exited without completing",
                        "Retry the task; Sens restarts the worker.",
                    ));
                }
                Err(error) => {
                    return Err(runtime_error(
                        "touch_worker_protocol",
                        format!("Touch worker protocol error: {error}"),
                        "Retry the task.",
                    ));
                }
            };
            let message: Value = serde_json::from_str(&line).map_err(|error| {
                runtime_error(
                    "touch_worker_protocol",
                    format!("Invalid Touch worker message: {error}"),
                    "Retry the task.",
                )
            })?;
            match message.get("type").and_then(Value::as_str) {
                Some("model_request") => {
                    let messages = message
                        .get("messages")
                        .cloned()
                        .unwrap_or_else(|| json!([]));
                    let tools = message.get("tools").cloned().unwrap_or_else(|| json!([]));
                    let response = self.provider_chat(id, &messages, &tools).await?;
                    self.worker_send(&mut worker, &response).await?;
                }
                Some("tool_request") => {
                    let tool = message
                        .get("tool")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned();
                    let args = message.get("args").cloned().unwrap_or_else(|| json!({}));
                    let response = self.execute_tool(id, job, &tool, args).await;
                    self.worker_send(&mut worker, &response).await?;
                }
                Some("event") => {
                    if let Some(event) = message.get("event").cloned() {
                        self.jobs
                            .update(id, |job| {
                                if let (Some(t), Some(kind)) = (
                                    event.get("t").and_then(Value::as_f64),
                                    event.get("kind").and_then(Value::as_str),
                                ) {
                                    job.progress.events.push(TouchProgressEvent {
                                        t,
                                        kind: kind.to_owned(),
                                        tool: event
                                            .get("tool")
                                            .and_then(Value::as_str)
                                            .map(str::to_owned),
                                        target: event
                                            .get("target")
                                            .and_then(Value::as_str)
                                            .map(str::to_owned),
                                    });
                                }
                            })
                            .await;
                    }
                }
                Some("complete") => {
                    let mut result: WorkerResult = serde_json::from_value(
                        message.get("result").cloned().unwrap_or_else(|| json!({})),
                    )
                    .map_err(|error| {
                        runtime_error(
                            "touch_worker_protocol",
                            format!("Invalid worker result: {error}"),
                            "Retry the task.",
                        )
                    })?;
                    self.finalize_result(id, &mut result).await;
                    let job_status = match result.status {
                        TouchJobStatus::Partial => TouchJobStatus::Partial,
                        _ => TouchJobStatus::Complete,
                    };
                    // The worker never sees usage accounting: the broker's
                    // cumulative numbers are the only truth.
                    if let Some(job) = self.jobs.get(id).await {
                        result.usage = job.cumulative_usage.clone();
                    }
                    let usage = result.usage.clone();
                    self.jobs
                        .update(id, |job| {
                            job.result =
                                Some(serde_json::to_value(&result).unwrap_or_else(|_| json!({})));
                            job.status = job_status;
                            job.progress.cost_estimate_usd = usage.cost_estimate_usd;
                        })
                        .await;
                    return Ok(());
                }
                Some("failed") => {
                    let reason = message
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown worker failure")
                        .to_owned();
                    return Err(runtime_error(
                        "touch_worker_failed",
                        reason,
                        "Retry the task.",
                    ));
                }
                other => {
                    return Err(runtime_error(
                        "touch_worker_protocol",
                        format!("Unknown worker message type: {other:?}"),
                        "Retry the task.",
                    ));
                }
            }
        }
    }

    async fn worker_send(
        &self,
        worker: &mut WorkerProcess,
        payload: &Value,
    ) -> Result<(), sens_protocol::SensError> {
        let mut encoded = serde_json::to_vec(payload)
            .map_err(|error| runtime_error("touch_protocol_error", error.to_string(), "Retry."))?;
        encoded.push(b'\n');
        worker.stdin.write_all(&encoded).await.map_err(|error| {
            runtime_error(
                "touch_worker_disconnected",
                format!("Touch worker disconnected: {error}"),
                "Retry; Sens restarts the worker.",
            )
        })?;
        worker.stdin.flush().await.map_err(|error| {
            runtime_error(
                "touch_worker_disconnected",
                format!("Touch worker could not receive the message: {error}"),
                "Retry; Sens restarts the worker.",
            )
        })
    }

    async fn spawn_worker(&self, _job: &Job) -> Result<WorkerProcess, sens_protocol::SensError> {
        if !self.config.worker_script.is_file() {
            return Err(runtime_error(
                "touch_adapter_missing",
                format!(
                    "Touch worker was not found at {}",
                    self.config.worker_script.display()
                ),
                "Reinstall or rebuild Sens.",
            ));
        }
        let mut command = Command::new(&self.config.python_executable);
        command
            .arg(&self.config.worker_script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        if let Some(script_dir) = self.config.worker_script.parent() {
            command.env("SENS_TOUCH_ROLES_DIR", script_dir.join("roles"));
        }
        hide_console(&mut command);
        let mut child = command.spawn().map_err(|error| {
            runtime_error(
                "touch_worker_start_failed",
                format!("Could not start Touch worker: {error}"),
                "Check the bundled Python runtime in Sens diagnostics.",
            )
        })?;
        let stdin = child.stdin.take().ok_or_else(|| {
            runtime_error(
                "touch_worker_start_failed",
                "Touch worker has no stdin",
                "Repair the Sens installation.",
            )
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            runtime_error(
                "touch_worker_start_failed",
                "Touch worker has no stdout",
                "Repair the Sens installation.",
            )
        })?;
        let _job_handle = KillOnCloseJob::assign(&child).map_err(|error| {
            runtime_error(
                "touch_worker_start_failed",
                format!("Could not isolate Touch worker: {error}"),
                "Repair the Sens installation.",
            )
        })?;
        Ok(WorkerProcess {
            _child: child,
            _job: _job_handle,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        })
    }

    // -- Provider proxy (broker-owned) --------------------------------------

    async fn provider_chat(
        &self,
        id: &str,
        messages: &Value,
        tools: &Value,
    ) -> Result<Value, sens_protocol::SensError> {
        let key = self.config.provider.api_key();
        if key.is_empty() {
            return Err(runtime_error(
                "touch_provider_key_missing",
                "Touch provider apiKey is empty; the key never leaves the broker.",
                "Add touch.provider.apiKey to the Sens config.",
            ));
        }
        let base = self
            .config
            .provider
            .base_url
            .trim_end_matches('/')
            .to_owned();
        let url = format!("{base}/chat/completions");
        let body = json!({
            "model": self.config.provider.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
        });
        let started = Instant::now();
        let response = self
            .client
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {key}"))
            .header(CONTENT_TYPE, "application/json")
            .header(USER_AGENT, "Sens-Touch/1.4.0")
            .json(&body)
            .send()
            .await
            .map_err(|error| {
                runtime_error(
                    "touch_provider_network",
                    format!("Provider request failed: {error}"),
                    "Check the provider endpoint and network.",
                )
            })?;
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "<unreadable body>".to_owned());
        if !status.is_success() {
            // Do not log the body: it may echo prompt content.
            return Err(runtime_error(
                "touch_provider_error",
                format!("Provider returned HTTP {status}"),
                "Check the provider key, model and quota.",
            ));
        }
        let parsed: Value = serde_json::from_str(&text).map_err(|error| {
            runtime_error(
                "touch_provider_protocol",
                format!("Provider returned non-JSON: {error}"),
                "Check the provider compatibility (OpenAI chat completions).",
            )
        })?;
        let latency = started.elapsed().as_millis() as u64;

        let message = parsed
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let usage = parsed.get("usage").cloned().unwrap_or_else(|| json!({}));
        let prompt_tokens = usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;
        let completion_tokens = usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u32;

        // Cumulative accounting (design v1.1): totals across ALL model calls.
        let mut limit_reason: Option<String> = None;
        self.jobs
            .update(id, |job| {
                job.cumulative_usage.steps += 1;
                job.cumulative_usage.total_input_tokens = job
                    .cumulative_usage
                    .total_input_tokens
                    .saturating_add(prompt_tokens);
                job.cumulative_usage.total_output_tokens = job
                    .cumulative_usage
                    .total_output_tokens
                    .saturating_add(completion_tokens);
                job.cumulative_usage.latency_ms =
                    job.cumulative_usage.latency_ms.saturating_add(latency);
                let budget = &job.packet.budget;
                if job.cumulative_usage.total_input_tokens > budget.max_total_input_tokens {
                    limit_reason = Some("input_tokens".to_owned());
                } else if job.cumulative_usage.total_output_tokens > budget.max_total_output_tokens
                {
                    limit_reason = Some("output_tokens".to_owned());
                } else if prompt_tokens > budget.max_context_tokens {
                    limit_reason = Some("context_tokens".to_owned());
                } else {
                    let cost = prompt_tokens as f64 * self.price_in() / 1e6
                        + completion_tokens as f64 * self.price_out() / 1e6;
                    job.cost_estimate_usd += cost;
                    job.cumulative_usage.cost_estimate_usd = job.cost_estimate_usd;
                    let task_limit = job
                        .packet
                        .budget
                        .max_spend_usd
                        .unwrap_or(self.config.spend.max_per_task_usd);
                    if job.cost_estimate_usd > task_limit {
                        limit_reason = Some("spend".to_owned());
                    }
                }
            })
            .await;

        if let Some(reason) = limit_reason {
            return Ok(json!({
                "type": "limit",
                "reason": reason,
                "message": message,
                "usage": usage,
            }));
        }

        // Reflect the real spend into the daily counter once per call.
        self.spend_today_micro_usd.fetch_add(
            micro_usd(
                prompt_tokens as f64 * self.price_in() / 1e6
                    + completion_tokens as f64 * self.price_out() / 1e6,
            ),
            Ordering::Relaxed,
        );

        Ok(json!({
            "type": "model_response",
            "message": message,
            "usage": usage,
            "cumulative": json!({
                "steps": self.jobs.get(id).await.map(|job| job.cumulative_usage.steps).unwrap_or(0),
                "totalInputTokens": self.jobs.get(id).await.map(|job| job.cumulative_usage.total_input_tokens).unwrap_or(0),
                "totalOutputTokens": self.jobs.get(id).await.map(|job| job.cumulative_usage.total_output_tokens).unwrap_or(0),
                "costEstimateUsd": self.jobs.get(id).await.map(|job| job.cumulative_usage.cost_estimate_usd).unwrap_or(0.0),
            }),
        }))
    }

    // -- Tool executor (broker executes; worker only requests) --------------

    async fn execute_tool(&self, id: &str, job: &Job, tool: &str, args: Value) -> Value {
        match tool {
            "read" => self.tool_read(id, job, args).await,
            "glob" => self.tool_glob(job, args).await,
            "grep" => self.tool_grep(id, job, args).await,
            "write" => self.tool_write(id, job, args).await,
            "web_fetch" => self.tool_web_fetch(id, args).await,
            "web_search" => self.tool_web_search(id, args).await,
            other => json!({
                "type": "tool_result",
                "ok": false,
                "error": format!("unknown tool: {other}"),
            }),
        }
    }

    fn scope_allows(&self, job: &Job, path: &Path) -> bool {
        let scope = &job.packet.scope;
        let base = job
            .packet
            .context
            .as_ref()
            .and_then(|context| context.cwd.as_deref())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let Ok(canonical) = std::fs::canonicalize(path) else {
            return false;
        };
        scope.iter().any(|entry| {
            if entry == "web" {
                return false;
            }
            let pattern = PathBuf::from(entry);
            let absolute = if pattern.is_absolute() {
                pattern.clone()
            } else {
                base.join(&pattern)
            };
            let Ok(prefix) = std::fs::canonicalize(static_prefix(&absolute)) else {
                return false;
            };
            if !canonical.starts_with(&prefix) {
                return false;
            }
            // A scope entry without glob metacharacters names a path prefix
            // (a directory or a single file): containment is the whole rule.
            if !entry.contains('*') && !entry.contains('?') {
                return true;
            }
            // Absolute patterns match the canonical path directly.
            let absolute_text = absolute.to_string_lossy().replace('\\', "/");
            if glob_match(&absolute_text, &canonical) {
                return true;
            }
            // Relative patterns also match against the path relative to the
            // workspace base (same coordinate system as the pattern).
            if !pattern.is_absolute()
                && let Ok(base_canonical) = std::fs::canonicalize(&base)
                && let Ok(relative) = canonical.strip_prefix(&base_canonical)
            {
                return glob_match(entry, relative);
            }
            false
        })
    }

    fn issue_receipt(
        &self,
        kind: EvidenceKind,
        path: Option<&Path>,
        url: Option<&str>,
        range: Option<[u32; 2]>,
        bytes: &[u8],
        snippet: String,
    ) -> EvidenceReceipt {
        EvidenceReceipt {
            evidence_id: format!("ev_{}", uuid::Uuid::new_v4().simple()),
            kind,
            path: path.map(|path| path.to_string_lossy().into_owned()),
            url: url.map(str::to_owned),
            range,
            sha256: sha256_hex(bytes),
            observed_at: now_rfc3339(),
            snippet,
        }
    }

    async fn tool_read(&self, id: &str, job: &Job, args: Value) -> Value {
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return tool_error("read requires path");
        };
        let path = PathBuf::from(path);
        if !self.scope_allows(job, &path) {
            return tool_error(format!("path outside scope: {}", path.display()));
        }
        let Ok(bytes) = std::fs::read(&path) else {
            return tool_error(format!("cannot read {}", path.display()));
        };
        if bytes.len() as u64 > MAX_FILE_READ_BYTES {
            return tool_error(format!(
                "file too large (>{MAX_FILE_READ_BYTES} bytes): {}",
                path.display()
            ));
        }
        let Ok(text) = String::from_utf8(bytes.clone()) else {
            return tool_error(format!("binary or non-UTF8 file: {}", path.display()));
        };
        let range: Option<[u32; 2]> = match (args.get("startLine"), args.get("endLine")) {
            (Some(start), Some(end)) => {
                let start = start.as_u64().unwrap_or(1).max(1) as u32;
                let end = end.as_u64().unwrap_or(start as u64).max(start as u64) as u32;
                Some([start, end])
            }
            _ => None,
        };
        let snippet = match range {
            Some([start, end]) => text
                .lines()
                .skip((start - 1) as usize)
                .take((end - start + 1) as usize)
                .collect::<Vec<_>>()
                .join("\n"),
            None => text.clone(),
        };
        let receipt = self.issue_receipt(
            EvidenceKind::FileRead,
            Some(&path),
            None,
            range,
            &bytes,
            snippet.clone(),
        );
        self.jobs
            .update(id, |job| job.receipts.push(receipt.clone()))
            .await;
        json!({
            "type": "tool_result",
            "ok": true,
            "result": { "text": snippet, "path": path.to_string_lossy(), "range": range, "bytes": bytes.len() },
            "evidence": receipt,
        })
    }

    async fn tool_glob(&self, job: &Job, args: Value) -> Value {
        let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
            return tool_error("glob requires pattern");
        };
        let base = job
            .packet
            .context
            .as_ref()
            .and_then(|context| context.cwd.as_deref())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let pattern_path = PathBuf::from(pattern);
        let absolute = if pattern_path.is_absolute() {
            pattern_path
        } else {
            base.join(pattern_path)
        };
        if !self.scope_allows(job, &absolute) {
            return tool_error(format!("pattern outside scope: {pattern}"));
        }
        let mut results = Vec::new();
        let prefix = static_prefix(&absolute);
        walk_glob(&absolute, &mut results, MAX_GLOB_RESULTS);
        if results.is_empty() && prefix.is_dir() && pattern.ends_with("/**") {
            if let Ok(entries) = std::fs::read_dir(&prefix) {
                for entry in entries.flatten() {
                    if results.len() >= MAX_GLOB_RESULTS {
                        break;
                    }
                    if let Some(name) = entry.file_name().to_str() {
                        results.push(name.to_owned());
                    }
                }
            }
        }
        json!({
            "type": "tool_result",
            "ok": true,
            "result": { "entries": results, "count": results.len(), "truncated": results.len() >= MAX_GLOB_RESULTS },
            "evidence": null,
        })
    }

    async fn tool_grep(&self, id: &str, job: &Job, args: Value) -> Value {
        let Some(pattern) = args.get("pattern").and_then(Value::as_str) else {
            return tool_error("grep requires pattern");
        };
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return tool_error("grep requires path");
        };
        let Ok(regex) = regex::Regex::new(pattern) else {
            return tool_error(format!("invalid pattern: {pattern}"));
        };
        let mut matched_lines = Vec::new();
        let mut total_bytes = 0u64;
        let base = job
            .packet
            .context
            .as_ref()
            .and_then(|context| context.cwd.as_deref())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let target = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            base.join(path)
        };
        let mut files = Vec::new();
        if target.is_file() {
            files.push(target);
        } else {
            walk_files(&target, &mut files, 50);
        }
        'outer: for file in files {
            if !self.scope_allows(job, &file) {
                continue;
            }
            let Ok(metadata) = std::fs::metadata(&file) else {
                continue;
            };
            if metadata.len() > MAX_GREP_FILE_BYTES {
                continue;
            }
            let Ok(bytes) = std::fs::read(&file) else {
                continue;
            };
            total_bytes = total_bytes.saturating_add(bytes.len() as u64);
            if total_bytes > MAX_GREP_FILE_BYTES * 4 {
                break;
            }
            let Ok(text) = String::from_utf8(bytes) else {
                continue;
            };
            for (index, line) in text.lines().enumerate() {
                if regex.is_match(line) {
                    matched_lines.push(json!({
                        "path": file.to_string_lossy(),
                        "line": index + 1,
                        "text": line.chars().take(400).collect::<String>(),
                    }));
                    if matched_lines.len() >= MAX_GREP_MATCHES {
                        break 'outer;
                    }
                }
            }
        }
        let snippet = matched_lines
            .iter()
            .map(|item| {
                format!(
                    "{}:{}: {}",
                    item.get("path").and_then(Value::as_str).unwrap_or(""),
                    item.get("line").and_then(Value::as_u64).unwrap_or(0),
                    item.get("text").and_then(Value::as_str).unwrap_or("")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let receipt = self.issue_receipt(
            EvidenceKind::Grep,
            None,
            None,
            None,
            snippet.as_bytes(),
            snippet.clone(),
        );
        self.jobs
            .update(id, |job| job.receipts.push(receipt.clone()))
            .await;
        json!({
            "type": "tool_result",
            "ok": true,
            "result": { "matches": matched_lines, "count": matched_lines.len() },
            "evidence": receipt,
        })
    }

    async fn tool_write(&self, id: &str, job: &Job, args: Value) -> Value {
        if job.role != TouchRole::Coder {
            return tool_error("write is only available to the coder role");
        }
        let Some(path) = args.get("path").and_then(Value::as_str) else {
            return tool_error("write requires path");
        };
        let Some(content) = args.get("content").and_then(Value::as_str) else {
            return tool_error("write requires content");
        };
        let sandbox = self.ensure_sandbox(id, job).await;
        let Some(sandbox) = sandbox else {
            return tool_error("sandbox unavailable");
        };
        let Ok(target) = std::fs::canonicalize(&sandbox) else {
            return tool_error("sandbox unavailable");
        };
        let candidate = sandbox.join(path.trim_start_matches(['/', '\\']));
        let Ok(candidate_canonical) = std::fs::canonicalize(candidate.parent().unwrap_or(&sandbox))
        else {
            return tool_error("write target directory missing or outside sandbox");
        };
        if !candidate_canonical.starts_with(&target) {
            return tool_error("write outside sandbox");
        }
        let bytes = content.as_bytes();
        if bytes.len() as u64 > MAX_FILE_READ_BYTES {
            return tool_error(format!("file too large (>{MAX_FILE_READ_BYTES} bytes)"));
        }
        if let Err(error) = std::fs::write(&candidate, bytes) {
            return tool_error(format!("write failed: {error}"));
        }
        let receipt = self.issue_receipt(
            EvidenceKind::SandboxWrite,
            Some(&candidate),
            None,
            None,
            bytes,
            String::new(),
        );
        self.jobs
            .update(id, |job| job.receipts.push(receipt.clone()))
            .await;
        json!({
            "type": "tool_result",
            "ok": true,
            "result": { "path": candidate.to_string_lossy(), "bytes": bytes.len() },
            "evidence": receipt,
        })
    }

    async fn ensure_sandbox(&self, id: &str, job: &Job) -> Option<PathBuf> {
        let root = self.config.sandbox.root.clone().unwrap_or_else(|| {
            RuntimePaths::discover()
                .data_dir
                .join("touch")
                .join("sandboxes")
        });
        let sandbox = root.join(id);
        if let Some(existing) = self.jobs.get(id).await.and_then(|job| job.sandbox.clone()) {
            return Some(existing);
        }
        if std::fs::create_dir_all(&sandbox).is_err() {
            return None;
        }
        // Copy scope files and minimal dependencies into the sandbox.
        let base = job
            .packet
            .context
            .as_ref()
            .and_then(|context| context.cwd.as_deref())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        let mut originals: HashMap<String, (String, Vec<u8>)> = HashMap::new();
        for entry in &job.packet.scope {
            if entry == "web" {
                continue;
            }
            let pattern = PathBuf::from(entry);
            let absolute = if pattern.is_absolute() {
                pattern
            } else {
                base.join(pattern)
            };
            let prefix = static_prefix(&absolute);
            let mut files = Vec::new();
            walk_files(&prefix, &mut files, 200);
            for file in files {
                if !self.scope_allows(job, &file) {
                    continue;
                }
                let Ok(bytes) = std::fs::read(&file) else {
                    continue;
                };
                let Ok(relative) = file.strip_prefix(&prefix) else {
                    continue;
                };
                let destination = sandbox.join(relative);
                if let Some(parent) = destination.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                if std::fs::write(&destination, &bytes).is_err() {
                    continue;
                }
                originals.insert(
                    destination.to_string_lossy().into_owned(),
                    (file.to_string_lossy().into_owned(), bytes),
                );
            }
        }
        for dependency in &self.config.sandbox.copy_dependencies {
            let candidate = base.join(dependency);
            if candidate.is_file() {
                let destination = sandbox.join(dependency);
                if let Ok(bytes) = std::fs::read(&candidate) {
                    let _ = std::fs::write(&destination, &bytes);
                }
            }
        }
        self.jobs
            .update(id, |job| {
                job.sandbox = Some(sandbox.clone());
                job.sandbox_originals = originals;
            })
            .await;
        Some(sandbox)
    }

    async fn cleanup_sandbox(&self, id: &str) {
        if let Some(job) = self.jobs.get(id).await {
            if job.no_store {
                if let Some(sandbox) = job.sandbox {
                    let _ = std::fs::remove_dir_all(sandbox);
                }
            }
        }
    }

    async fn tool_web_fetch(&self, id: &str, args: Value) -> Value {
        let Some(url) = args.get("url").and_then(Value::as_str) else {
            return tool_error("web_fetch requires url");
        };
        if let Err(error) = validate_web_url(url) {
            return tool_error(error);
        }
        let started = Instant::now();
        let response = timeout(WEB_FETCH_TIMEOUT, self.client.get(url).send()).await;
        let response = match response {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => return tool_error(format!("fetch failed: {error}")),
            Err(_) => return tool_error("fetch timed out"),
        };
        let status = response.status();
        if !status.is_success() {
            return tool_error(format!("fetch returned HTTP {status}"));
        }
        let bytes = match response.bytes().await {
            Ok(bytes) => bytes,
            Err(error) => return tool_error(format!("fetch body failed: {error}")),
        };
        if bytes.len() as u64 > MAX_WEB_FETCH_BYTES {
            return tool_error(format!("page too large (>{MAX_WEB_FETCH_BYTES} bytes)"));
        }
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let snippet: String = text.chars().take(2_000).collect();
        let job = self.jobs.get(id).await;
        let receipt = job.as_ref().map(|_job| {
            self.issue_receipt(
                EvidenceKind::WebFetch,
                None,
                Some(url),
                None,
                &bytes,
                snippet.clone(),
            )
        });
        if let Some(receipt) = receipt.clone() {
            self.jobs.update(id, |job| job.receipts.push(receipt)).await;
        }
        json!({
            "type": "tool_result",
            "ok": true,
            "result": {
                "url": url,
                "status": status.as_u16(),
                "text": snippet,
                "bytes": bytes.len(),
                "elapsedMs": started.elapsed().as_millis(),
            },
            "evidence": receipt,
        })
    }

    async fn tool_web_search(&self, id: &str, args: Value) -> Value {
        let key = self.config.web_search.api_key();
        if key.is_empty() {
            return json!({
                "type": "tool_result",
                "ok": false,
                "error": "web_search is disabled: no webSearch.apiKey in the Sens config; web_fetch still works for explicit URLs",
            });
        }
        let Some(query) = args.get("query").and_then(Value::as_str) else {
            return tool_error("web_search requires query");
        };
        let provider = self.config.web_search.provider.to_ascii_lowercase();
        let endpoint = match provider.as_str() {
            "tavily" => "https://api.tavily.com/search",
            "serpapi" => "https://serpapi.com/search",
            "brave" => "https://api.search.brave.com/res/v1/web/search",
            other => return tool_error(format!("unsupported webSearch.provider: {other}")),
        };
        let mut request = self.client.post(endpoint);
        match provider.as_str() {
            "tavily" => {
                request =
                    request.json(&json!({ "api_key": key, "query": query, "max_results": 10 }))
            }
            "serpapi" => {
                request = request.query(&[("engine", "google"), ("api_key", key), ("q", query)])
            }
            _ => {
                request = request
                    .header("X-Subscription-Token", key)
                    .query(&[("q", query), ("count", "10")])
            }
        }
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) => return tool_error(format!("search failed: {error}")),
        };
        let status = response.status();
        let text = match response.text().await {
            Ok(text) => text,
            Err(error) => return tool_error(format!("search body failed: {error}")),
        };
        if !status.is_success() {
            return tool_error(format!("search returned HTTP {status}"));
        }
        let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({}));
        let results = match provider.as_str() {
            "tavily" => parsed.get("results").cloned().unwrap_or_else(|| json!([])),
            _ => parsed.get("organic").cloned().unwrap_or_else(|| json!([])),
        };
        let results_text = results
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        format!(
                            "- {}: {}",
                            item.get("title").and_then(Value::as_str).unwrap_or(""),
                            item.get("url").and_then(Value::as_str).unwrap_or("")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let receipt_text = results_text.clone();
        let job = self.jobs.get(id).await;
        let receipt = job.as_ref().map(|_job| {
            let receipt_snippet = receipt_text.clone();
            let receipt_bytes = receipt_snippet.as_bytes().to_vec();
            self.issue_receipt(
                EvidenceKind::WebSearch,
                None,
                None,
                None,
                &receipt_bytes,
                receipt_snippet,
            )
        });
        if let Some(receipt) = receipt.clone() {
            self.jobs.update(id, |job| job.receipts.push(receipt)).await;
        }
        json!({
            "type": "tool_result",
            "ok": true,
            "result": { "results": results, "count": results.as_array().map(|items| items.len()).unwrap_or(0) },
            "evidence": receipt,
        })
    }

    // -- Result finalization and verification --------------------------------

    async fn finalize_result(&self, id: &str, result: &mut WorkerResult) {
        let job = self.jobs.get(id).await;
        let Some(job) = job else { return };
        let issued: std::collections::HashSet<String> = job
            .receipts
            .iter()
            .map(|receipt| receipt.evidence_id.clone())
            .collect();
        let mut warnings = Vec::new();
        for claim in &mut result.claims {
            for evidence in &mut claim.evidence {
                if issued.contains(&evidence.evidence_id) {
                    // Receipts are machine-issued: an existing receipt is
                    // verified evidence under this claim.
                    evidence.evidence_status = EvidenceStatus::Verified;
                } else {
                    evidence.evidence_status = EvidenceStatus::Unverifiable;
                    warnings.push(format!(
                        "claim references evidence_id {} which was never issued to this job",
                        evidence.evidence_id
                    ));
                }
            }
            if claim.claim_status == ClaimStatus::Verified {
                let all_verified = !claim.evidence.is_empty()
                    && claim
                        .evidence
                        .iter()
                        .all(|evidence| evidence.evidence_status == EvidenceStatus::Verified);
                if !all_verified {
                    claim.claim_status = ClaimStatus::Inferred;
                    warnings.push(format!(
                        "claim \"{}\" downgraded from verified to inferred: evidence missing or unverified",
                        claim.claim.chars().take(80).collect::<String>()
                    ));
                }
            }
            for error in claim.validation_errors() {
                warnings.push(error);
            }
        }
        result.job_id = id.to_owned();
        result.status = TouchJobStatus::Complete;
        result.warnings.extend(warnings);
        // Coder: attach the broker-generated patch.
        if result.role == TouchRole::Coder {
            if let Some(patch) = self.build_sandbox_patch(id).await {
                result.recommended_action = Some(patch);
            }
        }
    }

    async fn build_sandbox_patch(&self, id: &str) -> Option<String> {
        let job = self.jobs.get(id).await?;
        let sandbox = job.sandbox.clone()?;
        let mut patch_parts = Vec::new();
        for (destination, (original_path, original_bytes)) in &job.sandbox_originals {
            let current = std::fs::read(destination).ok()?;
            if current == *original_bytes {
                continue;
            }
            let original_text = String::from_utf8_lossy(original_bytes).into_owned();
            let current_text = String::from_utf8_lossy(&current).into_owned();
            let diff = similar::TextDiff::from_lines(&original_text, &current_text);
            let mut lines = Vec::new();
            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    similar::ChangeTag::Delete => "-",
                    similar::ChangeTag::Insert => "+",
                    similar::ChangeTag::Equal => " ",
                };
                lines.push(format!("{sign}{}", change.value().trim_end_matches('\n')));
            }
            patch_parts.push(format!(
                "--- a/{}\n+++ b/{}\n{}",
                original_path,
                destination,
                lines.join("\n")
            ));
        }
        // New files created by the worker inside the sandbox.
        if let Ok(entries) = walk_all_files(&sandbox) {
            for file in entries {
                let key = file.to_string_lossy().into_owned();
                if job.sandbox_originals.contains_key(&key) {
                    continue;
                }
                if let Ok(bytes) = std::fs::read(&file) {
                    let text = String::from_utf8_lossy(&bytes).into_owned();
                    patch_parts.push(format!(
                        "--- /dev/null\n+++ b/{}\n+{}",
                        key,
                        text.lines().collect::<Vec<_>>().join("\n+")
                    ));
                }
            }
        }
        if patch_parts.is_empty() {
            None
        } else {
            Some(patch_parts.join("\n"))
        }
    }

    // -- sens_touch_check: machine predicates, no LLM, no spend ---------------

    async fn check(&self, input: TouchCheckRequest) -> Result<Value, sens_protocol::SensError> {
        if input.assertions.is_empty() {
            return Err(runtime_error(
                "touch_check_empty",
                "check requires at least one assertion",
                "Pass assertions.",
            ));
        }
        let mut results = Vec::new();
        for assertion in input.assertions {
            let (status, detail) = self.evaluate_predicate(&assertion).await;
            results.push(PredicateResult {
                assertion,
                status,
                detail,
            });
        }
        serde_json::to_value(results)
            .map_err(|error| runtime_error("touch_protocol_error", error.to_string(), "Retry."))
    }

    async fn evaluate_predicate(&self, predicate: &Predicate) -> (EvidenceStatus, Option<String>) {
        match predicate {
            Predicate::FileExists { path } => {
                if Path::new(path).is_file() {
                    (EvidenceStatus::Verified, Some("file exists".into()))
                } else {
                    (EvidenceStatus::Refuted, Some("file does not exist".into()))
                }
            }
            Predicate::LineContains { path, line, value } => {
                let bytes = std::fs::read(path).ok();
                let Some(bytes) = bytes else {
                    return (
                        EvidenceStatus::Unverifiable,
                        Some("cannot read file".into()),
                    );
                };
                if bytes.len() as u64 > MAX_FILE_READ_BYTES {
                    return (EvidenceStatus::Unverifiable, Some("file too large".into()));
                }
                let Ok(text) = String::from_utf8(bytes) else {
                    return (EvidenceStatus::Unverifiable, Some("binary file".into()));
                };
                match text.lines().nth((line.saturating_sub(1)) as usize) {
                    Some(line_text) if line_text.contains(value) => {
                        (EvidenceStatus::Verified, Some("line contains value".into()))
                    }
                    Some(_) => (
                        EvidenceStatus::Refuted,
                        Some("line does not contain value".into()),
                    ),
                    None => (EvidenceStatus::Refuted, Some("line out of range".into())),
                }
            }
            Predicate::PatternCount {
                path,
                pattern,
                min,
                max,
            } => {
                let bytes = std::fs::read(path).ok();
                let Some(bytes) = bytes else {
                    return (
                        EvidenceStatus::Unverifiable,
                        Some("cannot read file".into()),
                    );
                };
                if bytes.len() as u64 > MAX_FILE_READ_BYTES {
                    return (EvidenceStatus::Unverifiable, Some("file too large".into()));
                }
                let Ok(text) = String::from_utf8(bytes) else {
                    return (EvidenceStatus::Unverifiable, Some("binary file".into()));
                };
                let Ok(regex) = regex::Regex::new(pattern) else {
                    return (EvidenceStatus::Unverifiable, Some("invalid pattern".into()));
                };
                let count = regex.find_iter(&text).count() as u32;
                let within =
                    min.is_none_or(|min| count >= min) && max.is_none_or(|max| count <= max);
                if within {
                    (EvidenceStatus::Verified, Some(format!("count={count}")))
                } else {
                    (EvidenceStatus::Refuted, Some(format!("count={count}")))
                }
            }
            Predicate::UrlContainsQuote { url, quote } => {
                if let Err(error) = validate_web_url(url) {
                    return (EvidenceStatus::Unverifiable, Some(error));
                }
                let response = timeout(WEB_FETCH_TIMEOUT, self.client.get(url).send()).await;
                let response = match response {
                    Ok(Ok(response)) => response,
                    _ => return (EvidenceStatus::Unverifiable, Some("fetch failed".into())),
                };
                if !response.status().is_success() {
                    return (EvidenceStatus::Unverifiable, Some("HTTP error".into()));
                }
                let bytes = match response.bytes().await {
                    Ok(bytes) => bytes,
                    Err(_) => return (EvidenceStatus::Unverifiable, Some("body failed".into())),
                };
                if bytes.len() as u64 > MAX_WEB_FETCH_BYTES {
                    return (EvidenceStatus::Unverifiable, Some("page too large".into()));
                }
                let text = String::from_utf8_lossy(&bytes);
                if text.contains(quote) {
                    (EvidenceStatus::Verified, Some("quote found".into()))
                } else {
                    (EvidenceStatus::Refuted, Some("quote not found".into()))
                }
            }
            Predicate::HashMatches { path, sha256 } => {
                let bytes = std::fs::read(path).ok();
                let Some(bytes) = bytes else {
                    return (
                        EvidenceStatus::Unverifiable,
                        Some("cannot read file".into()),
                    );
                };
                let actual = sha256_hex(&bytes);
                if actual == *sha256 {
                    (EvidenceStatus::Verified, Some("hash matches".into()))
                } else {
                    (EvidenceStatus::Refuted, Some(format!("actual {actual}")))
                }
            }
        }
    }
}

#[async_trait]
impl CapabilityExecutor for TouchExecutor {
    async fn invoke(
        &self,
        request: &sens_protocol::InvokeRequest,
    ) -> Result<CapabilityOutput, sens_protocol::SensError> {
        let started = Instant::now();
        info!(
            request_id = %request.request_id,
            operation = %request.operation,
            "Touch request received"
        );
        let data = self.dispatch(request).await?;
        Ok(CapabilityOutput {
            data,
            artifacts: Vec::new(),
            provenance: Vec::new(),
            usage: json!({ "protocolVersion": TOUCH_PROTOCOL_VERSION }),
            warnings: vec![format!("elapsed_ms={}", started.elapsed().as_millis())],
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_input<T: serde::de::DeserializeOwned>(
    request: &sens_protocol::InvokeRequest,
) -> Result<T, sens_protocol::SensError> {
    serde_json::from_value(request.input.clone()).map_err(|error| {
        runtime_error(
            "touch_invalid_input",
            format!("Invalid input: {error}"),
            "Check the tool arguments.",
        )
    })
}

fn tool_error(message: impl Into<String>) -> Value {
    json!({ "type": "tool_result", "ok": false, "error": message.into() })
}

/// Longest static (non-glob) prefix of a pattern.
fn static_prefix(pattern: &Path) -> PathBuf {
    let mut prefix = PathBuf::new();
    for component in pattern.components() {
        let text = component.as_os_str().to_string_lossy();
        if text.contains('*') || text.contains('?') {
            break;
        }
        prefix.push(component);
    }
    prefix
}

/// Minimal glob matcher: supports `**` (any depth), `*` (within a segment),
/// `?` (single char). Paths are normalized to forward slashes.
fn glob_match(pattern: &str, path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('\\', "/");
    let pattern = pattern.replace('\\', "/");
    let mut regex_source = String::new();
    let mut chars = pattern.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '*' => {
                if chars.peek() == Some(&'*') {
                    chars.next();
                    regex_source.push_str(".*");
                } else {
                    regex_source.push_str("[^/]*");
                }
            }
            '?' => regex_source.push_str("[^/]"),
            other => regex_source.push_str(&regex::escape(&other.to_string())),
        }
    }
    regex::Regex::new(&format!("^(?:{regex_source})$"))
        .map(|regex| regex.is_match(&normalized))
        .unwrap_or(false)
}

fn walk_files(root: &Path, output: &mut Vec<PathBuf>, limit: usize) {
    if output.len() >= limit {
        return;
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        if output.len() >= limit {
            return;
        }
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            walk_files(&path, output, limit);
        } else if file_type.is_file() {
            output.push(path);
        }
    }
}

fn walk_all_files(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut output = Vec::new();
    walk_files(root, &mut output, 1_000);
    Ok(output)
}

fn walk_glob(pattern: &Path, output: &mut Vec<String>, limit: usize) {
    let prefix = static_prefix(pattern);
    let mut files = Vec::new();
    walk_files(&prefix, &mut files, limit);
    let pattern_text = pattern.to_string_lossy().replace('\\', "/");
    for file in files {
        if output.len() >= limit {
            break;
        }
        if glob_match(&pattern_text, &file) {
            output.push(file.to_string_lossy().into_owned());
        }
    }
}

/// URL policy for model-controlled web access (SSRF): https only, no
/// private/loopback/link-local/multicast/reserved hosts. Provider endpoints
/// are a separate user-chosen trust boundary and never reach this check.
fn validate_web_url(url: &str) -> Result<(), String> {
    let parsed = reqwest::Url::parse(url).map_err(|error| format!("invalid URL: {error}"))?;
    if parsed.scheme() != "https" {
        return Err("web_fetch allows https only".to_owned());
    }
    let host = parsed.host_str().unwrap_or_default();
    if host.is_empty() {
        return Err("URL has no host".to_owned());
    }
    let lower = host.to_ascii_lowercase();
    if lower == "localhost" || lower.ends_with(".localhost") {
        return Err("loopback host is not allowed".to_owned());
    }
    if let Ok(ip) = host.parse::<std::net::IpAddr>() {
        if is_private_ip(ip) {
            return Err(format!("host {host} is not public"));
        }
    }
    // Hostnames that resolve to private ranges are rejected by IP-literal
    // checks only; DNS-resolution-based rejection is a documented v1 gap.
    Ok(())
}

fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(ipv4) => {
            ipv4.is_loopback()
                || ipv4.is_private()
                || ipv4.is_link_local()
                || ipv4.is_multicast()
                || ipv4.is_broadcast()
                || ipv4.is_unspecified()
                || ipv4.octets()[0] >= 224
        }
        std::net::IpAddr::V6(ipv6) => {
            ipv6.is_loopback()
                || ipv6.is_unspecified()
                || ipv6.is_multicast()
                || (ipv6.segments()[0] & 0xfe00) == 0xfc00 // fc00::/7 unique-local
                || (ipv6.segments()[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matching_covers_scope_patterns() {
        let cases = [
            ("src/network/**", "src/network/socket.ts", true),
            ("src/network/**", "src/hooks/useSocket.ts", false),
            ("src/**", "src/client/limiter.ts", true),
            ("src/client/*", "src/client/api.ts", true),
            ("src/client/*", "src/client/deep/api.ts", false),
            ("src/network/*", "src/network/socket.ts", true),
            ("src/network/?", "src/network/s", true),
            (
                r"D:\work\app\src\network\**",
                r"D:\work\app\src\network\socket.ts",
                true,
            ),
        ];
        for (pattern, path, expected) in cases {
            assert_eq!(
                glob_match(pattern, Path::new(path)),
                expected,
                "pattern {pattern} vs {path}"
            );
        }
    }

    #[test]
    fn static_prefix_splits_before_first_glob() {
        assert_eq!(
            static_prefix(Path::new(r"D:\work\app\src\network\**")),
            Path::new(r"D:\work\app\src\network")
        );
        assert_eq!(
            static_prefix(Path::new(r"D:\work\app\src\*.ts")),
            Path::new(r"D:\work\app\src")
        );
    }

    #[test]
    fn url_policy_rejects_private_and_accepts_public() {
        for url in [
            "http://127.0.0.1:8080/",
            "http://localhost/",
            "http://10.0.0.1/",
            "http://192.168.1.50:11434/v1/",
            "http://169.254.169.254/",
            "http://224.0.0.1/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "https://192.168.0.1/",
        ] {
            assert!(validate_web_url(url).is_err(), "must reject {url}");
        }
        for url in ["https://example.com/", "https://api.deepseek.com/"] {
            assert!(validate_web_url(url).is_ok(), "must allow {url}");
        }
    }

    #[test]
    fn rfc3339_timestamp_is_well_formed() {
        let stamp = now_rfc3339();
        assert!(stamp.ends_with('Z'));
        assert!(stamp.starts_with("20"));
        assert!(stamp.len() >= 20);
    }

    #[test]
    fn pessimistic_cost_uses_configured_prices() {
        let config = TouchRuntimeConfig {
            provider: ProviderConfig {
                kind: "openrouter".into(),
                base_url: "https://openrouter.ai/api/v1".into(),
                model: "deepseek/deepseek-v4-flash-0731".into(),
                api_key: "sk-test".into(),
                price_per_1m_in: Some(0.08),
                price_per_1m_out: Some(0.252),
            },
            ..config_for_test()
        };
        let executor = TouchExecutor::new(config);
        let budget = TouchBudget::default();
        let estimate = executor.pessimistic_cost(&budget);
        // 15 steps * 24k input * 0.08/1M + 6k output * 0.252/1M
        let expected = 15.0 * 24_000.0 * 0.08 / 1e6 + 6_000.0 * 0.252 / 1e6;
        assert!(
            (estimate - expected).abs() < 1e-9,
            "{estimate} vs {expected}"
        );
        // 15*24000*0.08/1e6 = 0.0288; + 0.001512 = 0.030312
        assert!(estimate > 0.02 && estimate < 0.04);
    }

    #[test]
    fn consent_required_above_threshold() {
        let config = TouchRuntimeConfig {
            spend: SpendLimits {
                confirm_above_usd: 0.01,
                ..SpendLimits::default()
            },
            ..config_for_test()
        };
        let executor = TouchExecutor::new(config);
        let (needs, estimate) = executor.needs_consent(&TouchBudget::default(), TouchConsent::Auto);
        assert!(needs);
        assert!(estimate > 0.01);
        let (confirmed, _) =
            executor.needs_consent(&TouchBudget::default(), TouchConsent::Confirmed);
        assert!(!confirmed);
    }

    #[test]
    fn provider_debug_masks_api_key() {
        let config = TouchRuntimeConfig {
            provider: ProviderConfig {
                kind: "openrouter".into(),
                base_url: "https://openrouter.ai/api/v1".into(),
                model: "deepseek/deepseek-v4-flash-0731".into(),
                api_key: "sk-super-secret-123".into(),
                price_per_1m_in: None,
                price_per_1m_out: None,
            },
            ..config_for_test()
        };
        let debug = format!("{config:?}");
        assert!(
            !debug.contains("sk-super-secret-123"),
            "key leaked in Debug"
        );
        assert!(debug.contains("<masked>"));
    }

    fn config_for_test() -> TouchRuntimeConfig {
        TouchRuntimeConfig {
            enabled: true,
            python_executable: PathBuf::from("python"),
            worker_script: PathBuf::from("touch-worker.py"),
            provider: ProviderConfig {
                kind: "openrouter".into(),
                base_url: "https://openrouter.ai/api/v1".into(),
                model: "deepseek/deepseek-v4-flash-0731".into(),
                api_key: "sk-test".into(),
                price_per_1m_in: None,
                price_per_1m_out: None,
            },
            web_search: WebSearchConfig {
                provider: "tavily".into(),
                api_key: String::new(),
            },
            limits: TouchLimits::default(),
            worker: TouchBudget::default(),
            spend: SpendLimits::default(),
            sandbox: SandboxConfig::default(),
            jobs: JobsConfig::default(),
        }
    }
}
