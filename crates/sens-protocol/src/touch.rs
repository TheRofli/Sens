//! Sens 1.4.0 Touch contracts — frozen at Slice 0 (design v1.1, external review
//! 2026-08-13).
//!
//! Two-axis semantics:
//! - `claim_status`: semantic conclusion of the worker model. Always
//!   `inferred` unless the claim IS a machine-checkable predicate.
//! - `evidence_status`: machine verification of broker-issued evidence
//!   receipts (`verified | refuted | unverifiable`).
//!
//! Evidence is never invented by the worker: the broker issues
//! `EvidenceReceipt` at the moment of the real read/fetch, and workers
//! reference receipts by `evidence_id` only.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const TOUCH_PROTOCOL_VERSION: &str = "0.1.0";
pub const TOUCH_CAPABILITY_ID: &str = "touch";

/// Worker roles shipped in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TouchRole {
    Researcher,
    Explorer,
    Coder,
    Reviewer,
    Critic,
}

/// Expected result shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Auto,
    Research,
    Coding,
}

/// Lifecycle of a Touch job (broker-owned).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TouchJobStatus {
    Queued,
    AwaitingConsent,
    Running,
    Partial,
    Complete,
    Failed,
    Cancelled,
}

/// Semantic status of a worker claim. `Verified` is reserved for
/// machine-checkable predicates; conclusions stay `Inferred`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Inferred,
    Verified,
}

/// Machine verification state of a broker-issued evidence receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceStatus {
    Verified,
    Refuted,
    Unverifiable,
}

/// What the broker actually did to produce a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    FileRead,
    Grep,
    Glob,
    WebFetch,
    WebSearch,
    SandboxWrite,
    Diff,
}

/// Consent mode for spend above the configured threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TouchConsent {
    /// Ask the client when the pessimistic cost estimate exceeds the
    /// configured `confirm_above_usd` threshold.
    Auto,
    /// Explicitly confirm in advance; spend is allowed up to budget limits.
    Confirmed,
}

pub const DEFAULT_MAX_STEPS: u32 = 15;
pub const DEFAULT_MAX_TOTAL_INPUT_TOKENS: u32 = 50_000;
pub const DEFAULT_MAX_TOTAL_OUTPUT_TOKENS: u32 = 6_000;
pub const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 24_000;
pub const DEFAULT_MAX_TOOL_RESULT_TOKENS: u32 = 6_000;
pub const DEFAULT_MAX_SINGLE_TOOL_RESULT_TOKENS: u32 = 2_500;
pub const DEFAULT_TIMEOUT_S: u64 = 180;

/// Worker budget. Token limits are CUMULATIVE across all model calls of the
/// job (usage is summed, not read from the last call).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchBudget {
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,
    #[serde(default = "default_max_total_input_tokens")]
    pub max_total_input_tokens: u32,
    #[serde(default = "default_max_total_output_tokens")]
    pub max_total_output_tokens: u32,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,
    #[serde(default = "default_max_tool_result_tokens")]
    pub max_tool_result_tokens: u32,
    #[serde(default = "default_max_single_tool_result_tokens")]
    pub max_single_tool_result_tokens: u32,
    #[serde(default = "default_timeout_s")]
    pub timeout_s: u64,
    #[serde(default)]
    pub max_spend_usd: Option<f64>,
}

impl Default for TouchBudget {
    fn default() -> Self {
        Self {
            max_steps: DEFAULT_MAX_STEPS,
            max_total_input_tokens: DEFAULT_MAX_TOTAL_INPUT_TOKENS,
            max_total_output_tokens: DEFAULT_MAX_TOTAL_OUTPUT_TOKENS,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            max_tool_result_tokens: DEFAULT_MAX_TOOL_RESULT_TOKENS,
            max_single_tool_result_tokens: DEFAULT_MAX_SINGLE_TOOL_RESULT_TOKENS,
            timeout_s: DEFAULT_TIMEOUT_S,
            max_spend_usd: None,
        }
    }
}

fn default_max_steps() -> u32 {
    DEFAULT_MAX_STEPS
}
fn default_max_total_input_tokens() -> u32 {
    DEFAULT_MAX_TOTAL_INPUT_TOKENS
}
fn default_max_total_output_tokens() -> u32 {
    DEFAULT_MAX_TOTAL_OUTPUT_TOKENS
}
fn default_max_context_tokens() -> u32 {
    DEFAULT_MAX_CONTEXT_TOKENS
}
fn default_max_tool_result_tokens() -> u32 {
    DEFAULT_MAX_TOOL_RESULT_TOKENS
}
fn default_max_single_tool_result_tokens() -> u32 {
    DEFAULT_MAX_SINGLE_TOOL_RESULT_TOKENS
}
fn default_timeout_s() -> u64 {
    DEFAULT_TIMEOUT_S
}
fn default_output_format_auto() -> OutputFormat {
    OutputFormat::Auto
}
fn default_consent_auto() -> TouchConsent {
    TouchConsent::Auto
}
fn default_role_researcher() -> TouchRole {
    TouchRole::Researcher
}
fn default_role_reviewer() -> TouchRole {
    TouchRole::Reviewer
}

/// Non-sensitive task metadata only; never file contents, never secrets.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TaskContext {
    #[serde(default)]
    pub os: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub repo: Option<RepoMeta>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct RepoMeta {
    #[serde(default)]
    pub branch: Option<String>,
    #[serde(default)]
    pub dirty: Option<bool>,
}

/// What the primary sends to the worker. The worker sees ONLY this packet.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchTaskPacket {
    pub packet_id: String,
    pub job_id: String,
    pub role: TouchRole,
    pub objective: String,
    /// Path globs (filesystem) and/or the literal `"web"`.
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub deliverable: Option<String>,
    #[serde(default)]
    pub max_findings: Option<u32>,
    #[serde(default = "default_output_format_auto")]
    pub output_format: OutputFormat,
    #[serde(default)]
    pub budget: TouchBudget,
    #[serde(default)]
    pub context: Option<TaskContext>,
}

/// Broker-issued evidence. Created at the moment of the real read/fetch so
/// later file changes cannot invalidate what the worker actually saw.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceReceipt {
    pub evidence_id: String,
    pub kind: EvidenceKind,
    /// Filesystem path (file/grep/glob/sandbox writes) or absent for web.
    #[serde(default)]
    pub path: Option<String>,
    /// Web URL (web_fetch/web_search) or absent for filesystem evidence.
    #[serde(default)]
    pub url: Option<String>,
    /// 1-based inclusive line range for file evidence.
    #[serde(default)]
    pub range: Option<[u32; 2]>,
    pub sha256: String,
    /// RFC 3339 timestamp of the observation.
    pub observed_at: String,
    pub snippet: String,
}

/// A worker's reference to a receipt the broker issued to it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRef {
    pub evidence_id: String,
    pub evidence_status: EvidenceStatus,
}

/// One claim of the worker result. Semantic conclusions stay `inferred`;
/// only machine-checkable predicates may be `verified`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct Claim {
    pub claim: String,
    pub claim_status: ClaimStatus,
    pub confidence: f64,
    #[serde(default)]
    pub evidence: Vec<EvidenceRef>,
}

impl Claim {
    /// Contract honesty check: a `verified` claim must reference only
    /// machine-verified evidence, and `inferred` claims must carry evidence
    /// refs when they assert anything about the scope.
    pub fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.claim_status == ClaimStatus::Verified {
            if self.evidence.is_empty() {
                errors.push("verified claim has no evidence refs".into());
            }
            if self
                .evidence
                .iter()
                .any(|item| item.evidence_status != EvidenceStatus::Verified)
            {
                errors.push("verified claim references non-verified evidence".into());
            }
        }
        if !(0.0..=1.0).contains(&self.confidence) {
            errors.push("confidence must be within [0.0, 1.0]".into());
        }
        errors
    }
}

/// Cumulative usage across ALL model calls of the job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchUsage {
    pub steps: u32,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub cost_estimate_usd: f64,
    pub latency_ms: u64,
}

/// Standard worker result (output_format auto/research).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct WorkerResult {
    pub job_id: String,
    pub status: TouchJobStatus,
    pub role: TouchRole,
    pub provider: String,
    pub model: String,
    pub conclusion: String,
    pub confidence: f64,
    #[serde(default)]
    pub claims: Vec<Claim>,
    #[serde(default)]
    pub findings: Vec<String>,
    #[serde(default)]
    pub risks: Vec<String>,
    #[serde(default)]
    pub recommended_action: Option<String>,
    #[serde(default)]
    pub unresolved: Vec<String>,
    pub usage: TouchUsage,
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Coder result: patch producer. The patch is generated by the broker from
/// the sandbox diff; tests are recommendations only (no execution in v1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ProposedChange {
    pub file: String,
    pub change: String,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct CodingResult {
    pub job_id: String,
    pub role: TouchRole,
    pub summary: String,
    #[serde(default)]
    pub files_examined: Vec<String>,
    #[serde(default)]
    pub proposed_changes: Vec<ProposedChange>,
    /// Recommendations; the primary decides whether/where to run tests.
    #[serde(default)]
    pub tests_required: Vec<String>,
    /// Unified diff generated by the broker (measured, not model text).
    pub patch: String,
    #[serde(default)]
    pub risks: Vec<String>,
    #[serde(default)]
    pub unresolved: Vec<String>,
}

/// Machine-checkable predicates. Only these may carry `claim_status: verified`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Predicate {
    FileExists {
        path: String,
    },
    LineContains {
        path: String,
        line: u32,
        value: String,
    },
    PatternCount {
        path: String,
        pattern: String,
        #[serde(default)]
        min: Option<u32>,
        #[serde(default)]
        max: Option<u32>,
    },
    UrlContainsQuote {
        url: String,
        quote: String,
    },
    HashMatches {
        path: String,
        sha256: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct PredicateResult {
    pub assertion: Predicate,
    pub status: EvidenceStatus,
    #[serde(default)]
    pub detail: Option<String>,
}

/// Progress surface for sens_touch_status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchProgressEvent {
    pub t: f64,
    pub kind: String,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub target: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchProgress {
    pub step: u32,
    pub max_steps: u32,
    #[serde(default)]
    pub events: Vec<TouchProgressEvent>,
    pub elapsed_s: f64,
    pub cost_estimate_usd: f64,
}

/// Shown when a job waits for spend consent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ConsentRequest {
    pub cost_estimate_usd: f64,
    pub confirm_above_usd: f64,
}

/// sens_touch_status response (fallback path for hosts without MCP Tasks).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchStatusResponse {
    pub job_id: String,
    pub status: TouchJobStatus,
    #[serde(default)]
    pub consent_request: Option<ConsentRequest>,
    #[serde(default)]
    pub progress: Option<TouchProgress>,
    /// Full WorkerResult/CodingResult/opinions payload on terminal states.
    #[serde(default)]
    pub result: Option<Value>,
    /// Failure/cancellation reason on failed/cancelled states.
    #[serde(default)]
    pub error: Option<String>,
}

// --- Tool request contracts (sens_touch*) ---

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchRequest {
    pub role: TouchRole,
    pub objective: String,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub deliverable: Option<String>,
    #[serde(default = "default_output_format_auto")]
    pub output_format: OutputFormat,
    #[serde(default)]
    pub budget: TouchBudget,
    #[serde(default = "default_consent_auto")]
    pub consent: TouchConsent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchParallelRequest {
    pub jobs: Vec<TouchRequest>,
    #[serde(default = "default_consent_auto")]
    pub consent: TouchConsent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchOpinionsRequest {
    pub objective: String,
    /// Explicit perspectives win; otherwise the broker applies role defaults.
    #[serde(default)]
    pub perspectives: Option<Vec<String>>,
    /// Number of candidates when perspectives are not given (default 3).
    #[serde(default)]
    pub perspectives_count: Option<u32>,
    #[serde(default = "default_role_researcher")]
    pub role: TouchRole,
    #[serde(default)]
    pub synthesize: bool,
    #[serde(default)]
    pub budget: TouchBudget,
    #[serde(default = "default_consent_auto")]
    pub consent: TouchConsent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchVerifyRequest {
    pub candidate: String,
    #[serde(default)]
    pub criteria: Vec<String>,
    #[serde(default = "default_role_reviewer")]
    pub role: TouchRole,
    #[serde(default)]
    pub scope: Vec<String>,
    #[serde(default = "default_consent_auto")]
    pub consent: TouchConsent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchStatusRequest {
    pub job_id: String,
    /// Confirm the consent request; resumes a job in `awaiting_consent`.
    #[serde(default)]
    pub consent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchCancelRequest {
    pub job_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TouchCheckRequest {
    pub assertions: Vec<Predicate>,
}

/// Default role perspectives for `sens_touch_opinions` when the primary did
/// not pass explicit ones (frozen at design v1.1).
pub fn default_perspectives(role: TouchRole) -> Vec<String> {
    match role {
        TouchRole::Researcher => vec![
            "evidence-first / authoritative sources".into(),
            "alternative implementations / market approaches".into(),
            "skeptical / contradictions and missing evidence".into(),
        ],
        TouchRole::Critic => vec![
            "correctness / counterexample".into(),
            "reliability / failure modes".into(),
            "complexity / maintainability / hidden assumptions".into(),
        ],
        TouchRole::Reviewer => vec![
            "correctness + regressions".into(),
            "edge cases + failure handling".into(),
            "maintainability + integration impact".into(),
        ],
        // Explorer and Coder have no opinion defaults: opinions are a
        // researcher/reviewer/critic pattern.
        TouchRole::Explorer | TouchRole::Coder => {
            vec!["independent read of the same objective".into()]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_defaults_are_frozen() {
        let budget = TouchBudget::default();
        assert_eq!(budget.max_steps, DEFAULT_MAX_STEPS);
        assert_eq!(budget.max_total_input_tokens, 50_000);
        assert_eq!(budget.max_total_output_tokens, 6_000);
        assert_eq!(budget.max_context_tokens, 24_000);
        assert_eq!(budget.max_tool_result_tokens, 6_000);
        assert_eq!(budget.max_single_tool_result_tokens, 2_500);
        assert_eq!(budget.timeout_s, 180);
        assert_eq!(budget.max_spend_usd, None);
    }

    #[test]
    fn packet_round_trip_preserves_two_axes() {
        let packet = TouchTaskPacket {
            packet_id: "pkt_1".into(),
            job_id: "tch_1".into(),
            role: TouchRole::Explorer,
            objective: "Find the leak".into(),
            scope: vec!["src/network/**".into()],
            constraints: vec!["read-only".into()],
            deliverable: Some("root_cause_report".into()),
            max_findings: Some(5),
            output_format: OutputFormat::Research,
            budget: TouchBudget::default(),
            context: Some(TaskContext {
                os: Some("windows".into()),
                cwd: Some(r"D:\work\app".into()),
                repo: Some(RepoMeta {
                    branch: Some("main".into()),
                    dirty: Some(true),
                }),
            }),
        };
        let encoded = serde_json::to_string(&packet).expect("encode");
        let decoded: TouchTaskPacket = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded.job_id, "tch_1");
        assert_eq!(decoded.role, TouchRole::Explorer);
        assert_eq!(decoded.scope, vec!["src/network/**"]);
        assert_eq!(decoded.budget.max_steps, 15);
    }

    #[test]
    fn verified_claim_requires_verified_evidence() {
        let ok = Claim {
            claim: "line 47 contains setInterval".into(),
            claim_status: ClaimStatus::Verified,
            confidence: 1.0,
            evidence: vec![EvidenceRef {
                evidence_id: "ev_1".into(),
                evidence_status: EvidenceStatus::Verified,
            }],
        };
        assert!(ok.validation_errors().is_empty());

        let no_evidence = Claim {
            claim: "this is the root cause".into(),
            claim_status: ClaimStatus::Verified,
            confidence: 0.9,
            evidence: vec![],
        };
        assert!(!no_evidence.validation_errors().is_empty());

        // Semantic conclusion must stay inferred even with verified evidence.
        let semantic = Claim {
            claim: "this is the root cause".into(),
            claim_status: ClaimStatus::Inferred,
            confidence: 0.82,
            evidence: vec![EvidenceRef {
                evidence_id: "ev_1".into(),
                evidence_status: EvidenceStatus::Verified,
            }],
        };
        assert!(semantic.validation_errors().is_empty());
        assert_ne!(semantic.claim_status, ClaimStatus::Verified);
    }

    #[test]
    fn predicates_round_trip_with_types() {
        let predicates = vec![
            Predicate::FileExists {
                path: "src/a.ts".into(),
            },
            Predicate::LineContains {
                path: "src/a.ts".into(),
                line: 47,
                value: "setInterval".into(),
            },
            Predicate::PatternCount {
                path: "src/**".into(),
                pattern: "clearInterval".into(),
                min: Some(0),
                max: Some(5),
            },
            Predicate::UrlContainsQuote {
                url: "https://example.com".into(),
                quote: "rate limit".into(),
            },
            Predicate::HashMatches {
                path: "a.bin".into(),
                sha256: "9f2c...".into(),
            },
        ];
        let encoded = serde_json::to_string(&predicates).expect("encode");
        let decoded: Vec<Predicate> = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded.len(), 5);
        assert!(matches!(
            decoded[1],
            Predicate::LineContains { line: 47, .. }
        ));
    }

    #[test]
    fn role_default_perspectives_are_frozen() {
        assert_eq!(default_perspectives(TouchRole::Researcher).len(), 3);
        assert_eq!(default_perspectives(TouchRole::Critic).len(), 3);
        assert_eq!(default_perspectives(TouchRole::Reviewer).len(), 3);
        assert_eq!(
            default_perspectives(TouchRole::Researcher)[0],
            "evidence-first / authoritative sources"
        );
    }

    #[test]
    fn status_response_round_trip() {
        let response = TouchStatusResponse {
            job_id: "tch_1".into(),
            status: TouchJobStatus::AwaitingConsent,
            consent_request: Some(ConsentRequest {
                cost_estimate_usd: 0.42,
                confirm_above_usd: 0.20,
            }),
            progress: None,
            result: None,
            error: None,
        };
        let encoded = serde_json::to_string(&response).expect("encode");
        let decoded: TouchStatusResponse = serde_json::from_str(&encoded).expect("decode");
        assert_eq!(decoded.status, TouchJobStatus::AwaitingConsent);
        assert_eq!(decoded.consent_request.unwrap().cost_estimate_usd, 0.42);
    }
}
