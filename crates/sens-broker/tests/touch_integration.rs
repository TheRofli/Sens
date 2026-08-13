//! End-to-end integration: broker -> real touch-worker.py -> mock provider.
//!
//! Slice-2 gate. Verifies the full loop (model_request -> tool -> evidence
//! receipt -> final answer -> claim verification) and that the provider proxy
//! authenticates every call with the broker-held key while the worker never
//! holds the key and never talks to the network itself.

use std::path::PathBuf;

use sens_broker::{TouchExecutor, TouchRuntimeConfig};
use sens_core::CapabilityExecutor;
use sens_protocol::{InvokeRequest, touch};
use serde_json::{Value, json};
use tokio::io::AsyncBufReadExt;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn find_python() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("SENS_PYTHON") {
        return Some(PathBuf::from(path));
    }
    for candidate in ["python", "py", "python3"] {
        if std::process::Command::new(candidate)
            .arg("--version")
            .output()
            .is_ok()
        {
            return Some(PathBuf::from(candidate));
        }
    }
    None
}

fn test_config(python: &std::path::Path, provider_url: &str) -> TouchRuntimeConfig {
    let mut config = TouchRuntimeConfig::discover().expect("config discover");
    config.enabled = true;
    config.python_executable = python.to_path_buf();
    config.worker_script = repo_root()
        .join("sidecars")
        .join("touch")
        .join("touch-worker.py");
    config.provider.base_url = provider_url.trim_end_matches('/').to_owned();
    config.provider.model = "mock/deepseek-v4-flash-0731".into();
    config.provider.api_key = "sk-integration-test-42".into();
    config.provider.price_per_1m_in = Some(0.08);
    config.provider.price_per_1m_out = Some(0.252);
    // Never ask consent in the integration test.
    config.spend.confirm_above_usd = 1000.0;
    config
}

#[tokio::test]
async fn worker_loop_with_mock_provider() {
    let Some(python) = find_python() else {
        eprintln!("skipping: no python interpreter found");
        return;
    };
    let fixtures = repo_root().join("tests").join("touch").join("fixtures");
    let source_file = fixtures.join("source_files").join("useSocket.ts");
    if !source_file.is_file() {
        eprintln!("skipping: fixture source file missing");
        return;
    }

    let temp =
        std::env::temp_dir().join(format!("touch-scenario-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&temp).expect("temp dir");
    let scenario_path = temp.join("scenario.json");
    let log_path = temp.join("requests.jsonl");
    let final_content = r#"{"conclusion": "line 47 contains setInterval", "confidence": 0.9, "claims": [{"claim": "line 47 contains setInterval", "claimStatus": "verified", "confidence": 1.0, "evidence": [{"evidenceId": "{EVIDENCE_ID}", "evidenceStatus": "verified"}]}], "findings": [], "risks": [], "recommendedAction": null, "unresolved": []}"#;
    let scenario = json!([
        {
            "role": "tool_call",
            "tool": "read",
            "args": { "path": source_file.to_string_lossy(), "startLine": 47, "endLine": 47 }
        },
        {
            "role": "final",
            "echo_evidence": true,
            "content": final_content,
            "usage": { "prompt_tokens": 1500, "completion_tokens": 300 }
        }
    ]);
    std::fs::write(
        &scenario_path,
        serde_json::to_vec(&scenario).expect("scenario json"),
    )
    .expect("write scenario");

    let mut provider = tokio::process::Command::new(&python)
        .arg(fixtures.join("mock_provider.py"))
        .arg("--port")
        .arg("0")
        .arg("--scenario")
        .arg(&scenario_path)
        .arg("--log")
        .arg(&log_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn mock provider");
    let stdout = provider.stdout.take().expect("provider stdout");
    let mut lines = tokio::io::BufReader::new(stdout).lines();
    let address = tokio::time::timeout(std::time::Duration::from_secs(10), lines.next_line())
        .await
        .expect("provider address timeout")
        .expect("provider stdout line")
        .expect("provider printed address");
    let base = address.rsplit(' ').next().unwrap_or_default().to_owned();

    let executor = TouchExecutor::new(test_config(&python, &base));

    let request = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "touch",
        json!({
            "role": "explorer",
            "objective": "Read line 47 of useSocket.ts and report what it contains",
            "scope": [fixtures.join("source_files").to_string_lossy()],
            "constraints": ["read-only"],
            "deliverable": "report",
            "outputFormat": "auto",
            "budget": { "maxSteps": 6, "timeoutS": 90 },
            "consent": "confirmed",
        }),
    );
    let created = executor.invoke(&request).await.expect("create job");
    let job_id = created
        .data
        .get("jobId")
        .and_then(Value::as_str)
        .expect("job id")
        .to_owned();

    let mut final_status = String::new();
    for _ in 0..120 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let status_request = InvokeRequest::new(
            touch::TOUCH_CAPABILITY_ID,
            "status",
            json!({ "jobId": job_id }),
        );
        let status = executor.invoke(&status_request).await.expect("status call");
        final_status = status
            .data
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if matches!(
            final_status.as_str(),
            "complete" | "failed" | "cancelled" | "partial"
        ) {
            break;
        }
    }
    provider.kill().await.ok();
    let _ = provider.wait().await;

    let status_request = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "status",
        json!({ "jobId": job_id }),
    );
    let status = executor
        .invoke(&status_request)
        .await
        .expect("final status");
    assert_eq!(
        final_status, "complete",
        "job must complete; full status: {}",
        status.data
    );
    let result = status
        .data
        .get("result")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let claims = result
        .get("claims")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert!(!claims.is_empty(), "worker must return claims");
    let first = claims.first().cloned().unwrap_or_default();
    assert_eq!(
        first.get("claimStatus").and_then(Value::as_str),
        Some("verified"),
        "predicate claim from the worker may stay verified; result: {}",
        result
    );
    let evidence = first
        .get("evidence")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        evidence
            .first()
            .and_then(|item| item.get("evidenceStatus"))
            .and_then(Value::as_str),
        Some("verified"),
        "receipt issued by the broker must verify under the claim"
    );

    // Provider proxy must authenticate every call and never leak the key
    // into request bodies or logs.
    let log = std::fs::read_to_string(&log_path).unwrap_or_default();
    let records: Vec<Value> = log
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    assert!(
        records.len() >= 2,
        "provider must receive at least two model calls"
    );
    assert!(
        records
            .iter()
            .all(|record| record.get("auth_present").and_then(Value::as_bool) == Some(true)),
        "provider proxy must authenticate every call"
    );
    assert!(
        !log.contains("sk-integration-test-42"),
        "key must not leak into provider request bodies or logs"
    );
}

/// sens_touch_check: machine predicates without any worker or spend.
#[tokio::test]
async fn deterministic_check_predicates() {
    let fixtures = repo_root().join("tests").join("touch").join("fixtures");
    let source_file = fixtures.join("source_files").join("useSocket.ts");
    let executor = TouchExecutor::new(test_config(
        &PathBuf::from("python"),
        "https://openrouter.ai/api/v1",
    ));
    let request = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "check",
        json!({
            "assertions": [
                { "type": "file_exists", "path": source_file.to_string_lossy() },
                { "type": "line_contains", "path": source_file.to_string_lossy(), "line": 47, "value": "setInterval" },
                { "type": "line_contains", "path": source_file.to_string_lossy(), "line": 47, "value": "doesNotExist" },
                { "type": "pattern_count", "path": source_file.to_string_lossy(), "pattern": "setInterval", "min": 2, "max": 5 },
                { "type": "hash_matches", "path": source_file.to_string_lossy(), "sha256": "0000000000000000000000000000000000000000000000000000000000000000" },
                { "type": "line_contains", "path": "C:/definitely/missing/file.ts", "line": 1, "value": "x" }
            ]
        }),
    );
    let output = executor.invoke(&request).await.expect("check runs");
    let results = output.data.as_array().expect("check returns a list");
    assert_eq!(results.len(), 6);
    let statuses: Vec<&str> = results
        .iter()
        .filter_map(|item| item.get("status").and_then(Value::as_str))
        .collect();
    assert_eq!(statuses[0], "verified", "file exists");
    assert_eq!(statuses[1], "verified", "line contains setInterval");
    assert_eq!(statuses[2], "refuted", "line does not contain the value");
    assert_eq!(statuses[3], "verified", "pattern count within bounds");
    assert_eq!(statuses[4], "refuted", "hash mismatch");
    assert_eq!(
        statuses[5], "unverifiable",
        "missing file cannot be verified"
    );
}

/// Coder slice: the worker writes inside the sandbox and the broker builds
/// the patch (measured, not model text).
#[tokio::test]
async fn coder_sandbox_produces_broker_patch() {
    let Some(python) = find_python() else {
        eprintln!("skipping: no python interpreter found");
        return;
    };
    let fixtures = repo_root().join("tests").join("touch").join("fixtures");
    let temp = std::env::temp_dir().join(format!("touch-coder-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&temp).expect("temp dir");
    let worktree = temp.join("worktree");
    std::fs::create_dir_all(&worktree).expect("worktree dir");
    std::fs::write(worktree.join("api.ts"), "export const retry = 3;\n").expect("seed file");

    let scenario_path = temp.join("scenario.json");
    let log_path = temp.join("requests.jsonl");
    let final_content = r#"{"conclusion": "candidate written", "confidence": 0.8, "claims": [], "findings": [], "risks": [], "recommendedAction": null, "unresolved": []}"#;
    let scenario = json!([
        {
            "role": "tool_call",
            "tool": "write",
            "args": { "path": "api.ts", "content": "export const retry = 5;\nexport const backoff = 1000;\n" }
        },
        { "role": "final", "echo_evidence": true, "content": final_content,
          "usage": { "prompt_tokens": 900, "completion_tokens": 120 } }
    ]);
    std::fs::write(
        &scenario_path,
        serde_json::to_vec(&scenario).expect("scenario json"),
    )
    .expect("write scenario");

    let mut provider = tokio::process::Command::new(&python)
        .arg(fixtures.join("mock_provider.py"))
        .arg("--port")
        .arg("0")
        .arg("--scenario")
        .arg(&scenario_path)
        .arg("--log")
        .arg(&log_path)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
        .expect("spawn mock provider");
    let stdout = provider.stdout.take().expect("provider stdout");
    let mut lines = tokio::io::BufReader::new(stdout).lines();
    let address = tokio::time::timeout(std::time::Duration::from_secs(10), lines.next_line())
        .await
        .expect("provider address timeout")
        .expect("provider stdout line")
        .expect("provider printed address");
    let base = address.rsplit(' ').next().unwrap_or_default().to_owned();

    let executor = TouchExecutor::new(test_config(&python, &base));
    let request = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "touch",
        json!({
            "role": "coder",
            "objective": "Change retry to 5 and add backoff",
            "scope": [worktree.to_string_lossy()],
            "constraints": ["keep it minimal"],
            "deliverable": "patch",
            "outputFormat": "coding",
            "budget": { "maxSteps": 4, "timeoutS": 90 },
            "consent": "confirmed",
        }),
    );
    let created = executor.invoke(&request).await.expect("create coder job");
    let job_id = created
        .data
        .get("jobId")
        .and_then(Value::as_str)
        .expect("job id")
        .to_owned();

    let mut final_status = String::new();
    for _ in 0..120 {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let status_request = InvokeRequest::new(
            touch::TOUCH_CAPABILITY_ID,
            "status",
            json!({ "jobId": job_id }),
        );
        let status = executor.invoke(&status_request).await.expect("status call");
        final_status = status
            .data
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if matches!(
            final_status.as_str(),
            "complete" | "failed" | "cancelled" | "partial"
        ) {
            break;
        }
    }
    provider.kill().await.ok();
    let _ = provider.wait().await;

    assert_eq!(final_status, "complete", "coder job must complete");
    let status_request = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "status",
        json!({ "jobId": job_id }),
    );
    let status = executor
        .invoke(&status_request)
        .await
        .expect("final status");
    let result = status
        .data
        .get("result")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let patch = result
        .get("recommendedAction")
        .and_then(Value::as_str)
        .unwrap_or_default();
    assert!(
        patch.contains("retry = 5") || patch.contains("+export const retry = 5"),
        "broker must build a patch from the sandbox diff: {patch}"
    );
    // The primary's worktree must stay untouched.
    let original = std::fs::read_to_string(worktree.join("api.ts")).expect("worktree file");
    assert_eq!(
        original, "export const retry = 3;\n",
        "primary worktree untouched"
    );
}

/// Opinions slice: N isolated jobs are created for one objective.
#[tokio::test]
async fn opinions_creates_isolated_jobs() {
    let executor = TouchExecutor::new(test_config(
        &PathBuf::from("python"),
        "https://openrouter.ai/api/v1",
    ));
    let request = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "opinions",
        json!({
            "objective": "Design onboarding",
            "perspectives": ["minimal", "gamified"],
            "role": "researcher",
            "consent": "confirmed",
            "budget": { "maxSteps": 1, "timeoutS": 10 }
        }),
    );
    let output = executor.invoke(&request).await.expect("opinions runs");
    let jobs = output
        .data
        .get("jobs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert_eq!(jobs.len(), 2, "two explicit perspectives -> two jobs");
    let perspectives = output
        .data
        .get("perspectives")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert_eq!(perspectives.len(), 2);

    // Default perspectives by role when only a count is given.
    let request2 = InvokeRequest::new(
        touch::TOUCH_CAPABILITY_ID,
        "opinions",
        json!({ "objective": "x", "perspectivesCount": 3, "role": "critic", "consent": "confirmed" }),
    );
    let output2 = executor.invoke(&request2).await.expect("opinions defaults");
    let jobs2 = output2
        .data
        .get("jobs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    assert_eq!(jobs2.len(), 3, "critic default perspectives are 3");
}
