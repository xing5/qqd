use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::Command;
use std::thread::sleep;
use std::time::Duration;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
struct SampleSummary {
    command: String,
    iterations: usize,
    avg_ms: f64,
    min_ms: u128,
    max_ms: u128,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    query: String,
    iterations: usize,
    result_count: usize,
    qqd: SampleSummary,
    qmd: SampleSummary,
    mcp_qqd: Option<SampleSummary>,
    mcp_qmd: Option<SampleSummary>,
    rust_wins: bool,
}

#[derive(Debug, Deserialize)]
struct BaselineFile {
    search_json: SampleSummary,
    http_query: SampleSummary,
}

#[derive(Debug, Deserialize)]
struct QualityFixture {
    name: String,
    collection: QualityCollection,
    documents: Vec<QualityDocument>,
    cases: Vec<QualityCase>,
    minimum_pass_rate: f64,
}

#[derive(Debug, Deserialize)]
struct QualityCollection {
    name: String,
    context: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QualityDocument {
    path: String,
    body: String,
}

#[derive(Debug, Deserialize)]
struct QualityCase {
    name: String,
    query: String,
    expected_top: Option<String>,
    must_include: Vec<String>,
    top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
struct QualityCaseReport {
    name: String,
    query: String,
    expected_top: Option<String>,
    actual_top: Option<String>,
    actual_paths: Vec<String>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct QualityReport {
    fixture: String,
    passed_cases: usize,
    total_cases: usize,
    pass_rate: f64,
    minimum_pass_rate: f64,
    meets_quality_gate: bool,
    cases: Vec<QualityCaseReport>,
}

#[derive(Debug, Serialize)]
struct MetricsCaseReport {
    name: String,
    query: String,
    top_k: usize,
    relevant: Vec<String>,
    actual_paths: Vec<String>,
    top1_hit: bool,
    accuracy: f64,
    precision_at_k: f64,
    recall_at_k: f64,
    f1_at_k: f64,
    reciprocal_rank: f64,
    ndcg_at_k: f64,
}

#[derive(Debug, Serialize)]
struct MetricsReport {
    fixture: String,
    cases: Vec<MetricsCaseReport>,
    mean_accuracy: f64,
    mean_precision_at_k: f64,
    mean_recall_at_k: f64,
    mean_f1_at_k: f64,
    mean_reciprocal_rank: f64,
    mean_ndcg_at_k: f64,
}

pub fn run(args: &[String]) -> Result<()> {
    let config = BenchConfig::parse(args)?;
    let result = benchmark(&config)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    if !result.rust_wins {
        return Err(anyhow!(
            "qqd did not beat qmd on the selected latency workload"
        ));
    }
    Ok(())
}

pub fn run_quality(args: &[String]) -> Result<()> {
    let fixture_path = args
        .first()
        .ok_or_else(|| anyhow!("usage: qqd bench-quality <fixture.json>"))?;
    let fixture = load_quality_fixture(fixture_path)?;
    let report = benchmark_quality(&fixture)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if !report.meets_quality_gate {
        return Err(anyhow!("qqd quality gate failed for {}", fixture.name));
    }
    Ok(())
}

pub fn run_metrics(args: &[String]) -> Result<()> {
    let fixture_path = args
        .first()
        .ok_or_else(|| anyhow!("usage: qqd bench-metrics <fixture.json>"))?;
    let fixture = load_quality_fixture(fixture_path)?;
    let report = benchmark_metrics(&fixture)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[derive(Debug)]
struct BenchConfig {
    query: String,
    iterations: usize,
    index_name: Option<String>,
    collections: Vec<String>,
}

impl BenchConfig {
    fn parse(args: &[String]) -> Result<Self> {
        if args.is_empty() {
            return Err(anyhow!(
                "usage: qqd bench-latency <query> [--iterations N] [--index NAME] [-c collection]"
            ));
        }

        let mut query_parts = Vec::new();
        let mut iterations = 5usize;
        let mut index_name = None;
        let mut collections = Vec::new();
        let mut idx = 0usize;

        while idx < args.len() {
            match args[idx].as_str() {
                "--iterations" => {
                    let value = args
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--iterations requires a value"))?;
                    iterations = value
                        .parse()
                        .with_context(|| "invalid --iterations value")?;
                    idx += 2;
                }
                "--index" => {
                    let value = args
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--index requires a value"))?;
                    index_name = Some(value.clone());
                    idx += 2;
                }
                "-c" | "--collection" => {
                    let value = args
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--collection requires a value"))?;
                    collections.push(value.clone());
                    idx += 2;
                }
                value => {
                    query_parts.push(value.to_string());
                    idx += 1;
                }
            }
        }

        let query = query_parts.join(" ");
        if query.trim().is_empty() {
            return Err(anyhow!("bench-latency requires a query string"));
        }

        Ok(Self {
            query,
            iterations,
            index_name,
            collections,
        })
    }
}

fn benchmark(config: &BenchConfig) -> Result<BenchmarkResult> {
    let qqd_args = native_search_args(config);
    let qqd_probe = command_output(&mut current_qqd_command(&qqd_args)?)?;
    let qqd_results = parse_json_results(&qqd_probe.stdout)?;
    if qqd_results.is_empty() {
        return Err(anyhow!(
            "benchmark requires a populated index and non-empty search results"
        ));
    }
    let baseline = load_qmd_baseline()?;

    let qqd_samples = (0..config.iterations)
        .map(|_| {
            let mut command = current_qqd_command(&qqd_args)?;
            time_command(&mut command)
        })
        .collect::<Result<Vec<_>>>()?;
    let qqd_summary = summarize("qqd search --json".to_string(), &qqd_samples);
    let qmd_summary = baseline.search_json;
    let mcp_qqd = benchmark_mcp(config).ok();
    let mcp_qmd = Some(baseline.http_query);

    Ok(BenchmarkResult {
        query: config.query.clone(),
        iterations: config.iterations,
        result_count: qqd_results.len(),
        rust_wins: qqd_summary.avg_ms < qmd_summary.avg_ms,
        qqd: qqd_summary,
        qmd: qmd_summary,
        mcp_qqd,
        mcp_qmd,
    })
}

fn native_search_args(config: &BenchConfig) -> Vec<String> {
    let mut args = Vec::new();
    if let Some(index_name) = &config.index_name {
        args.push("--index".to_string());
        args.push(index_name.clone());
    }
    args.push("search".to_string());
    args.push("--json".to_string());
    args.push("-n".to_string());
    args.push("10".to_string());
    for collection in &config.collections {
        args.push("-c".to_string());
        args.push(collection.clone());
    }
    args.push(config.query.clone());
    args
}

fn time_command(command: &mut Command) -> Result<u128> {
    let start = Instant::now();
    let output = command_output(command)?;
    if !output.status.success() {
        return Err(anyhow!(
            "benchmark command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(start.elapsed().as_millis())
}

fn command_output(command: &mut Command) -> Result<std::process::Output> {
    command
        .output()
        .with_context(|| "failed to run benchmark command")
}

fn parse_json_results(stdout: &[u8]) -> Result<Vec<Value>> {
    let text = String::from_utf8_lossy(stdout);
    for (index, _) in text.match_indices('[').rev() {
        if let Ok(value) = serde_json::from_str::<Vec<Value>>(&text[index..]) {
            return Ok(value);
        }
    }
    Err(anyhow!("benchmark command did not return JSON"))
}

fn load_qmd_baseline() -> Result<BaselineFile> {
    serde_json::from_str(include_str!("../benchmarks/qmd-baseline.json"))
        .context("failed to load committed qmd benchmark baseline")
}

fn load_quality_fixture(path: &str) -> Result<QualityFixture> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read quality fixture {path}"))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("failed to parse quality fixture {path}"))
}

fn summarize(command: String, samples: &[u128]) -> SampleSummary {
    let total = samples.iter().sum::<u128>() as f64;
    SampleSummary {
        command,
        iterations: samples.len(),
        avg_ms: total / samples.len() as f64,
        min_ms: *samples.iter().min().unwrap_or(&0),
        max_ms: *samples.iter().max().unwrap_or(&0),
    }
}

fn benchmark_quality(fixture: &QualityFixture) -> Result<QualityReport> {
    let scenario = QualityScenario::create(fixture)?;
    let mut reports = Vec::new();
    for case in &fixture.cases {
        let output = scenario.run_qqd(&["query", "--json", case.query.as_str()])?;
        if !output.status.success() {
            return Err(anyhow!(
                "quality case {} failed: {}",
                case.name,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let results = parse_json_results(&output.stdout)?;
        let actual_paths = results
            .iter()
            .filter_map(|entry| entry.get("file").and_then(Value::as_str))
            .map(|path| path.trim_start_matches("qmd://").to_string())
            .collect::<Vec<_>>();
        let actual_top = actual_paths.first().cloned();
        let top_k = case.top_k.unwrap_or(5).min(actual_paths.len());
        let window = &actual_paths[..top_k];
        let top_ok = case
            .expected_top
            .as_ref()
            .map(|expected| actual_top.as_ref() == Some(expected))
            .unwrap_or(true);
        let includes_ok = case
            .must_include
            .iter()
            .all(|expected| window.iter().any(|actual| actual == expected));
        reports.push(QualityCaseReport {
            name: case.name.clone(),
            query: case.query.clone(),
            expected_top: case.expected_top.clone(),
            actual_top,
            actual_paths,
            passed: top_ok && includes_ok,
        });
    }
    let passed_cases = reports.iter().filter(|case| case.passed).count();
    let total_cases = reports.len();
    let pass_rate = if total_cases == 0 {
        1.0
    } else {
        passed_cases as f64 / total_cases as f64
    };
    Ok(QualityReport {
        fixture: fixture.name.clone(),
        passed_cases,
        total_cases,
        pass_rate,
        minimum_pass_rate: fixture.minimum_pass_rate,
        meets_quality_gate: pass_rate >= fixture.minimum_pass_rate,
        cases: reports,
    })
}

fn benchmark_metrics(fixture: &QualityFixture) -> Result<MetricsReport> {
    let scenario = QualityScenario::create(fixture)?;
    let mut cases = Vec::new();
    for case in &fixture.cases {
        let output = scenario.run_qqd(&["query", "--json", case.query.as_str()])?;
        if !output.status.success() {
            return Err(anyhow!(
                "metrics case {} failed: {}",
                case.name,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let results = parse_json_results(&output.stdout)?;
        let actual_paths = results
            .iter()
            .filter_map(|entry| entry.get("file").and_then(Value::as_str))
            .map(|path| path.trim_start_matches("qmd://").to_string())
            .collect::<Vec<_>>();
        let top_k = case.top_k.unwrap_or(5).max(1);
        let actual_top_k = &actual_paths[..top_k.min(actual_paths.len())];
        let mut relevant = case.must_include.clone();
        if let Some(expected_top) = &case.expected_top {
            if !relevant.iter().any(|value| value == expected_top) {
                relevant.insert(0, expected_top.clone());
            }
        }
        let hits = actual_top_k
            .iter()
            .filter(|path| relevant.iter().any(|rel| rel == *path))
            .count();
        let precision = if actual_top_k.is_empty() {
            0.0
        } else {
            hits as f64 / actual_top_k.len() as f64
        };
        let recall = if relevant.is_empty() {
            1.0
        } else {
            hits as f64 / relevant.len() as f64
        };
        let f1 = if (precision + recall) == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        let top1_hit = actual_paths
            .first()
            .map(|path| relevant.iter().any(|rel| rel == path))
            .unwrap_or(false);
        let reciprocal_rank = actual_paths
            .iter()
            .position(|path| relevant.iter().any(|rel| rel == path))
            .map(|idx| 1.0 / (idx as f64 + 1.0))
            .unwrap_or(0.0);
        let dcg = actual_top_k
            .iter()
            .enumerate()
            .map(|(idx, path)| {
                let rel = if relevant.iter().any(|candidate| candidate == path) {
                    1.0
                } else {
                    0.0
                };
                rel / ((idx as f64) + 2.0).log2()
            })
            .sum::<f64>();
        let ideal_hits = relevant.len().min(top_k);
        let idcg = (0..ideal_hits)
            .map(|idx| 1.0 / ((idx as f64) + 2.0).log2())
            .sum::<f64>();
        let ndcg = if idcg == 0.0 { 1.0 } else { dcg / idcg };
        cases.push(MetricsCaseReport {
            name: case.name.clone(),
            query: case.query.clone(),
            top_k,
            relevant,
            actual_paths,
            top1_hit,
            accuracy: if top1_hit { 1.0 } else { 0.0 },
            precision_at_k: precision,
            recall_at_k: recall,
            f1_at_k: f1,
            reciprocal_rank,
            ndcg_at_k: ndcg,
        });
    }
    let denom = cases.len().max(1) as f64;
    Ok(MetricsReport {
        fixture: fixture.name.clone(),
        mean_accuracy: cases.iter().map(|case| case.accuracy).sum::<f64>() / denom,
        mean_precision_at_k: cases.iter().map(|case| case.precision_at_k).sum::<f64>() / denom,
        mean_recall_at_k: cases.iter().map(|case| case.recall_at_k).sum::<f64>() / denom,
        mean_f1_at_k: cases.iter().map(|case| case.f1_at_k).sum::<f64>() / denom,
        mean_reciprocal_rank: cases.iter().map(|case| case.reciprocal_rank).sum::<f64>() / denom,
        mean_ndcg_at_k: cases.iter().map(|case| case.ndcg_at_k).sum::<f64>() / denom,
        cases,
    })
}

struct QualityScenario {
    root: PathBuf,
    db_path: PathBuf,
    config_dir: PathBuf,
}

impl QualityScenario {
    fn create(fixture: &QualityFixture) -> Result<Self> {
        let root = unique_temp_dir("qqd-quality")?;
        let docs_dir = root.join("docs");
        let config_dir = root.join("config");
        fs::create_dir_all(&docs_dir)?;
        fs::create_dir_all(&config_dir)?;
        fs::write(config_dir.join("index.yml"), "collections: {}\n")?;
        for document in &fixture.documents {
            let full_path = docs_dir.join(&document.path);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(full_path, &document.body)?;
        }

        let db_path = root.join("index.sqlite");
        let scenario = Self {
            root,
            db_path,
            config_dir,
        };
        let add = scenario.run_qqd(&[
            "collection",
            "add",
            docs_dir
                .to_str()
                .ok_or_else(|| anyhow!("invalid docs path"))?,
            "--name",
            fixture.collection.name.as_str(),
        ])?;
        ensure_command_success("bench-quality collection add", &add)?;
        if let Some(context) = &fixture.collection.context {
            let target = format!("qmd://{}/", fixture.collection.name);
            let context_add = scenario.run_qqd(&["context", "add", &target, context])?;
            ensure_command_success("bench-quality context add", &context_add)?;
        }
        let embed = scenario.run_qqd(&["embed"])?;
        ensure_command_success("bench-quality embed", &embed)?;
        Ok(scenario)
    }

    fn run_qqd(&self, args: &[&str]) -> Result<std::process::Output> {
        let mut command = current_qqd_command(args)?;
        command
            .env("INDEX_PATH", &self.db_path)
            .env("QMD_CONFIG_DIR", &self.config_dir);
        command_output(&mut command)
    }
}

impl Drop for QualityScenario {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn unique_temp_dir(prefix: &str) -> Result<PathBuf> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nanos}", std::process::id()));
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn ensure_command_success(label: &str, output: &std::process::Output) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }
    Err(anyhow!(
        "{} failed: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    ))
}

fn benchmark_mcp(config: &BenchConfig) -> Result<SampleSummary> {
    let qqd_port = free_port()?;
    let current = std::env::current_exe().context("failed to resolve current qqd executable")?;
    let mut qqd = Command::new(current)
        .args(["mcp", "--http", "--port", &qqd_port.to_string()])
        .spawn()
        .context("failed to start qqd MCP server")?;
    wait_for_health(qqd_port)?;
    let _ = http_query_roundtrip(qqd_port, &config.query);
    let qqd_samples = (0..config.iterations)
        .map(|_| http_query_roundtrip(qqd_port, &config.query))
        .collect::<Result<Vec<_>>>()?;
    let _ = qqd.kill();

    Ok(summarize("qqd http query".to_string(), &qqd_samples))
}

fn current_qqd_command<S: AsRef<str>>(args: &[S]) -> Result<Command> {
    let exe = std::env::current_exe().context("failed to resolve current qqd executable")?;
    let mut command = Command::new(exe);
    for arg in args {
        command.arg(arg.as_ref());
    }
    Ok(command)
}

fn free_port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

fn wait_for_health(port: u16) -> Result<()> {
    for _ in 0..30 {
        if let Ok(response) = http_get(port, "/health") {
            if response.contains("\"status\":\"ok\"") || response.contains("\"status\": \"ok\"") {
                return Ok(());
            }
        }
        sleep(Duration::from_millis(200));
    }
    Err(anyhow!("timed out waiting for MCP health on port {}", port))
}

fn http_query_roundtrip(port: u16, query: &str) -> Result<u128> {
    let payload = serde_json::json!({
        "searches":[{"type":"lex","query":query}]
    });
    let start = Instant::now();
    let response = http_post_json(port, "/query", &payload)?;
    if !response.contains("\"results\"") {
        return Err(anyhow!("HTTP query did not return results"));
    }
    Ok(start.elapsed().as_millis())
}

fn http_get(port: u16, path: &str) -> Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn http_post_json(port: u16, path: &str, body: &serde_json::Value) -> Result<String> {
    let payload = serde_json::to_string(body)?;
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}
