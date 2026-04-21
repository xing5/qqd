use rusqlite::Connection;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::Duration;
use tempfile::TempDir;

fn create_fixture_db(path: &Path) {
    let db = Connection::open(path).expect("open fixture db");
    db.execute_batch(
        r#"
        PRAGMA journal_mode = WAL;
        CREATE TABLE content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          collection TEXT NOT NULL,
          path TEXT NOT NULL,
          title TEXT NOT NULL,
          hash TEXT NOT NULL,
          created_at TEXT NOT NULL,
          modified_at TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE store_collections (
          name TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          pattern TEXT NOT NULL DEFAULT '**/*.md',
          ignore_patterns TEXT,
          include_by_default INTEGER DEFAULT 1,
          update_command TEXT,
          context TEXT
        );
        CREATE TABLE content_vectors (
          hash TEXT NOT NULL,
          seq INTEGER NOT NULL DEFAULT 0,
          pos INTEGER NOT NULL DEFAULT 0,
          model TEXT NOT NULL,
          embedded_at TEXT NOT NULL,
          PRIMARY KEY (hash, seq)
        );
        CREATE TABLE llm_cache (
          hash TEXT PRIMARY KEY,
          result TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(filepath, title, body, tokenize='porter unicode61');
        CREATE TRIGGER documents_ai AFTER INSERT ON documents
        WHEN new.active = 1
        BEGIN
          INSERT INTO documents_fts(rowid, filepath, title, body)
          SELECT
            new.id,
            new.collection || '/' || new.path,
            new.title,
            (SELECT doc FROM content WHERE hash = new.hash);
        END;
        "#,
    )
    .expect("schema");

    db.execute(
        "INSERT INTO store_collections (name, path, pattern) VALUES (?1, ?2, ?3)",
        ("docs", "/tmp/docs", "**/*.md"),
    )
    .expect("collection");

    db.execute(
        "INSERT INTO content (hash, doc, created_at) VALUES (?1, ?2, datetime('now'))",
        ("abcdef123456", "# Alpha\n\nRust benchmark document.\n"),
    )
    .expect("content 1");
    db.execute(
        "INSERT INTO content (hash, doc, created_at) VALUES (?1, ?2, datetime('now'))",
        (
            "fedcba654321",
            "# Beta\n\nUse qmd search and set QMD_EDITOR_URI.\n",
        ),
    )
    .expect("content 2");
    db.execute(
        "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES (?1, ?2, ?3, ?4, datetime('now'), datetime('now'), 1)",
        ("docs", "alpha.md", "Alpha", "abcdef123456"),
    )
    .expect("doc 1");
    db.execute(
        "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES (?1, ?2, ?3, ?4, datetime('now'), datetime('now'), 1)",
        ("docs", "beta.md", "Beta", "fedcba654321"),
    )
    .expect("doc 2");
}

fn fixture_env() -> (TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("index.sqlite");
    create_fixture_db(&db_path);
    (temp, db_path)
}

fn empty_fixture_env() -> (TempDir, std::path::PathBuf) {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("index.sqlite");
    let db = Connection::open(&db_path).expect("open fixture db");
    db.execute_batch(
        r#"
        CREATE TABLE store_collections (
          name TEXT PRIMARY KEY,
          path TEXT NOT NULL,
          pattern TEXT NOT NULL DEFAULT '**/*.md',
          ignore_patterns TEXT,
          include_by_default INTEGER DEFAULT 1,
          update_command TEXT,
          context TEXT
        );
        CREATE TABLE content (
          hash TEXT PRIMARY KEY,
          doc TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE content_vectors (
          hash TEXT NOT NULL,
          seq INTEGER NOT NULL DEFAULT 0,
          pos INTEGER NOT NULL DEFAULT 0,
          model TEXT NOT NULL,
          embedded_at TEXT NOT NULL,
          PRIMARY KEY (hash, seq)
        );
        CREATE TABLE llm_cache (
          hash TEXT PRIMARY KEY,
          result TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE documents (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          collection TEXT NOT NULL,
          path TEXT NOT NULL,
          title TEXT NOT NULL,
          hash TEXT NOT NULL,
          created_at TEXT NOT NULL,
          modified_at TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1
        );
        CREATE VIRTUAL TABLE documents_fts USING fts5(filepath, title, body, tokenize='porter unicode61');
        "#,
    )
    .expect("empty schema");
    (temp, db_path)
}

fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind free port");
    listener.local_addr().expect("local addr").port()
}

fn http_get(port: u16, path: &str) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn http_post_json(port: u16, path: &str, body: &serde_json::Value) -> std::io::Result<String> {
    let payload = serde_json::to_string(body).expect("json payload");
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

fn http_post_json_with_session(
    port: u16,
    path: &str,
    body: &serde_json::Value,
    session_id: &str,
) -> std::io::Result<String> {
    let payload = serde_json::to_string(body).expect("json payload");
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    write!(
        stream,
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nmcp-session-id: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        session_id,
        payload.len(),
        payload
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

fn parse_json_array_from_mixed_output(bytes: &[u8]) -> serde_json::Value {
    let text = String::from_utf8(bytes.to_vec()).expect("utf8");
    let start = text.rfind('[').expect("json array start");
    serde_json::from_str(&text[start..]).expect("json array parse")
}

fn extract_header(response: &str, header_name: &str) -> Option<String> {
    response.lines().find_map(|line| {
        line.strip_prefix(&format!("{header_name}: "))
            .map(|value| value.trim().to_string())
    })
}

fn assert_contains_all(output: &str, expected: &[&str]) {
    for needle in expected {
        assert!(output.contains(needle), "missing {needle:?} in:\n{output}");
    }
}

fn stdio_rpc_roundtrip(
    child: &mut std::process::Child,
    body: &serde_json::Value,
) -> serde_json::Value {
    let payload = serde_json::to_string(body).expect("rpc payload");
    {
        let stdin = child.stdin.as_mut().expect("child stdin");
        write!(
            stdin,
            "Content-Length: {}\r\n\r\n{}",
            payload.len(),
            payload
        )
        .expect("write rpc");
        stdin.flush().expect("flush rpc");
    }

    let stdout = child.stdout.as_mut().expect("child stdout");
    let mut header_bytes = Vec::new();
    let mut buf = [0u8; 1];
    while !header_bytes.ends_with(b"\r\n\r\n") {
        stdout.read_exact(&mut buf).expect("read header");
        header_bytes.push(buf[0]);
    }
    let header = String::from_utf8(header_bytes).expect("header utf8");
    let length = header
        .lines()
        .find_map(|line| {
            line.strip_prefix("Content-Length: ")
                .and_then(|v| v.trim().parse::<usize>().ok())
        })
        .expect("content length");
    let mut body_bytes = vec![0u8; length];
    stdout.read_exact(&mut body_bytes).expect("read body");
    serde_json::from_slice(&body_bytes).expect("json rpc response")
}

fn qqd_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_qqd"));
    command.env("QQD_DISABLE_MODEL_AUTODISCOVERY", "1");
    command
}

#[test]
fn search_json_reads_qmd_compatible_sqlite() {
    let (_temp, db_path) = fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["search", "--json", "rust"])
        .output()
        .expect("run qqd");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("\"docid\": \"#abcdef\""));
    assert!(stdout.contains("\"file\": \"qmd://docs/alpha.md\""));
    assert!(stdout.contains("\"title\": \"Alpha\""));
}

#[test]
fn bench_latency_outputs_comparison_json() {
    let (_temp, db_path) = fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["bench-latency", "rust", "--iterations", "1"])
        .output()
        .expect("run benchmark");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("\"qqd\""));
    assert!(stdout.contains("\"qmd\""));
    assert!(stdout.contains("\"result_count\": 1"));
    assert!(stdout.contains("\"rust_wins\""));
}

#[test]
fn bench_quality_outputs_passing_report() {
    let output = qqd_command()
        .args(["bench-quality", "benchmarks/query-quality-fixture.json"])
        .output()
        .expect("run quality benchmark");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("\"meets_quality_gate\": true"));
    assert!(stdout.contains("\"hybrid_performance_notes\""));
    assert!(stdout.contains("\"hybrid_search_quality\""));
}

#[test]
fn bench_metrics_outputs_ir_scores() {
    let output = qqd_command()
        .args(["bench-metrics", "benchmarks/query-quality-fixture.json"])
        .output()
        .expect("run metrics benchmark");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("\"mean_accuracy\""));
    assert!(stdout.contains("\"mean_recall_at_k\""));
    assert!(stdout.contains("\"mean_f1_at_k\""));
    assert!(stdout.contains("\"mean_ndcg_at_k\""));
}

#[test]
fn bench_quality_fails_when_contract_is_not_met() {
    let fixture_dir = tempfile::tempdir().expect("fixture dir");
    let fixture_path = fixture_dir.path().join("bad-quality-fixture.json");
    std::fs::write(
        &fixture_path,
        serde_json::to_string_pretty(&serde_json::json!({
            "name": "bad-quality-contract",
            "collection": {
                "name": "docs",
                "context": "Performance and retrieval docs"
            },
            "documents": [
                {
                    "path": "alpha.md",
                    "body": "# Alpha\n\nRust benchmark document.\n"
                },
                {
                    "path": "beta.md",
                    "body": "# Beta\n\nLatency tuning notes and benchmark results.\n"
                }
            ],
            "cases": [
                {
                    "name": "impossible-top-hit",
                    "query": "lex: benchmark",
                    "expected_top": "docs/missing.md",
                    "must_include": ["docs/missing.md"],
                    "top_k": 1
                }
            ],
            "minimum_pass_rate": 1.0
        }))
        .expect("serialize bad fixture"),
    )
    .expect("write bad fixture");

    let output = qqd_command()
        .args([
            "bench-quality",
            fixture_path.to_str().expect("fixture path str"),
        ])
        .output()
        .expect("run bad quality benchmark");

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stdout.contains("\"meets_quality_gate\": false"));
    assert!(stderr.contains("quality gate failed"));
}

#[test]
fn status_command_runs_without_oracle() {
    let (_temp, db_path) = fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["status"])
        .output()
        .expect("run forwarded status");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("QQD Status"));
    assert!(stdout.contains("Total:    2 files indexed"));
}

#[test]
fn status_reports_expected_core_sections() {
    let (_temp, db_path) = fixture_env();
    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("status")
        .output()
        .expect("qqd status");

    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    let qqd = String::from_utf8(qqd.stdout).expect("qqd status utf8");
    assert_contains_all(
        &qqd,
        &[
            "QQD Status",
            "Total:    2 files indexed",
            "Vectors:  0 embedded",
            "Pending:  2 need embedding",
            "Embedding:   https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF",
        ],
    );
}

#[test]
fn vsearch_reports_empty_results_when_embeddings_are_missing() {
    let (_temp, db_path) = fixture_env();
    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["vsearch", "rust", "--json"])
        .output()
        .expect("qqd vsearch");

    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    assert_eq!(
        String::from_utf8(qqd.stdout).expect("qqd vsearch utf8"),
        "[]\n"
    );
    assert!(
        String::from_utf8_lossy(&qqd.stderr).contains("100%) need embeddings"),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
}

#[test]
fn query_lex_json_returns_expected_result_files() {
    let (_temp, db_path) = fixture_env();
    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["query", "--json", "lex: rust"])
        .output()
        .expect("qqd query");

    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    let qqd = parse_json_array_from_mixed_output(&qqd.stdout);
    let project = |value: &serde_json::Value| {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|entry| {
                (
                    entry
                        .get("file")
                        .and_then(|v| v.as_str())
                        .unwrap()
                        .to_string(),
                    entry
                        .get("title")
                        .and_then(|v| v.as_str())
                        .unwrap()
                        .to_string(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        project(&qqd),
        vec![("qmd://docs/alpha.md".to_string(), "Alpha".to_string())]
    );
}

#[test]
fn forwarded_collection_add_preserves_callers_working_directory() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("index.sqlite");
    let config_dir = temp.path().join("config");
    let docs_dir = temp.path().join("docs");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::create_dir_all(&docs_dir).expect("docs dir");
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").expect("config");
    std::fs::write(
        docs_dir.join("note.md"),
        "# Note\n\nForwarding should use cwd.\n",
    )
    .expect("doc");

    let add_output = qqd_command()
        .current_dir(&docs_dir)
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .arg("collection")
        .arg("add")
        .arg(".")
        .arg("--name")
        .arg("docs")
        .output()
        .expect("collection add");

    assert!(
        add_output.status.success(),
        "{}",
        String::from_utf8_lossy(&add_output.stderr)
    );

    let status_output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .arg("status")
        .output()
        .expect("status");

    assert!(
        status_output.status.success(),
        "{}",
        String::from_utf8_lossy(&status_output.stderr)
    );
    let stdout = String::from_utf8(status_output.stdout).expect("utf8");
    assert!(stdout.contains("docs (qmd://docs/)"));
    assert!(stdout.contains("Files:    1"));
}

#[test]
fn forwarded_get_preserves_document_text_without_rebranding() {
    let (_temp, db_path) = fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["get", "qmd://docs/beta.md"])
        .output()
        .expect("qqd get");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("Use qmd search and set QMD_EDITOR_URI."));
    assert!(!stdout.contains("Use qqd search and set QQD_EDITOR_URI."));
}

#[test]
fn get_returns_expected_virtual_path_document() {
    let (_temp, db_path) = fixture_env();
    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["get", "qmd://docs/beta.md"])
        .output()
        .expect("qqd get");

    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    assert_eq!(
        String::from_utf8(qqd.stdout).expect("qqd utf8"),
        "# Beta\n\nUse qmd search and set QMD_EDITOR_URI.\n\n"
    );
}

#[test]
fn multi_get_returns_expected_cli_and_json() {
    let (_temp, db_path) = fixture_env();
    let qqd_cli = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["multi-get", "docs/*.md"])
        .output()
        .expect("qqd multi-get cli");
    assert!(
        qqd_cli.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd_cli.stderr)
    );
    assert_eq!(
        String::from_utf8(qqd_cli.stdout).expect("qqd cli utf8"),
        "\n============================================================\nFile: alpha.md\n============================================================\n\n# Alpha\n\nRust benchmark document.\n\n\n============================================================\nFile: beta.md\n============================================================\n\n# Beta\n\nUse qmd search and set QMD_EDITOR_URI.\n\n"
    );

    let qqd_json = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["multi-get", "docs/*.md", "--json"])
        .output()
        .expect("qqd multi-get json");
    assert!(
        qqd_json.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd_json.stderr)
    );
    let qqd_json: serde_json::Value =
        serde_json::from_slice(&qqd_json.stdout).expect("qqd json parse");
    assert_eq!(
        qqd_json,
        serde_json::json!([
            {
                "body": "# Alpha\n\nRust benchmark document.\n",
                "file": "alpha.md",
                "title": "Alpha"
            },
            {
                "body": "# Beta\n\nUse qmd search and set QMD_EDITOR_URI.\n",
                "file": "beta.md",
                "title": "Beta"
            }
        ])
    );
}

#[test]
fn ls_reports_expected_collection_view() {
    let (_temp, db_path) = fixture_env();
    let config_dir = tempfile::tempdir().expect("config dir");
    std::fs::write(
        config_dir.path().join("index.yml"),
        "collections:\n  docs:\n    path: /tmp/docs\n    pattern: \"**/*.md\"\n",
    )
    .expect("ls config");
    let qqd_docs = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", config_dir.path())
        .args(["ls", "docs"])
        .output()
        .expect("qqd ls docs");
    assert!(
        qqd_docs.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd_docs.stderr)
    );
    let qqd_docs = String::from_utf8(qqd_docs.stdout).expect("qqd ls docs utf8");
    assert_contains_all(
        &qqd_docs,
        &["34 B", "47 B", "qmd://docs/alpha.md", "qmd://docs/beta.md"],
    );
}

#[test]
fn collection_list_reports_expected_basic_output() {
    let temp = tempfile::tempdir().expect("config dir");
    let db_path = temp.path().join("index.sqlite");
    let config_dir = temp.path().join("config");
    let docs_dir = temp.path().join("docs");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::create_dir_all(&docs_dir).expect("docs dir");
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").expect("config");
    std::fs::write(
        docs_dir.join("a.md"),
        "# Alpha\n\nRust search benchmark document.\n",
    )
    .expect("doc");
    std::fs::write(
        docs_dir.join("b.md"),
        "# Beta\n\nUse qmd search and set QMD_EDITOR_URI.\n",
    )
    .expect("doc");

    let seed = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args([
            "collection",
            "add",
            docs_dir.to_str().expect("docs dir str"),
            "--name",
            "docs",
        ])
        .output()
        .expect("seed collection");
    assert!(
        seed.status.success(),
        "{}",
        String::from_utf8_lossy(&seed.stderr)
    );

    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args(["collection", "list"])
        .output()
        .expect("qqd collection list");
    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    let qqd = String::from_utf8(qqd.stdout).expect("qqd collection list utf8");
    assert_contains_all(
        &qqd,
        &[
            "Collections (1):",
            "docs (qmd://docs/)",
            "Pattern:  **/*.md",
            "Files:    2",
        ],
    );
}

#[test]
fn collection_show_reports_expected_basic_output() {
    let temp = tempfile::tempdir().expect("config dir");
    let db_path = temp.path().join("index.sqlite");
    let config_dir = temp.path().join("config");
    let docs_dir = temp.path().join("docs");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::create_dir_all(&docs_dir).expect("docs dir");
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").expect("config");
    std::fs::write(
        docs_dir.join("a.md"),
        "# Alpha\n\nRust search benchmark document.\n",
    )
    .expect("doc");

    let seed = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args([
            "collection",
            "add",
            docs_dir.to_str().expect("docs dir str"),
            "--name",
            "docs",
        ])
        .output()
        .expect("seed collection");
    assert!(
        seed.status.success(),
        "{}",
        String::from_utf8_lossy(&seed.stderr)
    );

    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args(["collection", "show", "docs"])
        .output()
        .expect("qqd collection show");
    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    let qqd = String::from_utf8(qqd.stdout).expect("qqd show utf8");
    assert_contains_all(
        &qqd,
        &[
            "Collection: docs",
            &format!("Path:     {}", docs_dir.display()),
            "Pattern:  **/*.md",
            "Include:  yes (default)",
        ],
    );
}

#[test]
fn native_collection_rename_and_remove_update_db_state() {
    let (_temp, db_path) = fixture_env();

    let rename = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["collection", "rename", "docs", "archive"])
        .output()
        .expect("rename collection");
    assert!(
        rename.status.success(),
        "{}",
        String::from_utf8_lossy(&rename.stderr)
    );

    let show = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["collection", "show", "archive"])
        .output()
        .expect("show renamed collection");
    assert!(
        show.status.success(),
        "{}",
        String::from_utf8_lossy(&show.stderr)
    );
    let show = String::from_utf8(show.stdout).expect("show utf8");
    assert!(show.contains("Collection: archive"));

    let remove = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["collection", "remove", "archive"])
        .output()
        .expect("remove collection");
    assert!(
        remove.status.success(),
        "{}",
        String::from_utf8_lossy(&remove.stderr)
    );

    let status = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("status")
        .output()
        .expect("status after remove");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let status = String::from_utf8(status.stdout).expect("status utf8");
    assert!(status.contains("Total:    0 files indexed"));
}

#[test]
fn native_context_add_list_remove_roundtrip() {
    let (_temp, db_path) = fixture_env();

    let add = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["context", "add", "qmd://docs/", "Collection root context"])
        .output()
        .expect("context add");
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let list = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["context", "list"])
        .output()
        .expect("context list");
    assert!(
        list.status.success(),
        "{}",
        String::from_utf8_lossy(&list.stderr)
    );
    let list = String::from_utf8(list.stdout).expect("list utf8");
    assert!(list.contains("Configured Contexts"));
    assert!(list.contains("docs"));
    assert!(list.contains("Collection root context"));

    let remove = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["context", "rm", "qmd://docs/"])
        .output()
        .expect("context remove");
    assert!(
        remove.status.success(),
        "{}",
        String::from_utf8_lossy(&remove.stderr)
    );

    let list_after = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["context", "list"])
        .output()
        .expect("context list after remove");
    assert!(
        list_after.status.success(),
        "{}",
        String::from_utf8_lossy(&list_after.stderr)
    );
    let list_after = String::from_utf8(list_after.stdout).expect("list after utf8");
    assert!(list_after.contains("No contexts configured"));
}

#[test]
fn native_update_reindexes_changed_files() {
    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("index.sqlite");
    let config_dir = temp.path().join("config");
    let docs_dir = temp.path().join("docs");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::create_dir_all(&docs_dir).expect("docs dir");
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").expect("config");
    std::fs::write(docs_dir.join("a.md"), "# Alpha\n\nOriginal body.\n").expect("doc");

    let add = qqd_command()
        .current_dir(&docs_dir)
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args(["collection", "add", ".", "--name", "docs"])
        .output()
        .expect("collection add");
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );

    std::fs::write(docs_dir.join("a.md"), "# Alpha\n\nUpdated body.\n").expect("rewrite doc");
    let update = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .arg("update")
        .output()
        .expect("update");
    assert!(
        update.status.success(),
        "{}",
        String::from_utf8_lossy(&update.stderr)
    );
    let update_stdout = String::from_utf8(update.stdout).expect("update utf8");
    assert!(update_stdout.contains("Updating 1 collection(s)..."));
    assert!(update_stdout.contains("updated"));

    let get = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("get")
        .arg("qmd://docs/a.md")
        .output()
        .expect("get after update");
    assert!(
        get.status.success(),
        "{}",
        String::from_utf8_lossy(&get.stderr)
    );
    let get_stdout = String::from_utf8(get.stdout).expect("get utf8");
    assert!(get_stdout.contains("Updated body."));
}

#[test]
fn native_cleanup_removes_inactive_documents() {
    let (_temp, db_path) = fixture_env();
    let db = Connection::open(&db_path).expect("open db");
    db.execute("UPDATE documents SET active = 0 WHERE path = 'beta.md'", [])
        .expect("mark inactive");
    db.execute(
        "INSERT INTO llm_cache (hash, result, created_at) VALUES ('cache', '{}', datetime('now'))",
        [],
    )
    .expect("insert cache");
    drop(db);

    let cleanup = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("cleanup")
        .output()
        .expect("cleanup");
    assert!(
        cleanup.status.success(),
        "{}",
        String::from_utf8_lossy(&cleanup.stderr)
    );
    let cleanup_stdout = String::from_utf8(cleanup.stdout).expect("cleanup utf8");
    assert!(cleanup_stdout.contains("Cleared 1 cached API responses"));
    assert!(cleanup_stdout.contains("Removed 1 inactive document records"));

    let status = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("status")
        .output()
        .expect("status after cleanup");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let status = String::from_utf8(status.stdout).expect("status utf8");
    assert!(status.contains("Total:    1 files indexed"));
}

#[test]
fn native_embed_enables_vsearch_and_plain_query() {
    let (_temp, db_path) = fixture_env();

    let embed = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QQD_SKIP_QMD_EMBED", "1")
        .arg("embed")
        .output()
        .expect("embed");
    assert!(
        embed.status.success(),
        "{}",
        String::from_utf8_lossy(&embed.stderr)
    );
    let embed_stdout = String::from_utf8(embed.stdout).expect("embed utf8");
    assert!(embed_stdout.contains("Embedded"));

    let status = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("status")
        .output()
        .expect("status after embed");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let status = String::from_utf8(status.stdout).expect("status utf8");
    assert!(!status.contains("Pending:"));

    let vsearch = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["vsearch", "rust", "--json"])
        .output()
        .expect("vsearch after embed");
    assert!(
        vsearch.status.success(),
        "{}",
        String::from_utf8_lossy(&vsearch.stderr)
    );
    let vsearch: serde_json::Value = serde_json::from_slice(&vsearch.stdout).expect("vsearch json");
    assert!(!vsearch.as_array().unwrap().is_empty());

    let query = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["query", "--json", "rust"])
        .output()
        .expect("query after embed");
    assert!(
        query.status.success(),
        "{}",
        String::from_utf8_lossy(&query.stderr)
    );
    let query = parse_json_array_from_mixed_output(&query.stdout);
    assert!(!query.as_array().unwrap().is_empty());
}

#[test]
fn native_embed_does_not_require_qmd_oracle() {
    let (_temp, db_path) = fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QQD_QMD_ROOT", "/definitely/missing-qmd")
        .arg("embed")
        .output()
        .expect("native embed");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let status = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("status")
        .output()
        .expect("status after native embed");
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let status = String::from_utf8(status.stdout).expect("status utf8");
    assert!(status.contains("Vectors:  2 embedded"));
    assert!(!status.contains("Pending:"));
}

#[test]
fn embed_fails_with_missing_gguf_model_path() {
    let (_temp, db_path) = fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QQD_EMBED_MODEL", "/definitely/missing.gguf")
        .arg("embed")
        .output()
        .expect("embed with missing model");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("failed to load QQD embed model") || stderr.contains("missing.gguf"));
}

#[test]
fn live_gguf_models_enable_embed_and_query_when_env_set() {
    let embed_model = match std::env::var("QQD_TEST_EMBED_MODEL") {
        Ok(value) => value,
        Err(_) => return,
    };
    let rerank_model = match std::env::var("QQD_TEST_RERANK_MODEL") {
        Ok(value) => value,
        Err(_) => return,
    };

    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("index.sqlite");
    let config_dir = temp.path().join("config");
    let docs_dir = temp.path().join("docs");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::create_dir_all(&docs_dir).expect("docs dir");
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").expect("config");
    std::fs::write(
        docs_dir.join("alpha.md"),
        "# Alpha\n\nRust benchmark document.\n",
    )
    .expect("alpha doc");
    std::fs::write(
        docs_dir.join("beta.md"),
        "# Beta\n\nLatency tuning notes and benchmark results.\n",
    )
    .expect("beta doc");
    std::fs::write(
        docs_dir.join("gamma.md"),
        "# Gamma\n\nSemantic retrieval and hybrid search guidance.\n",
    )
    .expect("gamma doc");

    let add = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .env("QQD_EMBED_MODEL", &embed_model)
        .env("QQD_RERANK_MODEL", &rerank_model)
        .env("QQD_MODEL_THREADS", "1")
        .args([
            "collection",
            "add",
            docs_dir.to_str().expect("docs dir str"),
            "--name",
            "docs",
        ])
        .output()
        .expect("collection add");
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let embed = Command::new("timeout")
        .arg("90")
        .arg(env!("CARGO_BIN_EXE_qqd"))
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .env("QQD_EMBED_MODEL", &embed_model)
        .env("QQD_RERANK_MODEL", &rerank_model)
        .env("QQD_MODEL_THREADS", "1")
        .args(["embed", "--force"])
        .output()
        .expect("embed with live model");
    assert_ne!(
        embed.status.code(),
        Some(124),
        "live GGUF embed timed out: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&embed.stdout),
        String::from_utf8_lossy(&embed.stderr)
    );
    assert!(
        embed.status.success(),
        "{}",
        String::from_utf8_lossy(&embed.stderr)
    );

    let db = Connection::open(&db_path).expect("open db");
    let model_name: String = db
        .query_row(
            "SELECT model FROM content_vectors ORDER BY hash LIMIT 1",
            [],
            |row| row.get(0),
        )
        .expect("model row");
    let backend: String = db
        .query_row(
            "SELECT value FROM store_config WHERE key = 'qqd_embed_backend'",
            [],
            |row| row.get(0),
        )
        .expect("backend row");
    assert_eq!(model_name, "qqd-gguf");
    assert!(backend.starts_with("gguf:"));

    let query = Command::new("timeout")
        .arg("90")
        .arg(env!("CARGO_BIN_EXE_qqd"))
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .env("QQD_EMBED_MODEL", &embed_model)
        .env("QQD_RERANK_MODEL", &rerank_model)
        .env("QQD_MODEL_THREADS", "1")
        .args([
            "query",
            "--json",
            "intent: performance tuning\nlex: latency\nvec: benchmark results",
        ])
        .output()
        .expect("query with live model");
    assert_ne!(
        query.status.code(),
        Some(124),
        "live GGUF query timed out: stdout=\n{}\nstderr=\n{}",
        String::from_utf8_lossy(&query.stdout),
        String::from_utf8_lossy(&query.stderr)
    );
    assert!(
        query.status.success(),
        "{}",
        String::from_utf8_lossy(&query.stderr)
    );
    let results = parse_json_array_from_mixed_output(&query.stdout);
    let top = results
        .as_array()
        .and_then(|items| items.first())
        .cloned()
        .expect("top result");
    assert_eq!(top["file"], "qmd://docs/beta.md");
}

#[test]
fn live_gguf_backend_mismatch_requires_reembed() {
    let embed_model = match std::env::var("QQD_TEST_EMBED_MODEL") {
        Ok(value) => value,
        Err(_) => return,
    };

    let temp = tempfile::tempdir().expect("tempdir");
    let db_path = temp.path().join("index.sqlite");
    let config_dir = temp.path().join("config");
    let docs_dir = temp.path().join("docs");
    std::fs::create_dir_all(&config_dir).expect("config dir");
    std::fs::create_dir_all(&docs_dir).expect("docs dir");
    std::fs::write(config_dir.join("index.yml"), "collections: {}\n").expect("config");
    std::fs::write(
        docs_dir.join("alpha.md"),
        "# Alpha\n\nRust benchmark document.\n",
    )
    .expect("alpha doc");

    let add = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args([
            "collection",
            "add",
            docs_dir.to_str().expect("docs dir str"),
            "--name",
            "docs",
        ])
        .output()
        .expect("collection add");
    assert!(
        add.status.success(),
        "{}",
        String::from_utf8_lossy(&add.stderr)
    );

    let embed = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .args(["embed", "--force"])
        .output()
        .expect("deterministic embed");
    assert!(
        embed.status.success(),
        "{}",
        String::from_utf8_lossy(&embed.stderr)
    );

    let mismatch = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QMD_CONFIG_DIR", &config_dir)
        .env("QQD_EMBED_MODEL", &embed_model)
        .env("QQD_MODEL_THREADS", "1")
        .args(["vsearch", "rust", "--json"])
        .output()
        .expect("mismatch vsearch");
    assert!(!mismatch.status.success());
    let stderr = String::from_utf8(mismatch.stderr).expect("utf8");
    assert!(stderr.contains("qqd vectors were built with backend"));
    assert!(stderr.contains("qqd embed --force"));
}

#[test]
fn qmd_vector_state_bootstraps_qqd_vectors() {
    let (_temp, db_path) = fixture_env();
    let db = Connection::open(&db_path).expect("open db");
    db.execute(
        "INSERT INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES ('abcdef123456', 0, 0, 'embeddinggemma', datetime('now'))",
        [],
    )
    .expect("insert qmd vector marker");
    drop(db);

    let vsearch = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["vsearch", "rust", "--json"])
        .output()
        .expect("vsearch with qmd vector state");
    assert!(
        vsearch.status.success(),
        "{}",
        String::from_utf8_lossy(&vsearch.stderr)
    );
    let vsearch: serde_json::Value = serde_json::from_slice(&vsearch.stdout).expect("vsearch json");
    assert!(!vsearch.as_array().unwrap().is_empty());

    let db = Connection::open(&db_path).expect("reopen db");
    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM qqd_vectors", [], |row| row.get(0))
        .expect("count qqd vectors");
    assert!(count > 0);
}

#[test]
fn query_lex_json_matches_expected_output_when_qqd_vectors_exist() {
    let (_temp, db_path) = fixture_env();
    let db = Connection::open(&db_path).expect("open db");
    db.execute(
        "CREATE TABLE IF NOT EXISTS qqd_vectors (hash TEXT PRIMARY KEY, vector TEXT NOT NULL, updated_at TEXT NOT NULL)",
        [],
    )
    .expect("create qqd_vectors");
    db.execute(
        "INSERT INTO qqd_vectors (hash, vector, updated_at) VALUES ('abcdef123456', '[1.0,0.0]', datetime('now'))",
        [],
    )
    .expect("seed qqd_vectors");
    drop(db);

    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["query", "--json", "lex: alpha"])
        .output()
        .expect("qqd query with qqd_vectors");

    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    let qqd = parse_json_array_from_mixed_output(&qqd.stdout);
    assert_eq!(
        qqd,
        serde_json::json!([{
            "docid": "#abcdef",
            "score": 0.89,
            "file": "qmd://docs/alpha.md",
            "line": 1,
            "title": "Alpha",
            "snippet": "@@ -1,3 @@ (0 before, 1 after)\n# Alpha\n\nRust benchmark document."
        }])
    );
}

#[test]
fn query_lex_json_with_vectors_does_not_require_qmd_oracle() {
    let (_temp, db_path) = fixture_env();
    let db = Connection::open(&db_path).expect("open db");
    db.execute(
        "CREATE TABLE IF NOT EXISTS qqd_vectors (hash TEXT PRIMARY KEY, vector TEXT NOT NULL, updated_at TEXT NOT NULL)",
        [],
    )
    .expect("create qqd_vectors");
    db.execute(
        "INSERT INTO qqd_vectors (hash, vector, updated_at) VALUES ('abcdef123456', '[1.0,0.0]', datetime('now'))",
        [],
    )
    .expect("seed qqd_vectors");
    drop(db);

    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QQD_QMD_ROOT", "/definitely/missing-qmd")
        .args(["query", "--json", "lex: alpha"])
        .output()
        .expect("qqd query without qmd oracle");

    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed = parse_json_array_from_mixed_output(&output.stdout);
    let entry = &parsed.as_array().expect("array")[0];
    assert_eq!(entry["file"], "qmd://docs/alpha.md");
    assert_eq!(entry["score"], 0.89);
}

#[test]
fn help_is_rebranded_to_qqd() {
    let output = qqd_command().arg("--help").output().expect("help");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("qqd — Quick Markdown Search"));
    assert!(stdout.contains("qqd <command> [options]"));
}

#[test]
fn help_is_available_without_qmd_oracle() {
    let output = qqd_command()
        .env("QQD_QMD_ROOT", "/definitely/missing-qmd")
        .arg("--help")
        .output()
        .expect("help without oracle");

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("qqd — Quick Markdown Search"));
}

#[test]
fn subcommand_help_is_rebranded_to_qqd() {
    let output = qqd_command()
        .args(["search", "--help"])
        .output()
        .expect("subcommand help");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8");
    assert!(stdout.contains("qqd — Quick Markdown Search"));
    assert!(stdout.contains("qqd <command> [options]"));
}

#[test]
fn bench_latency_rejects_empty_indexes() {
    let (_temp, db_path) = empty_fixture_env();
    let output = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["bench-latency", "rust", "--iterations", "1"])
        .output()
        .expect("empty benchmark");

    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).expect("utf8");
    assert!(stderr.contains("populated index"));
}

#[test]
fn search_json_matches_expected_fixture_shape() {
    let (_temp, db_path) = fixture_env();
    let qqd = qqd_command()
        .env("INDEX_PATH", &db_path)
        .args(["search", "--json", "rust"])
        .output()
        .expect("qqd search");

    assert!(
        qqd.status.success(),
        "{}",
        String::from_utf8_lossy(&qqd.stderr)
    );
    assert_eq!(
        String::from_utf8(qqd.stdout).expect("qqd utf8"),
        "[\n  {\n    \"docid\": \"#abcdef\",\n    \"score\": 0,\n    \"file\": \"qmd://docs/alpha.md\",\n    \"line\": 3,\n    \"title\": \"Alpha\",\n    \"snippet\": \"@@ -2,3 @@ (1 before, 0 after)\\n\\nRust benchmark document.\\n\"\n  }\n]\n"
    );
}

#[test]
fn forwarded_mcp_http_healthcheck_works() {
    let (temp, db_path) = fixture_env();
    let xdg_cache_home = temp.path().join("xdg-cache");
    std::fs::create_dir_all(&xdg_cache_home).expect("cache dir");
    let port = free_port();

    let start = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("XDG_CACHE_HOME", &xdg_cache_home)
        .args(["mcp", "--http", "--daemon", "--port", &port.to_string()])
        .output()
        .expect("start mcp daemon");
    assert!(
        start.status.success(),
        "{}",
        String::from_utf8_lossy(&start.stderr)
    );

    let mut health = None;
    for _ in 0..20 {
        if let Ok(response) = http_get(port, "/health") {
            if response.contains("\"status\":\"ok\"") || response.contains("\"status\": \"ok\"") {
                health = Some(response);
                break;
            }
        }
        sleep(Duration::from_millis(200));
    }

    let stop = qqd_command()
        .env("XDG_CACHE_HOME", &xdg_cache_home)
        .args(["mcp", "stop"])
        .output()
        .expect("stop mcp daemon");
    assert!(
        stop.status.success(),
        "{}",
        String::from_utf8_lossy(&stop.stderr)
    );
    let stop_stdout = String::from_utf8(stop.stdout).expect("stop utf8");
    assert!(stop_stdout.contains("Stopped QQD MCP server"));

    let health = health.expect("health response");
    assert!(health.contains("200 OK"));
}

#[test]
fn native_mcp_http_supports_initialize_and_tools() {
    let (_temp, db_path) = fixture_env();
    let config_dir = tempfile::tempdir().expect("mcp config");
    std::fs::write(
        config_dir.path().join("index.yml"),
        "collections:\n  docs:\n    path: /tmp/docs\n    pattern: \"**/*.md\"\n    context:\n      /: Docs context\n  notes:\n    path: /tmp/notes\n    pattern: \"**/*.md\"\n",
    )
    .expect("mcp config write");
    let db = Connection::open(&db_path).expect("open mcp db");
    db.execute(
        "INSERT INTO store_collections (name, path, pattern) VALUES ('notes', '/tmp/notes', '**/*.md')",
        [],
    )
    .expect("insert notes collection");
    db.execute(
        "INSERT INTO content (hash, doc, created_at) VALUES ('abc999000111', '# Note\\n\\nSecondary collection.\\n', datetime('now'))",
        [],
    )
    .expect("insert note content");
    db.execute(
        "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES ('notes', 'beta.md', 'Beta', 'abc999000111', datetime('now'), datetime('now'), 1)",
        [],
    )
    .expect("insert note doc");
    drop(db);

    let xdg_cache_home = tempfile::tempdir().expect("xdg cache");
    let port = free_port();

    let mut child = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("XDG_CACHE_HOME", xdg_cache_home.path())
        .env("QMD_CONFIG_DIR", config_dir.path())
        .args(["mcp", "--http", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start native mcp");

    let initialize = serde_json::json!({
        "jsonrpc":"2.0",
        "id":1,
        "method":"initialize",
        "params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}
    });
    let mut response = None;
    for _ in 0..20 {
        if let Ok(raw) = http_post_json(port, "/mcp", &initialize) {
            if raw.contains("\"serverInfo\"") {
                response = Some(raw);
                break;
            }
        }
        sleep(Duration::from_millis(200));
    }
    let response = response.expect("initialize response");
    assert!(response.contains("\"name\":\"qmd\""));
    assert!(response.contains("QQD is your local search engine"));
    assert!(response.contains("Run `qqd embed`"));
    let session_id = extract_header(&response, "mcp-session-id").expect("session id");
    assert!(response.contains("Docs context"));
    let second_initialize =
        http_post_json(port, "/mcp", &initialize).expect("second initialize response");
    let second_session_id =
        extract_header(&second_initialize, "mcp-session-id").expect("second session id");
    assert_ne!(session_id, second_session_id);

    let tools_list = serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}});
    let tools_first_session = http_post_json_with_session(port, "/mcp", &tools_list, &session_id)
        .expect("tools/list first session");
    assert!(tools_first_session.contains("\"query\""));
    let session_id = second_session_id;
    let missing_session =
        http_post_json(port, "/mcp", &tools_list).expect("tools/list missing session");
    assert!(missing_session.contains("400"));
    assert!(missing_session.contains("\"jsonrpc\":\"2.0\""));

    let tools =
        http_post_json_with_session(port, "/mcp", &tools_list, &session_id).expect("tools/list");
    assert!(tools.contains("\"query\""));
    assert!(tools.contains("\"get\""));
    assert!(tools.contains("\"status\""));
    assert!(tools.contains("\"title\":\"Query\""));

    let query = serde_json::json!({
        "jsonrpc":"2.0","id":3,"method":"tools/call",
        "params":{"name":"query","arguments":{"searches":[{"type":"lex","query":"alpha"}],"collections":["docs"]}}
    });
    let query_response =
        http_post_json_with_session(port, "/mcp", &query, &session_id).expect("query");
    assert!(
        query_response.contains("\"structuredContent\""),
        "{query_response}"
    );
    assert!(query_response.contains("\"file\":\"docs/alpha.md\""));
    assert!(query_response.contains("\"context\":\"Docs context\""));
    assert!(query_response.contains("Found 1 result"));
    assert!(!query_response.contains("notes/beta.md"));

    let get = serde_json::json!({
        "jsonrpc":"2.0","id":4,"method":"tools/call",
        "params":{"name":"get","arguments":{"path":"alpha.md"}}
    });
    let get_response = http_post_json_with_session(port, "/mcp", &get, &session_id).expect("get");
    assert!(get_response.contains("\"mimeType\":\"text/markdown\""));
    assert!(get_response.contains("<!-- Context: Docs context -->"));

    let ranged_get = serde_json::json!({
        "jsonrpc":"2.0","id":5,"method":"tools/call",
        "params":{"name":"get","arguments":{"path":"alpha.md","fromLine":1,"maxLines":1,"lineNumbers":true}}
    });
    let ranged_get_response =
        http_post_json_with_session(port, "/mcp", &ranged_get, &session_id).expect("ranged get");
    assert!(ranged_get_response.contains("1: # Alpha"));

    let suffixed_get = serde_json::json!({
        "jsonrpc":"2.0","id":56,"method":"tools/call",
        "params":{"name":"get","arguments":{"file":"alpha.md:1","lineNumbers":true}}
    });
    let suffixed_get_response =
        http_post_json_with_session(port, "/mcp", &suffixed_get, &session_id).expect("suffix get");
    assert!(suffixed_get_response.contains("1: # Alpha"));

    let missing_get = serde_json::json!({
        "jsonrpc":"2.0","id":55,"method":"tools/call",
        "params":{"name":"get","arguments":{"path":"missing.md"}}
    });
    let missing_get_response =
        http_post_json_with_session(port, "/mcp", &missing_get, &session_id).expect("missing get");
    assert!(missing_get_response.contains("\"isError\":true"));
    assert!(missing_get_response.contains("Document not found"));

    let multi_get = serde_json::json!({
        "jsonrpc":"2.0","id":6,"method":"tools/call",
        "params":{"name":"multi_get","arguments":{"pattern":"docs/*.md","maxLines":1,"lineNumbers":true}}
    });
    let multi_get_response =
        http_post_json_with_session(port, "/mcp", &multi_get, &session_id).expect("multi_get");
    assert!(multi_get_response.contains("\"uri\":\"qmd://docs/alpha.md\""));
    assert!(multi_get_response.contains("1: # Alpha"));

    let status_call = serde_json::json!({
        "jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"status","arguments":{}}
    });
    let status_response =
        http_post_json_with_session(port, "/mcp", &status_call, &session_id).expect("status call");
    assert!(status_response.contains("QQD Index Status:"));

    let resource_read = serde_json::json!({
        "jsonrpc":"2.0","id":9,"method":"resources/read","params":{"uri":"qmd://docs/alpha.md"}
    });
    let resource_response = http_post_json_with_session(port, "/mcp", &resource_read, &session_id)
        .expect("resource read");
    assert!(resource_response.contains("\"mimeType\":\"text/markdown\""));
    assert!(resource_response.contains("# Alpha"));

    let db = Connection::open(&db_path).expect("open db for special path");
    db.execute(
        "INSERT INTO content (hash, doc, created_at) VALUES ('special123456', '# Special\\n\\nEncoded path body.\\n', datetime('now'))",
        [],
    )
    .expect("insert special content");
    db.execute(
        "INSERT INTO documents (collection, path, title, hash, created_at, modified_at, active) VALUES ('docs', 'My Note #1?.md', 'Special', 'special123456', datetime('now'), datetime('now'), 1)",
        [],
    )
    .expect("insert special document");
    drop(db);

    let encoded_resource = serde_json::json!({
        "jsonrpc":"2.0","id":10,"method":"resources/read","params":{"uri":"qmd://docs/My%20Note%20%231%3F.md"}
    });
    let encoded_response =
        http_post_json_with_session(port, "/mcp", &encoded_resource, &session_id)
            .expect("encoded resource read");
    assert!(encoded_response.contains("Encoded path body."));

    let get_request = format!(
        "GET /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nmcp-session-id: {}\r\nConnection: close\r\n\r\n",
        session_id
    );
    let mut get_stream = TcpStream::connect(("127.0.0.1", port)).expect("get connect");
    get_stream
        .write_all(get_request.as_bytes())
        .expect("get write");
    let mut get_session = String::new();
    get_stream
        .read_to_string(&mut get_session)
        .expect("get read");
    assert!(get_session.contains("406"));

    let delete_request = format!(
        "DELETE /mcp HTTP/1.1\r\nHost: 127.0.0.1\r\nmcp-session-id: {}\r\nConnection: close\r\n\r\n",
        session_id
    );
    let mut delete_stream = TcpStream::connect(("127.0.0.1", port)).expect("delete connect");
    delete_stream
        .write_all(delete_request.as_bytes())
        .expect("delete write");
    let mut delete_response = String::new();
    delete_stream
        .read_to_string(&mut delete_response)
        .expect("delete read");
    assert!(delete_response.contains("200 OK"));

    let stale_session =
        http_post_json_with_session(port, "/mcp", &tools_list, &session_id).expect("stale session");
    assert!(stale_session.contains("404"));

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn native_mcp_http_query_does_not_require_qmd_oracle() {
    let (_temp, db_path) = fixture_env();
    let port = free_port();
    let mut child = qqd_command()
        .env("INDEX_PATH", &db_path)
        .env("QQD_QMD_ROOT", "/definitely/missing-qmd")
        .args(["mcp", "--http", "--port", &port.to_string()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("start native mcp");

    let query = serde_json::json!({
        "searches":[{"type":"lex","query":"alpha"}],
        "collections":["docs"]
    });
    let mut response = None;
    for _ in 0..20 {
        if let Ok(raw) = http_post_json(port, "/query", &query) {
            if raw.contains("\"results\"") {
                response = Some(raw);
                break;
            }
        }
        sleep(Duration::from_millis(200));
    }
    let response = response.expect("query response");
    assert!(response.contains("\"file\":\"docs/alpha.md\""));

    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn native_mcp_stdio_supports_initialize_and_tools_list() {
    let (_temp, db_path) = fixture_env();
    let mut child = qqd_command()
        .env("INDEX_PATH", &db_path)
        .arg("mcp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn stdio mcp");

    let initialize = serde_json::json!({
        "jsonrpc":"2.0",
        "id":1,
        "method":"initialize",
        "params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}
    });
    let init = stdio_rpc_roundtrip(&mut child, &initialize);
    assert_eq!(init["result"]["serverInfo"]["name"], "qmd");
    assert_ne!(init["result"]["serverInfo"]["version"], "0.1.0");

    let tools_list = serde_json::json!({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}});
    let tools = stdio_rpc_roundtrip(&mut child, &tools_list);
    let tool_names = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|tool| tool.get("name").and_then(|value| value.as_str()))
        .collect::<Vec<_>>();
    assert!(tool_names.contains(&"query"));
    assert!(tool_names.contains(&"get"));
    assert!(tool_names.contains(&"status"));
    assert!(tools.to_string().contains("\"searches\""));
    assert!(tools.to_string().contains("\"required\":[\"searches\"]"));

    let _ = child.kill();
    let _ = child.wait();
}
