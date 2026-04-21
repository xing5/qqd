use crate::native;
use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use tiny_http::{Header, Method, Response, Server, StatusCode};

const SESSION_ID: &str = "qqd-session";
const QMD_COMPAT_VERSION: &str = "2.1.0";
static SESSION_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpConfig {
    pub index_name: Option<String>,
    pub http: bool,
    pub daemon: bool,
    pub stop: bool,
    pub port: u16,
}

impl McpConfig {
    pub fn parse(args: &[String]) -> Result<Option<Self>> {
        if args.is_empty() {
            return Ok(None);
        }
        let (index_name, rest) = native::split_index_args(args)?;
        if rest.first().map(String::as_str) != Some("mcp") {
            return Ok(None);
        }

        let mut http = false;
        let mut daemon = false;
        let mut stop = false;
        let mut port = 8181u16;
        let mut idx = 1usize;
        while idx < rest.len() {
            match rest[idx].as_str() {
                "--http" => {
                    http = true;
                    idx += 1;
                }
                "--daemon" => {
                    daemon = true;
                    idx += 1;
                }
                "--port" => {
                    let value = rest
                        .get(idx + 1)
                        .ok_or_else(|| anyhow!("--port requires a value"))?;
                    port = value.parse().with_context(|| "invalid --port value")?;
                    idx += 2;
                }
                "stop" => {
                    stop = true;
                    idx += 1;
                }
                _ => return Ok(None),
            }
        }

        Ok(Some(Self {
            index_name,
            http,
            daemon,
            stop,
            port,
        }))
    }
}

pub fn run(config: McpConfig) -> Result<()> {
    if config.stop {
        return stop_server();
    }
    if !config.http {
        return run_stdio_server(config.index_name.as_deref());
    }
    if config.daemon {
        return start_daemon(&config);
    }
    run_http_server(&config)
}

fn cache_dir() -> PathBuf {
    if let Ok(root) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(root).join("qmd");
    }
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".to_string()))
        .join(".cache")
        .join("qmd")
}

fn pid_path() -> PathBuf {
    cache_dir().join("mcp.pid")
}

fn start_daemon(config: &McpConfig) -> Result<()> {
    fs::create_dir_all(cache_dir())?;
    let exe = std::env::current_exe()?;
    let mut command = Command::new(exe);
    if let Some(index_name) = &config.index_name {
        command.arg("--index").arg(index_name);
    }
    command.args(["mcp", "--http", "--port", &config.port.to_string()]);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let child = command.spawn().context("failed to start MCP daemon")?;
    fs::write(pid_path(), child.id().to_string())?;
    println!(
        "Started on http://localhost:{}/mcp (PID {})",
        config.port,
        child.id()
    );
    Ok(())
}

fn stop_server() -> Result<()> {
    let pid = fs::read_to_string(pid_path()).context("Not running (no PID file).")?;
    let pid = pid.trim().to_string();
    let _ = Command::new("kill").args(["-TERM", &pid]).status();
    let _ = fs::remove_file(pid_path());
    println!("Stopped QQD MCP server (PID {}).", pid);
    Ok(())
}

fn run_http_server(config: &McpConfig) -> Result<()> {
    let server = Server::http(("127.0.0.1", config.port))
        .map_err(|error| anyhow!("failed to bind HTTP server: {error}"))?;
    let mut sessions: HashSet<String> = HashSet::new();
    for mut request in server.incoming_requests() {
        let method = request.method().clone();
        let url = request.url().to_string();
        let mut body = String::new();
        request.as_reader().read_to_string(&mut body).ok();
        let session_header = request
            .headers()
            .iter()
            .find(|header| header.field.equiv("mcp-session-id"))
            .map(|header| header.value.to_string());

        let response = match (method, url.as_str()) {
            (Method::Get, "/health") => json_response(json!({"status":"ok","uptime":0}), 200, None),
            (Method::Post, "/query") | (Method::Post, "/search") => {
                handle_query_endpoint(&body, config.index_name.as_deref())
            }
            (Method::Get, "/mcp") => {
                if let Some(session) = session_header.as_deref() {
                    if !sessions.contains(session) {
                        json_response(
                            json!({"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null}),
                            404,
                            None,
                        )
                    } else if !request
                        .headers()
                        .iter()
                        .find(|header| header.field.equiv("Accept"))
                        .map(|header| header.value.to_string())
                        .unwrap_or_default()
                        .contains("text/event-stream")
                    {
                        text_response("Client must accept text/event-stream", 406)
                    } else {
                        std::thread::sleep(std::time::Duration::from_secs(60));
                        text_response("", 200)
                    }
                } else {
                    json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: Missing session ID"},"id":null}),
                        400,
                        None,
                    )
                }
            }
            (Method::Delete, "/mcp") => {
                if let Some(session) = session_header.as_deref() {
                    if sessions.remove(session) {
                        json_response(json!({"jsonrpc":"2.0","result":{}}), 200, None)
                    } else {
                        json_response(
                            json!({"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null}),
                            404,
                            None,
                        )
                    }
                } else {
                    json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: Missing session ID"},"id":null}),
                        400,
                        None,
                    )
                }
            }
            (Method::Post, "/mcp") => handle_mcp_request(
                &body,
                config.index_name.as_deref(),
                session_header.as_deref(),
                &mut sessions,
            ),
            _ => text_response("Not found", 404),
        }?;
        let _ = request.respond(response);
    }
    Ok(())
}

fn handle_query_endpoint(
    body: &str,
    index_name: Option<&str>,
) -> Result<Response<std::io::Cursor<Vec<u8>>>> {
    let value: Value = serde_json::from_str(body)?;
    let searches = value
        .get("searches")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Missing required field: searches (array)"))?;
    let query = searches.to_vec();
    let limit = value
        .get("limit")
        .and_then(Value::as_u64)
        .map(|v| v as usize);
    let collections = value
        .get("collections")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let min_score = value.get("minScore").and_then(Value::as_f64);
    let intent = value.get("intent").and_then(Value::as_str);
    let rerank = value.get("rerank").and_then(Value::as_bool).unwrap_or(true);
    let results = native::query_results_for_mcp(
        index_name,
        &query,
        limit,
        min_score,
        &collections,
        intent,
        rerank,
    )?;
    json_response(json!({ "results": results }), 200, None)
}

fn handle_mcp_request(
    body: &str,
    index_name: Option<&str>,
    session_header: Option<&str>,
    sessions: &mut HashSet<String>,
) -> Result<Response<std::io::Cursor<Vec<u8>>>> {
    let request: Value = serde_json::from_str(body)?;
    let id = request.get("id").cloned().unwrap_or(json!(null));
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();

    let (result, session_header) = match method {
        "initialize" => (
            json!({
                "protocolVersion":"2025-03-26",
                "serverInfo": { "name":"qmd", "version": qmd_version() },
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "instructions": build_instructions(index_name)?
            }),
            {
                let session = format!(
                    "{}-{}",
                    SESSION_ID,
                    SESSION_COUNTER.fetch_add(1, Ordering::Relaxed)
                );
                sessions.insert(session.clone());
                Some(session)
            },
        ),
        "tools/list" => {
            match session_header {
                None => {
                    return json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: Missing session ID"},"id":id}),
                        400,
                        None,
                    );
                }
                Some(session) if !sessions.contains(session) => {
                    return json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null}),
                        404,
                        None,
                    );
                }
                _ => {}
            }
            (
                json!({
                    "tools": [
                        {"name":"query","title":"Query","description":"Search the knowledge base using typed sub-queries.","inputSchema":{"type":"object","required":["searches"],"properties":{"searches":{"type":"array","items":{"type":"object","required":["type","query"],"properties":{"type":{"type":"string","enum":["lex","vec","hyde"]},"query":{"type":"string"}}}},"limit":{"type":"number","default":10},"minScore":{"type":"number","default":0},"candidateLimit":{"type":"number"},"collections":{"type":"array","items":{"type":"string"}},"intent":{"type":"string"},"rerank":{"type":"boolean","default":true}}}},
                        {"name":"get","title":"Get Document","description":"Retrieve a document by path or docid.","inputSchema":{"type":"object","properties":{"file":{"type":"string"},"fromLine":{"type":"number"},"maxLines":{"type":"number"},"lineNumbers":{"type":"boolean"}}}},
                        {"name":"multi_get","title":"Multi-Get Documents","description":"Retrieve multiple documents by glob or comma-separated list.","inputSchema":{"type":"object","properties":{"pattern":{"type":"string"},"maxLines":{"type":"number"},"maxBytes":{"type":"number"},"lineNumbers":{"type":"boolean"}}}},
                        {"name":"status","title":"Index Status","description":"Show index status and collections.","inputSchema":{"type":"object","properties":{}}}
                    ]
                }),
                None,
            )
        }
        "tools/call" => {
            match session_header {
                None => {
                    return json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: Missing session ID"},"id":id}),
                        400,
                        None,
                    );
                }
                Some(session) if !sessions.contains(session) => {
                    return json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null}),
                        404,
                        None,
                    );
                }
                _ => {}
            }
            let params = request.get("params").cloned().unwrap_or(json!({}));
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            let result = match name {
                "query" => {
                    let searches = arguments
                        .get("searches")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let limit = arguments
                        .get("limit")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    let collections = arguments
                        .get("collections")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .map(ToOwned::to_owned)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let min_score = arguments.get("minScore").and_then(Value::as_f64);
                    let intent = arguments.get("intent").and_then(Value::as_str);
                    let rerank = arguments
                        .get("rerank")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    let results = native::query_results_for_mcp(
                        index_name,
                        &searches,
                        limit,
                        min_score,
                        &collections,
                        intent,
                        rerank,
                    )?;
                    json!({
                        "content":[{"type":"text","text": format_query_summary(&results, &searches)}],
                        "structuredContent":{"results":results}
                    })
                }
                "get" => {
                    let path = arguments
                        .get("path")
                        .or_else(|| arguments.get("file"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let mut from_line = arguments
                        .get("fromLine")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    let mut lookup = path.to_string();
                    if from_line.is_none() {
                        if let Some((candidate, line)) = split_line_suffix(path) {
                            lookup = candidate;
                            from_line = Some(line);
                        }
                    }
                    let max_lines = arguments
                        .get("maxLines")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    let line_numbers = arguments
                        .get("lineNumbers")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    match native::get_document_for_mcp(
                        index_name,
                        &lookup,
                        from_line,
                        max_lines,
                        line_numbers,
                    ) {
                        Ok(doc) => {
                            json!({"content":[{"type":"resource","resource":{"uri":format!("qmd://{}", native::encode_qmd_path(&doc.file)),"name":doc.file,"title":doc.title,"mimeType":"text/markdown","text":doc.body}}]})
                        }
                        Err(_) => {
                            json!({"content":[{"type":"text","text": format!("Document not found: {}", path)}],"isError": true})
                        }
                    }
                }
                "multi_get" => {
                    let pattern = arguments
                        .get("pattern")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let max_lines = arguments
                        .get("maxLines")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    let max_bytes = arguments
                        .get("maxBytes")
                        .and_then(Value::as_u64)
                        .map(|value| value as usize);
                    let line_numbers = arguments
                        .get("lineNumbers")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let docs = native::multi_get_for_mcp(
                        index_name,
                        pattern,
                        max_lines,
                        max_bytes,
                        line_numbers,
                    )?;
                    if docs.is_empty() {
                        json!({"content":[{"type":"text","text": format!("No files matched pattern: {}", pattern)}],"isError": true})
                    } else {
                        json!({"content":docs})
                    }
                }
                "status" => {
                    let status = native::status_for_mcp(index_name)?;
                    json!({"content":[{"type":"text","text": format_status_summary(&status)}],"structuredContent":status})
                }
                _ => {
                    return json_response(
                        json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"method not found"}}),
                        200,
                        None,
                    );
                }
            };
            (result, None)
        }
        "resources/read" => {
            match session_header {
                None => {
                    return json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32000,"message":"Bad Request: Missing session ID"},"id":id}),
                        400,
                        None,
                    );
                }
                Some(session) if !sessions.contains(session) => {
                    return json_response(
                        json!({"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"},"id":null}),
                        404,
                        None,
                    );
                }
                _ => {}
            }
            let params = request.get("params").cloned().unwrap_or(json!({}));
            let uri = params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let target = native::decode_qmd_path(uri.trim_start_matches("qmd://"));
            let result = match native::get_document_for_mcp(index_name, &target, None, None, true) {
                Ok(doc) => json!({
                    "contents":[{
                        "uri": uri,
                        "name": doc.file,
                        "title": doc.title,
                        "mimeType": "text/markdown",
                        "text": doc.body
                    }]
                }),
                Err(_) => json!({
                    "contents":[{
                        "uri": uri,
                        "text": format!("Document not found: {}", target)
                    }]
                }),
            };
            (result, None)
        }
        _ => {
            return json_response(
                json!({"jsonrpc":"2.0","id":id,"error":{"code":-32601,"message":"method not found"}}),
                200,
                None,
            );
        }
    };

    json_response(
        json!({
            "jsonrpc":"2.0",
            "id": id,
            "result": result
        }),
        200,
        session_header.as_deref(),
    )
}

fn json_response(
    body: Value,
    status: u16,
    session_header: Option<&str>,
) -> Result<Response<std::io::Cursor<Vec<u8>>>> {
    let mut response =
        Response::from_data(serde_json::to_vec(&body)?).with_status_code(StatusCode(status));
    response.add_header(Header::from_bytes("Content-Type", "application/json").unwrap());
    if let Some(session_header) = session_header {
        response.add_header(Header::from_bytes("mcp-session-id", session_header).unwrap());
    }
    Ok(response)
}

fn text_response(body: &str, status: u16) -> Result<Response<std::io::Cursor<Vec<u8>>>> {
    Ok(Response::from_string(body).with_status_code(StatusCode(status)))
}

fn build_instructions(index_name: Option<&str>) -> Result<String> {
    let status = native::status_for_mcp(index_name)?;
    let mut lines = vec![format!(
        "QQD is your local search engine over {} markdown documents.",
        status.total_documents
    )];
    let file_config = native::load_file_config(index_name)?;
    if let Some(global) = file_config.global_context {
        lines.push(format!("Context: {}", global));
    }
    if !status.collections.is_empty() {
        lines.push(String::new());
        lines.push("Collections (scope with `collection` parameter):".to_string());
        for collection in status.collections {
            let root_context = file_config
                .collections
                .get(&collection.name)
                .and_then(|collection| collection.context.as_ref())
                .and_then(|map| map.get("/").or_else(|| map.get("")))
                .cloned();
            let desc = root_context
                .map(|context| format!(" — {}", context))
                .unwrap_or_default();
            lines.push(format!(
                "  - \"{}\" ({} docs){}",
                collection.name, collection.documents, desc
            ));
        }
    }
    if !status.has_vector_index {
        lines.push(String::new());
        lines.push(
            "Note: No vector embeddings yet. Run `qqd embed` to enable semantic search (vec/hyde)."
                .to_string(),
        );
    } else if status.needs_embedding > 0 {
        lines.push(String::new());
        lines.push(format!(
            "Note: {} documents need embedding. Run `qqd embed` to update.",
            status.needs_embedding
        ));
    }
    lines.push(String::new());
    lines.push("Search: Use `query` with sub-queries (lex/vec/hyde):".to_string());
    lines.push("  - type:'lex' — BM25 keyword search (exact terms, fast)".to_string());
    lines.push("  - type:'vec' — semantic vector search (meaning-based)".to_string());
    lines.push(
        "  - type:'hyde' — hypothetical document (write what the answer looks like)".to_string(),
    );
    lines.push(String::new());
    lines.push(
        "  Always provide `intent` on every search call to disambiguate and improve snippets."
            .to_string(),
    );
    lines.push(String::new());
    lines.push("Examples:".to_string());
    lines.push("  Quick keyword lookup: [{type:'lex', query:'error handling'}]".to_string());
    lines.push(
        "  Semantic search: [{type:'vec', query:'how to handle errors gracefully'}]".to_string(),
    );
    lines.push("  Best results: [{type:'lex', query:'error'}, {type:'vec', query:'error handling best practices'}]".to_string());
    lines.push(
        "  With intent: searches=[{type:'lex', query:'performance'}], intent='web page load times'"
            .to_string(),
    );
    lines.push(String::new());
    lines.push("Retrieval:".to_string());
    lines.push("  - `get` — single document by path or docid (#abc123). Supports line offset (`file.md:100`).".to_string());
    lines.push("  - `multi_get` — batch retrieve by glob (`journals/2025-05*.md`) or comma-separated list.".to_string());
    lines.push(String::new());
    lines.push("Tips:".to_string());
    lines.push("  - File paths in results are relative to their collection.".to_string());
    lines.push("  - Use `minScore: 0.5` to filter low-confidence results.".to_string());
    lines.push("  - Results include a `context` field describing the content type.".to_string());
    Ok(lines.join("\n"))
}

fn format_query_summary(results: &[native::McpSearchResult], searches: &[Value]) -> String {
    let query = searches
        .iter()
        .find_map(|search| search.get("query").and_then(Value::as_str))
        .unwrap_or_default();
    if results.is_empty() {
        return format!("No results found for \"{}\"", query);
    }
    let mut lines = vec![format!(
        "Found {} result{} for \"{}\":\n",
        results.len(),
        if results.len() == 1 { "" } else { "s" },
        query
    )];
    for result in results {
        lines.push(format!(
            "{} {}% {} - {}",
            result.docid,
            (result.score * 100.0).round() as i64,
            result.file,
            result.title
        ));
    }
    lines.join("\n")
}

fn format_status_summary(status: &native::McpStatus) -> String {
    let mut lines = vec![
        "QQD Index Status:".to_string(),
        format!("  Total documents: {}", status.total_documents),
        format!("  Needs embedding: {}", status.needs_embedding),
        format!(
            "  Vector index: {}",
            if status.has_vector_index { "yes" } else { "no" }
        ),
        format!("  Collections: {}", status.collections.len()),
    ];
    for collection in &status.collections {
        lines.push(format!(
            "    - {}: {} ({} docs)",
            collection.name, collection.path, collection.documents
        ));
    }
    lines.join("\n")
}

fn split_line_suffix(value: &str) -> Option<(String, usize)> {
    let (path, line) = value.rsplit_once(':')?;
    if path.starts_with('#') {
        return None;
    }
    Some((path.to_string(), line.parse().ok()?))
}

fn qmd_version() -> String {
    QMD_COMPAT_VERSION.to_string()
}

fn run_stdio_server(index_name: Option<&str>) -> Result<()> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut stdout = std::io::stdout().lock();

    loop {
        let Some(body) = read_content_length_message(&mut reader)? else {
            break;
        };
        let response = handle_stdio_request(&body, index_name)?;
        let payload = serde_json::to_vec(&response)?;
        write!(stdout, "Content-Length: {}\r\n\r\n", payload.len())?;
        stdout.write_all(&payload)?;
        stdout.flush()?;
    }

    Ok(())
}

fn read_content_length_message<R: BufRead>(reader: &mut R) -> Result<Option<String>> {
    let mut content_length = None;
    loop {
        let mut line = String::new();
        let read = reader.read_line(&mut line)?;
        if read == 0 {
            return Ok(None);
        }
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            break;
        }
        if let Some(value) = line.strip_prefix("Content-Length:") {
            content_length = Some(value.trim().parse::<usize>()?);
        }
    }
    let content_length = content_length.ok_or_else(|| anyhow!("missing Content-Length header"))?;
    let mut body = vec![0u8; content_length];
    reader.read_exact(&mut body)?;
    Ok(Some(String::from_utf8(body)?))
}

fn handle_stdio_request(body: &str, index_name: Option<&str>) -> Result<Value> {
    let request: Value = serde_json::from_str(body)?;
    let id = request.get("id").cloned().unwrap_or(json!(null));
    let method = request
        .get("method")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let result = match method {
        "initialize" => json!({
            "protocolVersion":"2025-03-26",
            "serverInfo": { "name":"qmd", "version": qmd_version() },
            "capabilities": { "tools": {}, "resources": {} },
            "instructions": build_instructions(index_name)?
        }),
        "tools/list" => json!({
            "tools": [
                {"name":"query","title":"Query","description":"Search the knowledge base using typed sub-queries.","inputSchema":{"type":"object","required":["searches"],"properties":{"searches":{"type":"array","items":{"type":"object","required":["type","query"],"properties":{"type":{"type":"string","enum":["lex","vec","hyde"]},"query":{"type":"string"}}}},"limit":{"type":"number","default":10},"minScore":{"type":"number","default":0},"candidateLimit":{"type":"number"},"collections":{"type":"array","items":{"type":"string"}},"intent":{"type":"string"},"rerank":{"type":"boolean","default":true}}}},
                {"name":"get","title":"Get Document","description":"Retrieve a document by path or docid.","inputSchema":{"type":"object","properties":{"file":{"type":"string"},"fromLine":{"type":"number"},"maxLines":{"type":"number"},"lineNumbers":{"type":"boolean"}}}},
                {"name":"multi_get","title":"Multi-Get Documents","description":"Retrieve multiple documents by glob or comma-separated list.","inputSchema":{"type":"object","properties":{"pattern":{"type":"string"},"maxLines":{"type":"number"},"maxBytes":{"type":"number"},"lineNumbers":{"type":"boolean"}}}},
                {"name":"status","title":"Index Status","description":"Show index status and collections.","inputSchema":{"type":"object","properties":{}}}
            ]
        }),
        "tools/call" => {
            let params = request.get("params").cloned().unwrap_or(json!({}));
            let name = params
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let arguments = params.get("arguments").cloned().unwrap_or(json!({}));
            match name {
                "query" => {
                    let searches = arguments
                        .get("searches")
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let limit = arguments
                        .get("limit")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize);
                    let collections = arguments
                        .get("collections")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .filter_map(Value::as_str)
                                .map(ToOwned::to_owned)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default();
                    let min_score = arguments.get("minScore").and_then(Value::as_f64);
                    let intent = arguments.get("intent").and_then(Value::as_str);
                    let rerank = arguments
                        .get("rerank")
                        .and_then(Value::as_bool)
                        .unwrap_or(true);
                    let results = native::query_results_for_mcp(
                        index_name,
                        &searches,
                        limit,
                        min_score,
                        &collections,
                        intent,
                        rerank,
                    )?;
                    json!({"content":[{"type":"text","text": format_query_summary(&results, &searches)}],"structuredContent":{"results":results}})
                }
                "get" => {
                    let path = arguments
                        .get("path")
                        .or_else(|| arguments.get("file"))
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let mut from_line = arguments
                        .get("fromLine")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize);
                    let mut lookup = path.to_string();
                    if from_line.is_none() {
                        if let Some((candidate, line)) = split_line_suffix(path) {
                            lookup = candidate;
                            from_line = Some(line);
                        }
                    }
                    let max_lines = arguments
                        .get("maxLines")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize);
                    let line_numbers = arguments
                        .get("lineNumbers")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    match native::get_document_for_mcp(
                        index_name,
                        &lookup,
                        from_line,
                        max_lines,
                        line_numbers,
                    ) {
                        Ok(doc) => {
                            json!({"content":[{"type":"resource","resource":{"uri":format!("qmd://{}", native::encode_qmd_path(&doc.file)),"name":doc.file,"title":doc.title,"mimeType":"text/markdown","text":doc.body}}]})
                        }
                        Err(_) => {
                            json!({"content":[{"type":"text","text": format!("Document not found: {}", path)}],"isError": true})
                        }
                    }
                }
                "multi_get" => {
                    let pattern = arguments
                        .get("pattern")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let max_lines = arguments
                        .get("maxLines")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize);
                    let max_bytes = arguments
                        .get("maxBytes")
                        .and_then(Value::as_u64)
                        .map(|v| v as usize);
                    let line_numbers = arguments
                        .get("lineNumbers")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let docs = native::multi_get_for_mcp(
                        index_name,
                        pattern,
                        max_lines,
                        max_bytes,
                        line_numbers,
                    )?;
                    if docs.is_empty() {
                        json!({"content":[{"type":"text","text": format!("No files matched pattern: {}", pattern)}],"isError": true})
                    } else {
                        json!({"content":docs})
                    }
                }
                "status" => {
                    let status = native::status_for_mcp(index_name)?;
                    json!({"content":[{"type":"text","text": format_status_summary(&status)}],"structuredContent":status})
                }
                _ => json!({"error":{"code":-32601,"message":"method not found"}}),
            }
        }
        "resources/read" => {
            let params = request.get("params").cloned().unwrap_or(json!({}));
            let uri = params
                .get("uri")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let target = native::decode_qmd_path(uri.trim_start_matches("qmd://"));
            match native::get_document_for_mcp(index_name, &target, None, None, true) {
                Ok(doc) => json!({
                    "contents":[{
                        "uri": uri,
                        "name": doc.file,
                        "title": doc.title,
                        "mimeType": "text/markdown",
                        "text": doc.body
                    }]
                }),
                Err(_) => json!({
                    "contents":[{
                        "uri": uri,
                        "text": format!("Document not found: {}", target)
                    }]
                }),
            }
        }
        _ => json!({"error":{"code":-32601,"message":"method not found"}}),
    };
    Ok(json!({"jsonrpc":"2.0","id":id,"result":result}))
}
