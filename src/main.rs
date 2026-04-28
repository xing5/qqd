mod benchmark;
mod mcp;
mod models;
mod native;
mod sqlite_vec;

use anyhow::{Result, anyhow};
use std::env;

fn main() -> Result<()> {
    let args = env::args().collect::<Vec<_>>();

    if native::should_render_help(&args[1..]) {
        return native::run_help();
    }

    if let Some(command) = args.get(1).map(String::as_str) {
        if command == "bench-latency" {
            return benchmark::run(&args[2..]);
        }
        if command == "bench-quality" {
            return benchmark::run_quality(&args[2..]);
        }
        if command == "bench-metrics" {
            return benchmark::run_metrics(&args[2..]);
        }
    }

    if let Some(config) = native::SearchConfig::parse(&args[1..])? {
        return native::run_search(&config);
    }

    if let Some(config) = native::VSearchConfig::parse(&args[1..])? {
        return native::run_vsearch(&config);
    }

    if let Some(config) = native::QueryConfig::parse(&args[1..])? {
        return native::run_query(&config);
    }

    if let Some(config) = native::GetConfig::parse(&args[1..])? {
        return native::run_get(&config);
    }

    if let Some(config) = native::MultiGetConfig::parse(&args[1..])? {
        return native::run_multi_get(&config);
    }

    if let Some(config) = native::LsConfig::parse(&args[1..])? {
        return native::run_ls(&config);
    }

    if let Some(config) = native::StatusConfig::parse(&args[1..])? {
        return native::run_status(&config);
    }

    if let Some(config) = native::CollectionListConfig::parse(&args[1..])? {
        return native::run_collection_list(&config);
    }

    if let Some(config) = native::CollectionShowConfig::parse(&args[1..])? {
        return native::run_collection_show(&config);
    }

    if let Some(config) = native::CollectionAddConfig::parse(&args[1..])? {
        return native::run_collection_add(&config);
    }

    if let Some(config) = native::CollectionRemoveConfig::parse(&args[1..])? {
        return native::run_collection_remove(&config);
    }

    if let Some(config) = native::CollectionRenameConfig::parse(&args[1..])? {
        return native::run_collection_rename(&config);
    }

    if let Some(config) = native::ContextListConfig::parse(&args[1..])? {
        return native::run_context_list(&config);
    }

    if let Some(config) = native::ContextAddConfig::parse(&args[1..])? {
        return native::run_context_add(&config);
    }

    if let Some(config) = native::ContextRemoveConfig::parse(&args[1..])? {
        return native::run_context_remove(&config);
    }

    if let Some(config) = native::UpdateConfig::parse(&args[1..])? {
        return native::run_update(&config);
    }

    if let Some(config) = native::CleanupConfig::parse(&args[1..])? {
        return native::run_cleanup(&config);
    }

    if let Some(config) = native::EmbedConfig::parse(&args[1..])? {
        return native::run_embed(&config);
    }

    if let Some(config) = mcp::McpConfig::parse(&args[1..])? {
        return mcp::run(config);
    }

    Err(anyhow!(
        "unsupported qqd command: {}",
        args.get(1).map(String::as_str).unwrap_or("<none>")
    ))
}
