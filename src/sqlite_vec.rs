#![allow(dead_code)]

use anyhow::{Context, Result, anyhow};
use rusqlite::{Connection, LoadExtensionGuard};
use std::env;
use std::path::{Path, PathBuf};

const SQLITE_VEC_ENV: &str = "QQD_SQLITE_VEC_PATH";
const DEFAULT_QMD_SQLITE_VEC_PATHS: &[&str] = &[
    "/tmp/qmd-upstream/node_modules/sqlite-vec-linux-x64/vec0.so",
    "/tmp/qmd-upstream/node_modules/.pnpm/sqlite-vec-linux-x64@0.1.9/node_modules/sqlite-vec-linux-x64/vec0.so",
];

pub fn discover_vec0_path() -> Option<PathBuf> {
    env::var(SQLITE_VEC_ENV)
        .ok()
        .map(PathBuf::from)
        .filter(|path| path.is_file())
        .or_else(|| {
            DEFAULT_QMD_SQLITE_VEC_PATHS
                .iter()
                .map(PathBuf::from)
                .find(|path| path.is_file())
        })
}

pub fn required_vec0_path() -> Result<PathBuf> {
    discover_vec0_path().ok_or_else(|| {
        anyhow!(
            "sqlite-vec extension not found; set {SQLITE_VEC_ENV} or install upstream qmd under /tmp/qmd-upstream"
        )
    })
}

pub fn load_vec0_extension(connection: &Connection, path: &Path) -> Result<()> {
    unsafe {
        let _guard = LoadExtensionGuard::new(connection)?;
        connection
            .load_extension(path, None)
            .with_context(|| format!("failed to load sqlite-vec extension at {}", path.display()))
    }
}

pub fn vectors_vec_is_usable(connection: &Connection) -> Result<bool> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE name = 'vectors_vec'",
        [],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Ok(false);
    }
    let Some(path) = discover_vec0_path() else {
        return Ok(false);
    };
    load_vec0_extension(connection, &path)?;
    let mut statement = connection.prepare("SELECT 1 FROM vectors_vec LIMIT 0")?;
    let _ = statement.query([])?;
    Ok(true)
}

pub fn ensure_vec0_table(
    connection: &Connection,
    table_name: &str,
    dimensions: usize,
) -> Result<()> {
    connection.execute_batch(&format!(
        "DROP TABLE IF EXISTS {table_name};
         CREATE VIRTUAL TABLE {table_name} USING vec0(
           hash_seq TEXT PRIMARY KEY,
           embedding float[{dimensions}] distance_metric=cosine
         );"
    ))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_vec0_path_from_upstream_install() {
        let path = required_vec0_path().expect("vec0 path");
        assert!(path.ends_with("vec0.so"));
        assert!(path.is_file());
    }

    #[test]
    fn vec0_supports_qmd_style_lifecycle() {
        let path = required_vec0_path().expect("vec0 path");
        let connection = Connection::open_in_memory().expect("memory db");

        load_vec0_extension(&connection, &path).expect("load vec0");
        ensure_vec0_table(&connection, "vectors_vec", 3).expect("create vectors_vec");

        connection
            .execute(
                "INSERT INTO vectors_vec(hash_seq, embedding) VALUES (?1, ?2)",
                ("a_0", "[1,0,0]"),
            )
            .expect("insert a_0");
        connection
            .execute(
                "INSERT INTO vectors_vec(hash_seq, embedding) VALUES (?1, ?2)",
                ("b_0", "[0,1,0]"),
            )
            .expect("insert b_0");

        let initial = connection
            .prepare(
                "SELECT hash_seq, distance
                 FROM vectors_vec
                 WHERE embedding MATCH ?1 AND k = 2",
            )
            .expect("prepare initial search")
            .query_map(["[1,0,0]"], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .expect("query initial search")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("collect initial search");
        assert_eq!(initial[0].0, "a_0");
        assert_eq!(initial[0].1, 0.0);

        connection
            .execute("DELETE FROM vectors_vec WHERE hash_seq = ?1", ["a_0"])
            .expect("delete a_0");
        connection
            .execute(
                "INSERT INTO vectors_vec(hash_seq, embedding) VALUES (?1, ?2)",
                ("a_0", "[0.5,0.5,0]"),
            )
            .expect("reinsert a_0");

        let replaced = connection
            .prepare(
                "SELECT hash_seq, distance
                 FROM vectors_vec
                 WHERE embedding MATCH ?1 AND k = 2",
            )
            .expect("prepare replaced search")
            .query_map(["[1,0,0]"], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
            })
            .expect("query replaced search")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("collect replaced search");
        assert_eq!(replaced[0].0, "a_0");
        assert!(replaced[0].1 > 0.0);
        assert!(replaced[0].1 < 1.0);

        connection
            .execute_batch(
                "DELETE FROM vectors_vec;
                 DROP TABLE IF EXISTS vectors_vec;",
            )
            .expect("cleanup vectors_vec");

        let exists: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'vectors_vec'",
                [],
                |row| row.get(0),
            )
            .expect("sqlite_master count");
        assert_eq!(exists, 0);
    }
}
