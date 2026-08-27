"""注册中心（Registry）：SQLite 存储智能体元信息、版本、评测结果、运行日志。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id              TEXT PRIMARY KEY,
    name            TEXT,
    description     TEXT,
    status          TEXT DEFAULT 'developing',  -- developing / registered / releasable
    source          TEXT DEFAULT 'developed',   -- developed / orchestrated
    current_version TEXT DEFAULT 'v1',
    agent_dir       TEXT,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS agent_versions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id   TEXT,
    version    TEXT,
    agent_dir  TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT,
    version     TEXT,
    metrics     TEXT,   -- JSON
    passed      INTEGER,
    report_path TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id          TEXT,
    version           TEXT,
    input             TEXT,
    output            TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        REAL,
    tool_calls        INTEGER,
    tool_errors       INTEGER,
    status            TEXT,
    created_at        TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---- 智能体 ----

    def register_agent(self, meta: dict) -> str:
        """注册（或更新）一个智能体。meta 需含 id/name/description/agent_dir。"""
        agent_id = meta["id"]
        now = _now()
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM agents WHERE id = ?", (agent_id,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE agents SET name=?, description=?, agent_dir=?, current_version=? WHERE id=?",
                    (
                        meta.get("name", agent_id),
                        meta.get("description", ""),
                        meta.get("agent_dir", ""),
                        meta.get("version", "v1"),
                        agent_id,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO agents (id, name, description, status, source, current_version, agent_dir, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        agent_id,
                        meta.get("name", agent_id),
                        meta.get("description", ""),
                        meta.get("status", "registered"),
                        meta.get("source", "developed"),
                        meta.get("version", "v1"),
                        meta.get("agent_dir", ""),
                        now,
                    ),
                )
            conn.execute(
                "INSERT INTO agent_versions (agent_id, version, agent_dir, created_at) VALUES (?, ?, ?, ?)",
                (agent_id, meta.get("version", "v1"), meta.get("agent_dir", ""), now),
            )
        return agent_id

    def get_agent(self, agent_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        return dict(row) if row else None

    def list_agents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, status, source, current_version, description, created_at "
                "FROM agents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def set_status(self, agent_id: str, status: str) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE agents SET status = ? WHERE id = ?", (status, agent_id))

    def get_versions(self, agent_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_versions WHERE agent_id = ? ORDER BY id DESC", (agent_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---- 评测 ----

    def save_evaluation(
        self,
        agent_id: str,
        version: str,
        metrics: dict,
        passed: bool,
        report_path: str,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO evaluations (agent_id, version, metrics, passed, report_path, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    version,
                    json.dumps(metrics, ensure_ascii=False),
                    int(passed),
                    report_path,
                    _now(),
                ),
            )
            return cur.lastrowid

    def latest_evaluation(self, agent_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM evaluations WHERE agent_id = ? ORDER BY id DESC LIMIT 1", (agent_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["metrics"] = json.loads(d["metrics"] or "{}")
        return d

    # ---- 运行日志 ----

    def save_run(self, agent_id: str, version: str, user_input: str, run) -> int:
        """保存一次运行日志。run 为 runtime.RunResult。"""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO runs (agent_id, version, input, output, prompt_tokens, "
                "completion_tokens, latency_ms, tool_calls, tool_errors, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    agent_id,
                    version,
                    user_input,
                    run.output,
                    run.prompt_tokens,
                    run.completion_tokens,
                    run.latency_ms,
                    run.tool_calls,
                    run.tool_errors,
                    "error" if run.error else "ok",
                    _now(),
                ),
            )
            return cur.lastrowid

    def list_runs(self, agent_id: str, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs WHERE agent_id = ? ORDER BY id DESC LIMIT ?", (agent_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]
