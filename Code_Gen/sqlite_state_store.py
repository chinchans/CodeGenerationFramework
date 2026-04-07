from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


IST = timezone(timedelta(hours=5, minutes=30))


def _ist_now_iso() -> str:
    # Store all timestamps in IST for consistency with user expectations.
    return datetime.now(IST).replace(microsecond=0).isoformat()


def _json_dumps_small(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return json.dumps({"_nonserializable": str(value)}, ensure_ascii=False, separators=(",", ":"))


def _ensure_parent_dir(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY,
  intent TEXT,
  template_path TEXT,
  code_generation_prompt TEXT,
  code_generation_prompt_path TEXT,
  resolved_summary_json TEXT,
  ambiguity_questions_json TEXT,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_runs (
  stage_run_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  stage_name TEXT NOT NULL,
  stage_status TEXT NOT NULL,
  output_json TEXT,
  input_summary_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  UNIQUE(session_id, stage_name)
);

CREATE TABLE IF NOT EXISTS agent_runs (
  agent_run_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  stage_run_id TEXT,
  agent_name TEXT NOT NULL,
  agent_version TEXT,
  run_status TEXT NOT NULL,
  input_json TEXT,
  output_json TEXT,
  error_json TEXT,
  started_at TEXT,
  completed_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
  FOREIGN KEY(stage_run_id) REFERENCES stage_runs(stage_run_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS code_validations (
  validation_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL,
  branch TEXT NOT NULL,
  results_json TEXT NOT NULL,
  status_messages_json TEXT NOT NULL,
  git_pull_output TEXT,
  git_diff_output TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_stage_runs_session_id ON stage_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_session_id ON agent_runs(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_stage_run_id ON agent_runs(stage_run_id);
CREATE INDEX IF NOT EXISTS idx_code_validations_session_id ON code_validations(session_id);
"""


@dataclass(frozen=True)
class AgentRunHandle:
    agent_run_id: str


class SqliteStateStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        _ensure_parent_dir(self.db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def ensure_session(
        self,
        *,
        session_id: str,
        intent: Optional[str] = None,
        status: str = "started",
        template_path: Optional[str] = None,
        code_generation_prompt: Optional[str] = None,
        code_generation_prompt_path: Optional[str] = None,
        ambiguity_questions: Any = None,
        resolved_summary: Any = None,
    ) -> None:
        now = _ist_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions(
                  session_id,intent,template_path,code_generation_prompt,code_generation_prompt_path,
                  resolved_summary_json,ambiguity_questions_json,status,created_at,updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                  intent=COALESCE(excluded.intent, sessions.intent),
                  template_path=COALESCE(excluded.template_path, sessions.template_path),
                  code_generation_prompt=COALESCE(excluded.code_generation_prompt, sessions.code_generation_prompt),
                  code_generation_prompt_path=COALESCE(excluded.code_generation_prompt_path, sessions.code_generation_prompt_path),
                  resolved_summary_json=COALESCE(excluded.resolved_summary_json, sessions.resolved_summary_json),
                  ambiguity_questions_json=COALESCE(excluded.ambiguity_questions_json, sessions.ambiguity_questions_json),
                  status=excluded.status,
                  updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    intent,
                    template_path,
                    code_generation_prompt,
                    code_generation_prompt_path,
                    _json_dumps_small(resolved_summary),
                    _json_dumps_small(ambiguity_questions),
                    status,
                    now,
                    now,
                ),
            )

    def upsert_stage_run(
        self,
        *,
        session_id: str,
        stage_name: str,
        stage_status: str,
        input_summary: Any = None,
        output: Any = None,
    ) -> str:
        now = _ist_now_iso()
        stage_run_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO stage_runs(
                  stage_run_id,session_id,stage_name,stage_status,output_json,input_summary_json,created_at,updated_at
                )
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id, stage_name) DO UPDATE SET
                  stage_status=excluded.stage_status,
                  output_json=COALESCE(excluded.output_json, stage_runs.output_json),
                  input_summary_json=COALESCE(excluded.input_summary_json, stage_runs.input_summary_json),
                  updated_at=excluded.updated_at
                """,
                (
                    stage_run_id,
                    session_id,
                    stage_name,
                    stage_status,
                    _json_dumps_small(output),
                    _json_dumps_small(input_summary),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT stage_run_id FROM stage_runs WHERE session_id=? AND stage_name=?",
                (session_id, stage_name),
            ).fetchone()
            return str(row["stage_run_id"]) if row else stage_run_id

    def start_agent_run(
        self,
        *,
        session_id: str,
        agent_name: str,
        stage_run_id: Optional[str] = None,
        agent_version: Optional[str] = None,
        input_payload: Any = None,
    ) -> AgentRunHandle:
        now = _ist_now_iso()
        agent_run_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_runs(
                  agent_run_id,session_id,stage_run_id,agent_name,agent_version,run_status,
                  input_json,output_json,error_json,started_at,completed_at,created_at,updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    agent_run_id,
                    session_id,
                    stage_run_id,
                    agent_name,
                    agent_version,
                    "started",
                    _json_dumps_small(input_payload),
                    None,
                    None,
                    now,
                    None,
                    now,
                    now,
                ),
            )
        return AgentRunHandle(agent_run_id=agent_run_id)

    def complete_agent_run(
        self,
        handle: AgentRunHandle,
        *,
        output_payload: Any = None,
        error_payload: Any = None,
        status: str = "completed",
    ) -> None:
        now = _ist_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_runs
                SET run_status=?,
                    output_json=COALESCE(?, output_json),
                    error_json=COALESCE(?, error_json),
                    completed_at=COALESCE(completed_at, ?),
                    updated_at=?
                WHERE agent_run_id=?
                """,
                (
                    status,
                    _json_dumps_small(output_payload),
                    _json_dumps_small(error_payload),
                    now,
                    now,
                    handle.agent_run_id,
                ),
            )

    @contextmanager
    def agent_run(
        self,
        *,
        session_id: str,
        agent_name: str,
        stage_run_id: Optional[str] = None,
        agent_version: Optional[str] = None,
        input_payload: Any = None,
        output_capture_keys: Optional[Iterable[str]] = None,
    ):
        handle = self.start_agent_run(
            session_id=session_id,
            agent_name=agent_name,
            stage_run_id=stage_run_id,
            agent_version=agent_version,
            input_payload=input_payload,
        )
        try:
            result = yield handle
            output_payload = result
            if output_capture_keys and isinstance(result, dict):
                output_payload = {k: result.get(k) for k in output_capture_keys}
            self.complete_agent_run(handle, output_payload=output_payload, status="completed")
        except Exception as exc:  # noqa: BLE001 - we need to persist failures
            self.complete_agent_run(
                handle,
                error_payload={"error": str(exc), "type": exc.__class__.__name__},
                status="failed",
            )
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session row as a dict, or None if missing."""
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?",
                (sid,),
            ).fetchone()
            if row is None:
                return None
            d = {k: row[k] for k in row.keys()}
            for key in ("resolved_summary_json", "ambiguity_questions_json"):
                raw = d.get(key)
                if isinstance(raw, str) and raw.strip():
                    try:
                        d[key.removesuffix("_json")] = json.loads(raw)
                    except json.JSONDecodeError:
                        d[key.removesuffix("_json")] = None
                else:
                    d[key.removesuffix("_json")] = None
            return d

    def insert_code_validation(
        self,
        *,
        session_id: str,
        branch: str,
        results: Any,
        status_messages: Any,
        git_pull_output: Optional[str] = None,
        git_diff_output: Optional[str] = None,
    ) -> str:
        validation_id = str(uuid.uuid4())
        now = _ist_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO code_validations(
                  validation_id,session_id,branch,results_json,status_messages_json,git_pull_output,git_diff_output,created_at
                )
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    validation_id,
                    session_id,
                    branch,
                    _json_dumps_small(results) or "{}",
                    _json_dumps_small(status_messages) or "{}",
                    git_pull_output,
                    git_diff_output,
                    now,
                ),
            )
        return validation_id

