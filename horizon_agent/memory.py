"""Durable, provenance-aware memory. Only audited handoffs are stored."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class Memory:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS facts (task_id TEXT, stage TEXT, fact TEXT, source TEXT, confidence REAL, PRIMARY KEY(task_id, stage, fact, source))")
        self.db.execute("CREATE TABLE IF NOT EXISTS handoffs (task_id TEXT, stage TEXT, body TEXT, PRIMARY KEY(task_id, stage))")
        self.db.commit()

    def put_handoff(self, task_id: str, stage: str, body: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO handoffs VALUES (?, ?, ?)", (task_id, stage, json.dumps(body)))
        self.db.commit()

    def get_handoffs(self, task_id: str) -> list[dict]:
        rows = self.db.execute("SELECT body FROM handoffs WHERE task_id=? ORDER BY rowid", (task_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put_fact(self, task_id: str, stage: str, fact: str, source: str, confidence: float) -> None:
        if not source or not 0 <= confidence <= 1:
            raise ValueError("Fact requires a source and confidence in [0,1]")
        self.db.execute("INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?, ?)",
                        (task_id, stage, fact[:1000], source[:500], confidence))
        self.db.commit()

    def facts(self, task_id: str, limit: int = 30) -> list[dict]:
        rows = self.db.execute("SELECT stage, fact, source, confidence FROM facts WHERE task_id=? ORDER BY confidence DESC LIMIT ?", (task_id, limit)).fetchall()
        return [dict(zip(("stage", "fact", "source", "confidence"), row)) for row in rows]

    def close(self) -> None:
        self.db.close()
