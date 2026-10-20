"""Independent, deterministic outcome checks and tamper gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(workspace: Path, judge_path: Path, expected_hash: str | None = None) -> dict:
    if expected_hash and digest(judge_path) != expected_hash:
        return {"success": False, "score": 0.0, "gates": {}, "reason": "Judge changed during run"}
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    gates = {}
    for name, rule in judge["gates"].items():
        target = (workspace.resolve() / rule["artifact"]).resolve()
        if workspace.resolve() not in target.parents:
            raise ValueError("Judge artifact escapes workspace")
        content = target.read_text(encoding="utf-8") if target.is_file() else ""
        gates[name] = bool(content) and all(s in content for s in rule.get("contains", [])) and \
            all(s not in content for s in rule.get("excludes", []))
    success = bool(gates) and all(gates.values())
    return {"success": success, "score": sum(gates.values()) / len(gates) if gates else 0.0,
            "gates": gates, "reason": "all gates passed" if success else "missing or failed gate"}
