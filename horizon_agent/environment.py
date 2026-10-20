"""Artifact-only environment: no shell, network, or access to evaluator files."""
from __future__ import annotations

import hashlib
from pathlib import Path

MAX_FILE = 80_000


class Environment:
    def __init__(self, workspace: Path, sources: Path):
        self.workspace = workspace.resolve()
        self.sources = sources.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        if self.workspace == self.sources or self.workspace in self.sources.parents or self.sources in self.workspace.parents:
            raise ValueError("Sources and workspace must be separate directories")

    @staticmethod
    def _inside(root: Path, name: str) -> Path:
        path = (root / name).resolve()
        if path == root or root not in path.parents or any(part.startswith(".") for part in path.relative_to(root).parts):
            raise ValueError("Path outside allowed directory")
        return path

    def act(self, action: dict) -> dict:
        tool = action.get("tool")
        if tool == "list_sources":
            files = sorted(str(p.relative_to(self.sources)) for p in self.sources.rglob("*") if p.is_file())
            return {"ok": True, "output": "\n".join(files[:200])[:20_000]}
        if tool == "read_source":
            path = self._inside(self.sources, str(action["path"]))
            content = path.read_text(encoding="utf-8")[:20_000]
            return {"ok": True, "output": content,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if tool == "read_artifact":
            path = self._inside(self.workspace, str(action["path"]))
            return {"ok": True, "output": path.read_text(encoding="utf-8")[:20_000]}
        if tool == "write_artifact":
            path = self._inside(self.workspace, str(action["path"]))
            content = str(action["content"])
            if len(content) > MAX_FILE:
                raise ValueError("Artifact too large")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"ok": True, "output": f"Wrote {path.relative_to(self.workspace)}"}
        raise ValueError(f"Unknown tool: {tool}")
