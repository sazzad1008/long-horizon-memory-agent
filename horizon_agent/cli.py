"""Run a long-horizon task; keep judge files outside the agent workspace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import run
from .evaluate import digest, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--model", default=os.getenv("HORIZON_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--stage-steps", type=int, default=10)
    args = parser.parse_args()
    if args.stage_steps < 1:
        parser.error("--stage-steps must be positive")
    workspace_root = args.workspace.resolve()
    sources_root = args.sources.resolve()
    for protected in (args.judge, args.task, args.memory):
        resolved = protected.resolve()
        if any(resolved == root or root in resolved.parents for root in (workspace_root, sources_root)):
            parser.error("task, judge, and memory must be outside agent workspace and sources")
    task = json.loads(args.task.read_text(encoding="utf-8"))
    judge_hash = digest(args.judge)
    result = run(task, args.workspace, args.sources, args.memory, args.model, args.stage_steps)
    verdict = evaluate(args.workspace, args.judge, judge_hash)
    print(result["answer"])
    print(json.dumps(verdict))
    args.rollout.parent.mkdir(parents=True, exist_ok=True)
    with args.rollout.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"task_id": task["id"], "trace": result["trace"],
                               "handoffs": result["handoffs"], "verdict": verdict,
                               "judge_hash": judge_hash}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
