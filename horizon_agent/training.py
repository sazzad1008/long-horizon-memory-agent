"""Outcome-gated rollout scoring, stage credit, and LoRA supervised training."""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path

from .evaluate import digest, evaluate


def _replay_score(trace: list[dict], judge_path: Path, excluded_stage: int | None = None) -> dict:
    """Score artifacts rebuilt from the trace; optionally ablate one stage's writes."""
    judge_hash = digest(judge_path)
    with tempfile.TemporaryDirectory(prefix="horizon-replay-") as directory:
        workspace = Path(directory)
        for event in trace:
            if event.get("role") != "tool" or event.get("action", {}).get("tool") != "write_artifact" or not event.get("result", {}).get("ok"):
                continue
            if event.get("stage") == excluded_stage:
                continue
            action = event["action"]
            path = (workspace / str(action["path"])).resolve()
            if workspace.resolve() not in path.parents:
                raise ValueError("Unsafe replay path")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(action["content"]), encoding="utf-8")
        return evaluate(workspace, judge_path, judge_hash)


def stage_credit(trace: list[dict], judge_path: Path, gamma: float = 0.9) -> dict[int, float]:
    """Ablation credit plus a discounted outcome signal.

    Ablation asks how much the final verified score drops if one stage's writes
    are removed while later actions stay fixed. This is a useful diagnostic,
    not a true counterfactual policy rollout.
    """
    stages = sorted({int(event["stage"]) for event in trace if event.get("role") == "handoff"
                     and isinstance(event.get("stage"), int)})
    full = _replay_score(trace, judge_path)
    terminal = 1.0 if full["success"] else 0.0
    return {stage: max(0.0, full["score"] - _replay_score(trace, judge_path, stage)["score"])
                   + (gamma ** (len(stages) - index - 1)) * terminal
            for index, stage in enumerate(stages)}


def prepare(rollouts: Path, judge_dir: Path, output: Path) -> dict:
    """Use only externally verified successful rollouts; retain stage credit metadata."""
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rollouts": 0, "accepted": 0, "examples": 0}
    with rollouts.open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            counts["rollouts"] += 1
            episode = json.loads(line)
            task_id = str(episode["task_id"])
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", task_id):
                continue
            judge_path = judge_dir / f"{task_id}.json"
            if not judge_path.is_file() or not episode.get("judge_hash") or digest(judge_path) != episode["judge_hash"]:
                continue
            with tempfile.TemporaryDirectory(prefix="horizon-verify-") as directory:
                workspace = Path(directory)
                for event in episode["trace"]:
                    if event.get("role") == "tool" and event.get("action", {}).get("tool") == "write_artifact" and event.get("result", {}).get("ok"):
                        action = event["action"]
                        path = (workspace / str(action["path"])).resolve()
                        if workspace.resolve() not in path.parents:
                            raise ValueError("Unsafe replay path")
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(str(action["content"]), encoding="utf-8")
                verdict = evaluate(workspace, judge_path, episode["judge_hash"])
            if not verdict["success"]:
                continue
            counts["accepted"] += 1
            credit = stage_credit(episode["trace"], judge_path)
            for event in episode["trace"]:
                if event.get("role") not in {"planner", "worker", "auditor"}:
                    continue
                stage = int(event["stage"])
                if credit.get(stage, 0.0) <= 0:
                    continue
                target.write(json.dumps({"prompt": event["prompt"], "completion": event["response"],
                                         "stage_credit": credit[stage], "role": event["role"]},
                                        ensure_ascii=False) + "\n")
                counts["examples"] += 1
    return counts


def train(data_path: Path, model_id: str, output: Path, steps: int) -> None:
    from datasets import load_dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = load_dataset("json", data_files=str(data_path), split="train")
    if len(dataset) == 0:
        raise ValueError("No verified examples")
    # Integer replication gives higher-credit stages more training weight.
    # It is a simple weighted SFT approximation, not policy-gradient RL.
    from datasets import Dataset
    weighted = []
    for row in dataset:
        copies = 1 + min(3, int(float(row["stage_credit"]) * 2))
        weighted.extend({"prompt": row["prompt"], "completion": row["completion"]} for _ in range(copies))
    dataset = Dataset.from_list(weighted)
    config = SFTConfig(output_dir=str(output), max_steps=steps, max_length=2048,
                       per_device_train_batch_size=1, gradient_accumulation_steps=4,
                       learning_rate=2e-4, completion_only_loss=True, save_strategy="no", report_to="none")
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules="all-linear")
    trainer = SFTTrainer(model=model_id, args=config, train_dataset=dataset, peft_config=lora)
    trainer.train()
    trainer.save_model(str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("rollouts", type=Path)
    prep.add_argument("judge_dir", type=Path)
    prep.add_argument("output", type=Path)
    fit = sub.add_parser("train")
    fit.add_argument("data", type=Path)
    fit.add_argument("--model", required=True)
    fit.add_argument("--output", type=Path, default=Path("checkpoints/horizon-lora"))
    fit.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.rollouts, args.judge_dir, args.output))
    else:
        train(args.data, args.model, args.output, args.steps)


if __name__ == "__main__":
    main()
