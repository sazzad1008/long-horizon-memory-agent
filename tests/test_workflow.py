import json
from pathlib import Path

import pytest

from horizon_agent.agent import run
from horizon_agent.environment import Environment
from horizon_agent.evaluate import digest, evaluate
from horizon_agent.memory import Memory
from horizon_agent.training import prepare, stage_credit


class Response:
    def __init__(self, content):
        self.content = content


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return Response(json.dumps(next(self.responses)))


def fixture(tmp_path):
    sources = tmp_path / "sources"
    workspace = tmp_path / "workspace"
    sources.mkdir()
    (sources / "note.txt").write_text("Fact A from note.\n")
    judge = tmp_path / "judge.json"
    judge.write_text(json.dumps({"gates": {"one": {"artifact": "one.md", "contains": ["Fact A"]},
                                           "two": {"artifact": "two.md", "contains": ["Fact A", "done"]}}}))
    task = {"id": "task-1", "goal": "Produce two grounded artifacts",
            "stages": [{"id": "one", "objective": "Write one.md"}, {"id": "two", "objective": "Write two.md"}]}
    return sources, workspace, judge, task


def test_handoffs_memory_resume_and_credit(tmp_path):
    sources, workspace, judge, task = fixture(tmp_path)
    model = FakeModel([
        {"plan": "Read note and write one", "success_criteria": "one.md"},
        {"tool": "read_source", "path": "note.txt"},
        {"tool": "write_artifact", "path": "one.md", "content": "Fact A (note.txt)"},
        {"tool": "finish", "summary": "one done"},
        {"summary": "Fact A found; one.md written", "evidence": ["note.txt"], "open_issues": [],
         "facts": [{"fact": "Fact A", "source": "note.txt", "confidence": 0.9}]},
        {"plan": "Use handoff", "success_criteria": "two.md"},
        {"tool": "write_artifact", "path": "two.md", "content": "Fact A (note.txt); done"},
        {"tool": "finish", "summary": "two done"},
        {"summary": "Second artifact done", "evidence": [], "open_issues": [], "facts": []},
    ])
    memory_path = tmp_path / "memory.sqlite"
    result = run(task, workspace, sources, memory_path, "unused", max_stage_steps=5, model=model)
    assert evaluate(workspace, judge)["success"]
    assert "Fact A found" in model.prompts[5]
    assert len(result["handoffs"]) == 2
    with DummyContext(Memory(memory_path)) as memory:
        assert memory.facts("task-1")[0]["fact"] == "Fact A"
    assert run(task, workspace, sources, memory_path, "unused", model=FakeModel([]))["answer"] == "Already completed all stages"
    credit = stage_credit(result["trace"], judge)
    assert credit[0] > 0 and credit[1] > 0
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps({"task_id": "task-1", "trace": result["trace"],
                                    "judge_hash": digest(judge)}) + "\n")
    judge_dir = tmp_path / "judges"
    judge_dir.mkdir()
    (judge_dir / "task-1.json").write_bytes(judge.read_bytes())
    output = tmp_path / "train.jsonl"
    counts = prepare(rollouts, judge_dir, output)
    assert counts["accepted"] == 1 and counts["examples"] > 0


class DummyContext:
    def __init__(self, memory):
        self.memory = memory
    def __enter__(self):
        return self.memory
    def __exit__(self, *args):
        self.memory.close()


def test_environment_rejects_escape(tmp_path):
    sources, workspace, judge, task = fixture(tmp_path)
    env = Environment(workspace, sources)
    with pytest.raises(ValueError):
        env.act({"tool": "write_artifact", "path": "../judge.json", "content": "tamper"})
    with pytest.raises(ValueError):
        env.act({"tool": "read_source", "path": "../judge.json"})
    assert env.act({"tool": "read_source", "path": "note.txt"})["ok"]


def test_judge_tamper_gate(tmp_path):
    sources, workspace, judge, task = fixture(tmp_path)
    workspace.mkdir()
    original = digest(judge)
    judge.write_text(judge.read_text() + " ")
    assert evaluate(workspace, judge, original)["reason"] == "Judge changed during run"


def test_training_rejects_unverified_rollout(tmp_path):
    sources, workspace, judge, task = fixture(tmp_path)
    judges = tmp_path / "judges"
    judges.mkdir()
    (judges / "task-1.json").write_bytes(judge.read_bytes())
    rollouts = tmp_path / "rollouts.jsonl"
    rollouts.write_text(json.dumps({"task_id": "task-1", "trace": [], "judge_hash": digest(judge)}) + "\n")
    assert prepare(rollouts, judges, tmp_path / "train.jsonl")["accepted"] == 0
