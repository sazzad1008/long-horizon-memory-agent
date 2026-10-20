# Long-horizon memory agent

A separate multi-agent workflow for tasks that take several stages and must carry evidence forward. The example task produces an incident timeline, cause analysis, and final brief from local source documents. A planner, worker, and auditor collaborate through explicit handoffs; SQLite stores summaries and source-backed facts for later stages and resumed runs.

```text
Task manifest → stage planner → worker/tool loop → auditor handoff → next stage
                                          ↓                  ↓
                                    artifacts          SQLite memory
                                          ↓
                    independent hidden evaluator → stage ablation credit
                                          ↓
                              verified examples → LoRA SFT
```

## Architecture

- **Long horizon:** A task manifest has ordered stages. Each stage has a bounded tool loop. The next planner receives the previous summary, recent handoffs, and durable memory. A run resumes from saved stage handoffs.
- **Agent-to-agent context:** Handoffs have a summary, evidence paths, source-backed facts with confidence, and open issues. The auditor can only store facts whose sources the worker actually read in that stage.
- **Memory:** SQLite persists handoffs and facts by task ID. Memory is kept outside the agent workspace. Summaries are deliberately short; the full trace is retained separately for audit/training.
- **Tools:** The worker can list and read local sources, and read/write artifacts. It cannot run shell commands or access the judge through the tool API. Paths are confined to source and artifact directories.
- **Independent reward:** The evaluator is separate from the model and checks final artifacts against a judge manifest outside the workspace. A hash detects changes to the judge during a run. A task only succeeds when all gates pass.
- **Credit assignment:** Training replays the artifact-writing trace. For each stage, it removes that stage's writes and measures the drop in final verified score, plus a discounted terminal-success signal. This is an ablation heuristic; later policy behavior is held fixed, so it is not a causal estimate.
- **Training:** Only fully verified trajectories enter the dataset. Stage credit controls example replication for LoRA supervised fine-tuning. This is reward-weighted SFT, not online RL or GRPO.

## Install

Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
export OPENAI_API_KEY='your-key'
```

`HORIZON_MODEL` selects the chat model (default `gpt-4.1-mini`). `OPENAI_BASE_URL` can point to an OpenAI-compatible server. The sample task is in `examples/task.json`, source document in `examples/sources`, and independent judge in `examples/judges-incident-001.json`.

## Run a task

From this repository root:

```bash
mkdir -p runs/incident-workspace runs/judges
cp examples/judges-incident-001.json runs/judges/incident-001.json
horizon-agent examples/task.json \
  --sources examples/sources \
  --workspace runs/incident-workspace \
  --memory runs/memory.sqlite \
  --judge runs/judges/incident-001.json \
  --rollout runs/rollouts.jsonl
```

The CLI prints the outcome verdict and stores the observable trace. Run the same command to resume from stored handoffs. To start over, use a new task ID, memory database, and workspace. The judge and memory must remain outside the agent workspace. Keep rollouts private: they contain source text and model outputs.

## Prepare data and train

```bash
python -m horizon_agent.training prepare \
  runs/rollouts.jsonl runs/judges runs/train.jsonl
pip install -e '.[train]'
python -m horizon_agent.training train runs/train.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --output checkpoints/horizon-lora \
  --steps 100
```

Training requires suitable accelerator memory and many varied, independently judged tasks. The output is a PEFT adapter. Serve it with the base model through a compatible chat endpoint, then compare held-out task success, false-evidence rate, memory usefulness, and cost against the base model. The example is a pipeline smoke test, not evidence of generalization.

## Tests

```bash
pytest -q
```

The tests cover multi-stage handoffs, summary transfer, durable memory, resume, path protection, evaluator tamper detection, credit calculation, and verified-data gating. GitHub Actions runs the offline tests.

## Limits and reward-hacking risk

The included judge uses simple required-string gates to keep the example easy to run. A capable model could satisfy those strings without a sound analysis. For real tasks, use private held-out checks that test factual support and task outcome, audit cited evidence, vary task templates, and monitor performance on unseen cases. Hashing the judge detects modification, but it cannot prevent a weak judge from being gamed. Stage ablation gives useful training weights, but it does not fully solve long-horizon credit assignment.
