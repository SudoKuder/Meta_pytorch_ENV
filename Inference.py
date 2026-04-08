"""Deterministic baseline evaluation for DeepMatrix contest tasks.

This script runs one baseline agent per task across fixed seeds and writes a
reproducible score report that can be attached to submissions.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

# Ensure local task modules can be imported when run as a script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks.task1_budget_survival import conservative_agent, run_task as run_task1
from tasks.task2_service_level import safety_stock_agent, run_task as run_task2
from tasks.task3_profit_max import adaptive_agent, run_task as run_task3


def parse_seeds(raw: str) -> list[int]:
    """Parse comma-separated seeds while preserving order."""
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds or [42, 123, 999]


def evaluate_task(task_name: str, runner, agent_fn, seeds: list[int]) -> dict[str, Any]:
    """Run one task baseline across all seeds and summarize stable metrics."""
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        result = runner(agent_fn, seed=seed)
        runs.append(result)

    mean_score = mean(r["score"] for r in runs)
    mean_budget = mean(r.get("final_budget", 0.0) for r in runs)

    return {
        "task": task_name,
        "seeds": seeds,
        "mean_score": round(mean_score, 4),
        "mean_final_budget": round(mean_budget, 2),
        "runs": runs,
    }


def main() -> int:
    seeds = parse_seeds(os.getenv("DEEPMATRIX_BASELINE_SEEDS", "42,123,999"))
    output_file = os.getenv("DEEPMATRIX_BASELINE_OUTPUT", "baseline_scores.json")

    report: dict[str, Any] = {
        "benchmark": "DeepMatrix",
        "reproducible": True,
        "seeds": seeds,
        "tasks": {
            "easy": evaluate_task("task1_budget_survival", run_task1, conservative_agent, seeds),
            "medium": evaluate_task("task2_service_level", run_task2, safety_stock_agent, seeds),
            "hard": evaluate_task("task3_profit_max", run_task3, adaptive_agent, seeds),
        },
    }

    scores = [
        report["tasks"]["easy"]["mean_score"],
        report["tasks"]["medium"]["mean_score"],
        report["tasks"]["hard"]["mean_score"],
    ]
    report["overall_mean_score"] = round(mean(scores), 4)

    out_path = ROOT / output_file
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DeepMatrix baseline evaluation complete")
    print(f"Seeds: {seeds}")
    print(f"Easy mean score:   {report['tasks']['easy']['mean_score']:.4f}")
    print(f"Medium mean score: {report['tasks']['medium']['mean_score']:.4f}")
    print(f"Hard mean score:   {report['tasks']['hard']['mean_score']:.4f}")
    print(f"Overall mean:      {report['overall_mean_score']:.4f}")
    print(f"Wrote report:      {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
