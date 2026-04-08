# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Task 2 — MEDIUM: Service-Level Optimiser

Objective:
  Achieve and maintain a cumulative fill-rate (service level) ≥ 0.80
  over a full 52-week year while keeping waste below 15% of total demand.

Grader (partial-credit scoring):
  service_score = clamp(cum_service_level / 0.80, 0, 1)       weight 0.60
  waste_score   = clamp(1 - waste_ratio / 0.15, 0, 1)         weight 0.25
  survival_bonus = 0.15 if budget > 0 at end of episode        weight 0.15

  score = 0.60 * service_score + 0.25 * waste_score + 0.15 * survival_bonus

Score range: 0.0 – 1.0

Difficulty: MEDIUM
  Requires the agent to balance stocking up (to fill demand) against
  over-ordering (which causes waste and budget drain).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.DeepMatrix_environment import (
    DeepmatrixEnvironment, DEMAND_MEAN, LEAD_TIMES,
    SERVICE_LEVEL_Z, N
)
from models import DeepmatrixAction

TASK_NAME = "service_level_optimizer"
TASK_DIFFICULTY = "medium"
MAX_WEEKS = 52
TARGET_SERVICE_LEVEL = 0.80
WASTE_BUDGET_RATIO = 0.15   # max acceptable waste / total demand


def run_task(agent_fn, seed: int = 42) -> dict:
    """
    Run the service-level optimiser task.

    Parameters
    ----------
    agent_fn : callable
        Takes a DeepmatrixObservation, returns a DeepmatrixAction.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: score, service_level, waste_ratio, survived,
                    weeks_run, cumulative_profit, metadata.
    """
    env = DeepmatrixEnvironment(max_weeks=MAX_WEEKS, seed=seed)
    obs = env.reset()

    weeks_run = 0
    for _ in range(MAX_WEEKS):
        action = agent_fn(obs)
        obs = env.step(action)
        weeks_run += 1
        if obs.done:
            break

    cum_svc = obs.cumulative_service_level
    total_demand_approx = max(obs.cumulative_waste_units + 1, 1)
    waste_ratio = obs.cumulative_waste_units / max(
        sum(int(d * weeks_run) for d in DEMAND_MEAN), 1
    )

    service_score = min(cum_svc / TARGET_SERVICE_LEVEL, 1.0)
    waste_score = max(0.0, min(1.0 - waste_ratio / WASTE_BUDGET_RATIO, 1.0))
    survival_bonus = 1.0 if obs.budget > 0 else 0.0

    score = 0.60 * service_score + 0.25 * waste_score + 0.15 * survival_bonus

    return {
        "task": TASK_NAME,
        "difficulty": TASK_DIFFICULTY,
        "score": round(score, 4),
        "service_level": round(cum_svc, 4),
        "waste_ratio": round(waste_ratio, 4),
        "waste_units": obs.cumulative_waste_units,
        "survived": obs.budget > 0,
        "weeks_run": weeks_run,
        "final_budget": round(obs.budget, 2),
        "cumulative_profit": round(obs.cumulative_profit, 2),
        "metadata": {
            "target_service_level": TARGET_SERVICE_LEVEL,
            "waste_budget_ratio": WASTE_BUDGET_RATIO,
            "weights": {"service": 0.60, "waste": 0.25, "survival": 0.15},
            "component_scores": {
                "service_score": round(service_score, 4),
                "waste_score": round(waste_score, 4),
                "survival_bonus": survival_bonus,
            },
        },
    }


# --------------------------------------------------------------------------- #
# Safety-stock agent (safety-stock formula)                                    #
# --------------------------------------------------------------------------- #
def safety_stock_agent(obs) -> DeepmatrixAction:
    """
    Implements q* = max(0, F_{t+L} + z*sigma_{t+L} - I_t - T_t).
    Never orders more than budget / (N * price) per SKU.
    """
    import numpy as np

    order = []
    per_sku_budget = obs.budget / (N * 2)  # split budget conservatively

    for i in range(N):
        q_star = max(
            0,
            int(
                obs.demand_forecast[i]
                + SERVICE_LEVEL_Z * obs.demand_forecast_std[i]
                - obs.inventory[i]
                - obs.in_transit[i]
            ),
        )
        price = obs.buying_price[i]
        if price > 0:
            q_affordable = int(per_sku_budget // price)
        else:
            q_affordable = q_star
        order.append(min(q_star, q_affordable))

    return DeepmatrixAction(items_to_buy=order)


if __name__ == "__main__":
    result = run_task(safety_stock_agent)
    print(f"Task: {result['task']} | Difficulty: {result['difficulty']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Service level: {result['service_level']:.4f} (target ≥ {TARGET_SERVICE_LEVEL})")
    print(f"Waste ratio: {result['waste_ratio']:.4f} (budget ≤ {WASTE_BUDGET_RATIO})")
    print(f"Survived: {result['survived']} | Weeks run: {result['weeks_run']}")
    print(f"Cumulative profit: ${result['cumulative_profit']:,.2f}")
    print(f"Component scores: {result['metadata']['component_scores']}")
