# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Task 1 — EASY: Budget Survival

Objective:
  Keep the budget above zero for at least 26 consecutive weeks (half a year)
  while fulfilling some demand.

Grader:
  score = weeks_survived / 26  (capped at 1.0)
  +0.1 bonus if average weekly service level ≥ 0.50

Score range: 0.0 – 1.0

Difficulty: EASY
  An agent that orders nothing survives perfectly but earns no revenue.
  A random agent may overspend early. A safe conservative agent orders
  only the cheapest SKUs in small quantities and survives.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.DeepMatrix_environment import DeepmatrixEnvironment
from models import DeepmatrixAction

TASK_NAME = "budget_survival"
TASK_DIFFICULTY = "easy"
MAX_WEEKS = 26
TARGET_WEEKS = 26


def run_task(agent_fn, seed: int = 42) -> dict:
    """
    Run the budget-survival task.

    Parameters
    ----------
    agent_fn : callable
        Takes a DeepmatrixObservation, returns a DeepmatrixAction.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: score (float 0–1), weeks_survived, avg_service_level,
                    final_budget, metadata.
    """
    env = DeepmatrixEnvironment(max_weeks=MAX_WEEKS, seed=seed)
    obs = env.reset()

    weeks_survived = 0
    service_levels: list[float] = []

    for _ in range(MAX_WEEKS):
        action = agent_fn(obs)
        obs = env.step(action)
        weeks_survived += 1
        service_levels.append(obs.cumulative_service_level)

        if obs.done:
            break

    avg_svc = service_levels[-1] if service_levels else 0.0
    base_score = min(weeks_survived / TARGET_WEEKS, 0.95)
    bonus = 0.04 if avg_svc >= 0.50 else 0.0
    score = max(min(base_score + bonus, 0.999), 0.001)

    return {
        "task": TASK_NAME,
        "difficulty": TASK_DIFFICULTY,
        "score": round(score, 4),
        "weeks_survived": weeks_survived,
        "avg_service_level": round(avg_svc, 4),
        "final_budget": round(obs.budget, 2),
        "metadata": {
            "target_weeks": TARGET_WEEKS,
            "bonus_threshold_service_level": 0.50,
            "bonus_awarded": bonus > 0,
        },
    }


# --------------------------------------------------------------------------- #
# Default agent for testing                                                    #
# --------------------------------------------------------------------------- #
def conservative_agent(obs) -> DeepmatrixAction:
    """
    Conservative baseline: order only enough to cover 1 week of mean demand
    for the three cheapest SKUs, but only if inventory is very low.
    """
    import numpy as np
    from server.DeepMatrix_environment import DEMAND_MEAN, N, BASE_PRICES

    order = [0] * N
    budget_per_sku = obs.budget * 0.05  # never spend more than 5% per SKU

    # Sort SKUs by price ascending
    price_order = sorted(range(N), key=lambda i: obs.buying_price[i])
    for i in price_order[:3]:
        if obs.inventory[i] + obs.in_transit[i] < int(DEMAND_MEAN[i] * 1.5):
            affordable = int(budget_per_sku // obs.buying_price[i]) if obs.buying_price[i] > 0 else 0
            order[i] = min(affordable, int(DEMAND_MEAN[i]))

    return DeepmatrixAction(items_to_buy=order)


if __name__ == "__main__":
    result = run_task(conservative_agent)
    print(f"Task: {result['task']} | Difficulty: {result['difficulty']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Weeks survived: {result['weeks_survived']}/{TARGET_WEEKS}")
    print(f"Avg service level: {result['avg_service_level']:.4f}")
    print(f"Final budget: ${result['final_budget']:,.2f}")
