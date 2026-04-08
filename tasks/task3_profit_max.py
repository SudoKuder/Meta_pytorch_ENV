# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Task 3 — HARD: Profit Maximisation Under Operational Constraints

Objective:
  Maximise total profit over 52 weeks while simultaneously satisfying
  three operational constraints:
    1. Cumulative fill-rate (service level) ≥ 0.85
    2. Waste ratio (waste units / total demand)  ≤ 0.10
    3. End-of-year budget ≥ 25% of INITIAL_BUDGET (= $12,500)

Grader (partial-credit scoring with hard constraint penalties):
  profit_score     = sigmoid normalisation of cumulative_profit
                     benchmark_profit = $80,000 (aggressive agent)
                     floor_profit     = $0
  constraint_1_ok  = 1.0 if fill_rate ≥ 0.85 else fill_rate / 0.85
  constraint_2_ok  = 1.0 if waste_ratio ≤ 0.10 else max(0, 1 - (waste_ratio - 0.10) / 0.10)
  constraint_3_ok  = 1.0 if final_budget ≥ 12500 else final_budget / 12500

  constraint_factor = (constraint_1_ok * constraint_2_ok * constraint_3_ok) ** (1/3)
  score = profit_score * constraint_factor

  Partial credit:
    - Profit alone with 0 constraints satisfied → ≤ 0.30 score.
    - All constraints satisfied but zero profit → 0.0 score.
    - Both required to approach 1.0.

Score range: 0.0 – 1.0

Difficulty: HARD
  Requires intelligent trade-offs: ordering too little hurts service level;
  ordering too much wastes budget and creates perishable waste; holding
  excess stock erodes profit via holding costs.  Bulk discounts are real
  incentives but can over-commit capital to slow-moving SKUs.
"""
from __future__ import annotations

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.DeepMatrix_environment import (
    DeepmatrixEnvironment, DEMAND_MEAN, LEAD_TIMES,
    SERVICE_LEVEL_Z, N, INITIAL_BUDGET
)
from models import DeepmatrixAction

TASK_NAME = "profit_maximization"
TASK_DIFFICULTY = "hard"
MAX_WEEKS = 52

# Constraint thresholds
MIN_SERVICE_LEVEL = 0.85
MAX_WASTE_RATIO = 0.10
MIN_END_BUDGET_FRACTION = 0.25
MIN_END_BUDGET = INITIAL_BUDGET * MIN_END_BUDGET_FRACTION

# Profit normalisation anchors (empirically estimated)
BENCHMARK_PROFIT = 80_000.0
FLOOR_PROFIT = 0.0


def _sigmoid_profit_score(profit: float) -> float:
    """Map profit to [0, 1] using a shifted sigmoid centred on benchmark."""
    if profit <= FLOOR_PROFIT:
        return 0.0
    if profit >= BENCHMARK_PROFIT:
        return 1.0
    return profit / BENCHMARK_PROFIT


def run_task(agent_fn, seed: int = 42) -> dict:
    """
    Run the profit-maximisation task.

    Parameters
    ----------
    agent_fn : callable
        Takes a DeepmatrixObservation, returns a DeepmatrixAction.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: score (0–1), cumulative_profit, service_level,
                    waste_ratio, final_budget, weeks_run, constraints_met,
                    metadata.
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

    # ---- Compute metrics ---- #
    cum_profit = obs.cumulative_profit
    service_level = obs.cumulative_service_level
    # Approximate total demand over episode
    total_demand_est = sum(DEMAND_MEAN) * weeks_run
    waste_ratio = obs.cumulative_waste_units / max(total_demand_est, 1)
    final_budget = obs.budget

    # ---- Partial constraint scores ---- #
    c1 = 1.0 if service_level >= MIN_SERVICE_LEVEL else service_level / MIN_SERVICE_LEVEL
    c2 = (
        1.0
        if waste_ratio <= MAX_WASTE_RATIO
        else max(0.0, 1.0 - (waste_ratio - MAX_WASTE_RATIO) / MAX_WASTE_RATIO)
    )
    c3 = 1.0 if final_budget >= MIN_END_BUDGET else final_budget / MIN_END_BUDGET

    constraint_factor = (c1 * c2 * c3) ** (1 / 3)
    profit_score = _sigmoid_profit_score(cum_profit)
    score = profit_score * constraint_factor

    constraints_met = {
        "service_level": service_level >= MIN_SERVICE_LEVEL,
        "waste_ratio": waste_ratio <= MAX_WASTE_RATIO,
        "end_budget": final_budget >= MIN_END_BUDGET,
    }

    return {
        "task": TASK_NAME,
        "difficulty": TASK_DIFFICULTY,
        "score": round(score, 4),
        "cumulative_profit": round(cum_profit, 2),
        "service_level": round(service_level, 4),
        "waste_ratio": round(waste_ratio, 4),
        "waste_units": obs.cumulative_waste_units,
        "final_budget": round(final_budget, 2),
        "weeks_run": weeks_run,
        "constraints_met": constraints_met,
        "all_constraints_met": all(constraints_met.values()),
        "metadata": {
            "thresholds": {
                "min_service_level": MIN_SERVICE_LEVEL,
                "max_waste_ratio": MAX_WASTE_RATIO,
                "min_end_budget": MIN_END_BUDGET,
            },
            "component_scores": {
                "profit_score": round(profit_score, 4),
                "c1_service": round(c1, 4),
                "c2_waste": round(c2, 4),
                "c3_budget": round(c3, 4),
                "constraint_factor": round(constraint_factor, 4),
            },
        },
    }


# --------------------------------------------------------------------------- #
# Adaptive safety-stock agent (uses bulk discount when profitable)             #
# --------------------------------------------------------------------------- #
def adaptive_agent(obs) -> DeepmatrixAction:
    """
    Adaptive agent:
      - Uses safety-stock formula with dynamic z (increases when service low)
      - Exploits bulk discount (orders ≥100) when margin is positive
      - Reserves MIN_END_BUDGET_FRACTION of budget as a floor
    """
    BUDGET_RESERVE = INITIAL_BUDGET * MIN_END_BUDGET_FRACTION
    spendable = max(obs.budget - BUDGET_RESERVE, 0.0)
    if spendable <= 0:
        return DeepmatrixAction(items_to_buy=[0] * N)

    per_sku_budget = spendable / N

    # Dynamically increase z when service level is lagging
    svc = obs.cumulative_service_level
    z = SERVICE_LEVEL_Z if svc >= MIN_SERVICE_LEVEL else SERVICE_LEVEL_Z * 1.3

    order = []
    for i in range(N):
        q_star = max(
            0,
            int(
                obs.demand_forecast[i]
                + z * obs.demand_forecast_std[i]
                - obs.inventory[i]
                - obs.in_transit[i]
            ),
        )
        price = obs.buying_price[i]
        if price <= 0:
            order.append(0)
            continue

        q_affordable = int(per_sku_budget // price)
        q = min(q_star, q_affordable)

        # Try bulk discount if it saves money and we can afford it
        if q < 100 and q > 0:
            q_bulk = 100
            bulk_price = price * 0.92
            bulk_cost = q_bulk * bulk_price
            normal_cost = q * price
            # Only upgrade if we can afford bulk and saving per unit is worth it
            if bulk_cost <= per_sku_budget and bulk_price * q_bulk < price * q_bulk:
                # Check if we'd waste the extra units
                excess = q_bulk - q_star
                if excess < 5:  # tolerate small excess
                    q = q_bulk

        order.append(max(0, q))

    return DeepmatrixAction(items_to_buy=order)


if __name__ == "__main__":
    result = run_task(adaptive_agent)
    print(f"Task: {result['task']} | Difficulty: {result['difficulty']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Cumulative profit: ${result['cumulative_profit']:,.2f}")
    print(f"Service level: {result['service_level']:.4f} (target ≥ {MIN_SERVICE_LEVEL})")
    print(f"Waste ratio: {result['waste_ratio']:.4f} (limit ≤ {MAX_WASTE_RATIO})")
    print(f"Final budget: ${result['final_budget']:,.2f} (min ${MIN_END_BUDGET:,.0f})")
    print(f"All constraints met: {result['all_constraints_met']}")
    print(f"Component scores: {result['metadata']['component_scores']}")
