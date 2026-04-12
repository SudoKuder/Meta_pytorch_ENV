"""
Inference Script — DeepMatrix (Supply Chain Inventory Management)
===================================
MANDATORY environment variables:
    API_BASE_URL        LLM endpoint (default: https://router.huggingface.co/v1)
    MODEL_NAME          Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN            HuggingFace / API key
    SPACE_URL           DeepMatrix HF Space URL (default: https://sudokuder-deepmatrix.hf.space)
    DEEPMATRIX_TASK     Task to run: task1 | task2 | task3 (default: task1)

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, List, Optional

import requests
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
SPACE_URL    = os.getenv("SPACE_URL",    "https://sudokuder-deepmatrix.hf.space").rstrip("/")
TASK_KEY     = os.getenv("DEEPMATRIX_TASK", "task1")
BENCHMARK    = "deepmatrix"
TEMPERATURE  = 0.2
MAX_TOKENS   = 256

# SKU constants mirrored here — no local env import needed
N              = 10
LEAD_TIMES     = [2, 2, 3, 3, 4, 4, 2, 3, 2, 4]
SHELF_LIVES    = [4, 6, 3, 5, 8, 4, 6, 3, 5, 7]
BASE_PRICES    = [10.0, 12.0, 8.0, 15.0, 20.0, 9.0, 11.0, 14.0, 7.0, 18.0]
SELLING_PRICES = [round(p * 1.5, 2) for p in BASE_PRICES]
DEMAND_MEAN    = [20.0, 15.0, 30.0, 10.0, 8.0, 25.0, 18.0, 12.0, 35.0, 6.0]
INITIAL_BUDGET = 50_000.0
SERVICE_LEVEL_Z = 1.65

# Task registry: display name, max steps, success threshold
TASK_REGISTRY: dict[str, tuple[str, int, float]] = {
    "task1": ("budget_survival",         30,  0.80),
    "task2": ("service_level_optimizer", 50,  0.70),
    "task3": ("profit_maximization",     100, 0.50),
}
TASK_NAME, MAX_STEPS, SUCCESS_THRESHOLD = TASK_REGISTRY.get(
    TASK_KEY, TASK_REGISTRY["task1"]
)


# --------------------------------------------------------------------------- #
# HTTP client — talks to deployed HF Space                                    #
# --------------------------------------------------------------------------- #
class DeepMatrixHTTPEnv:
    """Thin HTTP wrapper around the deployed OpenEnv HF Space."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def reset(self) -> dict[str, Any]:
        resp = self._session.post(f"{self.base_url}/reset", json={}, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def step(self, items_to_buy: list[int]) -> dict[str, Any]:
        payload = {"action": {"items_to_buy": items_to_buy}}
        resp = self._session.post(f"{self.base_url}/step", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self._session.close()


def _get_obs(result: dict[str, Any]) -> dict[str, Any]:
    """Extract observation dict from reset/step response."""
    return result.get("observation", result)


def _get_field(obs: dict, key: str, default: Any) -> Any:
    return obs.get(key, default)


# --------------------------------------------------------------------------- #
# Stdout logging — mandatory format                                            #
# --------------------------------------------------------------------------- #
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: list,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val  = error if error else "null"
    done_val   = str(done).lower()
    action_str = json.dumps(action)
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# LLM prompts                                                                  #
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = textwrap.dedent(f"""
    You are an expert supply-chain inventory manager controlling {N} SKUs
    over a weekly horizon.

    Key environment facts:
      - Lead times per SKU (weeks):  {LEAD_TIMES}
      - Shelf lives per SKU (weeks): {SHELF_LIVES}
      - Base buying prices:          {BASE_PRICES}
      - Selling prices:              {SELLING_PRICES}
      - Bulk discount: 8% off any SKU where you order >= 100 units
      - Safety-stock formula: q* = max(0, forecast + {SERVICE_LEVEL_Z}*std - inventory - in_transit)
      - Starting budget: ${INITIAL_BUDGET:,.0f} — episode ends if budget hits 0

    Your goal: maximise cumulative profit while maintaining high service level
    and low waste.

    RESPONSE FORMAT — reply with ONLY a valid JSON object, no prose, no markdown:
    {{"items_to_buy": [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9]}}

    All values must be non-negative integers. The list must have exactly {N} elements.
""").strip()


def build_user_prompt(
    step: int,
    obs: dict[str, Any],
    last_reward: float,
    history: List[str],
) -> str:
    inventory    = _get_field(obs, "inventory",           [0] * N)
    in_transit   = _get_field(obs, "in_transit",          [0] * N)
    demand       = _get_field(obs, "demand",              [0] * N)
    waste        = _get_field(obs, "waste",               [0] * N)
    forecast     = _get_field(obs, "demand_forecast",     [0.0] * N)
    forecast_std = _get_field(obs, "demand_forecast_std", [0.0] * N)
    prices       = _get_field(obs, "buying_price",        BASE_PRICES)
    arrival      = _get_field(obs, "arrival_time",        [0] * N)
    expiry       = _get_field(obs, "expiry_time",         [0] * N)
    budget       = _get_field(obs, "budget",              0.0)

    recommendations = []
    for i in range(N):
        q_star = max(
            0,
            int(
                forecast[i]
                + SERVICE_LEVEL_Z * forecast_std[i]
                - inventory[i]
                - in_transit[i]
            ),
        )
        recommendations.append(q_star)

    history_block = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        Week: {step}
        Budget remaining:        ${budget:,.2f}
        Inventory (on-hand):     {inventory}
        In-transit:              {in_transit}
        Demand fulfilled (last): {demand}
        Waste this week:         {waste}
        Demand forecast:         {[round(f, 1) for f in forecast]}
        Forecast std:            {[round(s, 1) for s in forecast_std]}
        Buying prices (current): {[round(p, 2) for p in prices]}
        Arrival weeks:           {arrival}
        Expiry weeks:            {expiry}
        Last reward:             {last_reward:.2f}
        Safety-stock recs:       {recommendations}

        Recent history:
        {history_block}

        Decide your order quantities for all {N} SKUs this week.
        Reply ONLY with JSON: {{"items_to_buy": [q0..q9]}}
    """).strip()


# --------------------------------------------------------------------------- #
# LLM call + action parsing                                                    #
# --------------------------------------------------------------------------- #
def get_llm_action(
    client: OpenAI,
    step: int,
    obs: dict[str, Any],
    last_reward: float,
    history: List[str],
) -> list[int]:
    user_prompt = build_user_prompt(step, obs, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return _parse_action(text)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return _fallback_action(obs)


def _parse_action(text: str) -> list[int]:
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        data  = json.loads(text)
        items = data["items_to_buy"]
        if len(items) != N:
            raise ValueError(f"Expected {N} items, got {len(items)}")
        return [max(0, int(q)) for q in items]
    except Exception as exc:
        print(f"[DEBUG] Parse failed ({exc}), extracting raw integers.", flush=True)
        nums = re.findall(r"\d+", text)
        if len(nums) >= N:
            return [int(n) for n in nums[:N]]
        return [0] * N


def _fallback_action(obs: dict[str, Any]) -> list[int]:
    """Safety-stock heuristic when LLM fails."""
    budget       = _get_field(obs, "budget",              INITIAL_BUDGET)
    inventory    = _get_field(obs, "inventory",           [0] * N)
    in_transit   = _get_field(obs, "in_transit",          [0] * N)
    forecast     = _get_field(obs, "demand_forecast",     DEMAND_MEAN)
    forecast_std = _get_field(obs, "demand_forecast_std", [0.0] * N)
    prices       = _get_field(obs, "buying_price",        BASE_PRICES)

    per_sku_budget = budget / (N * 2)
    order = []
    for i in range(N):
        q_star = max(
            0,
            int(
                forecast[i]
                + SERVICE_LEVEL_Z * forecast_std[i]
                - inventory[i]
                - in_transit[i]
            ),
        )
        price        = prices[i] if prices[i] > 0 else BASE_PRICES[i]
        q_affordable = int(per_sku_budget // price)
        order.append(min(q_star, q_affordable))
    return order


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #
def compute_score(task_key: str, step_records: List[dict[str, Any]]) -> float:
    if not step_records:
        return 0.001

    if task_key == "task1":
        survived = sum(1 for r in step_records if r.get("budget", 0.0) > 0.0)
        score = survived / MAX_STEPS

    elif task_key == "task2":
        weeks_above = 0
        for r in step_records:
            fulfilled    = sum(r.get("demand",       [0] * N))
            unmet        = sum(r.get("unmet_demand", [0] * N))
            total_demand = fulfilled + unmet
            fill_rate    = (fulfilled / total_demand) if total_demand > 0 else 1.0
            if fill_rate >= 0.80:
                weeks_above += 1
        score = weeks_above / MAX_STEPS

    else:  # task3
        total_profit    = sum(r.get("reward", 0.0) for r in step_records)
        max_per_week    = sum(DEMAND_MEAN[i] * SELLING_PRICES[i] for i in range(N))
        theoretical_max = max_per_week * MAX_STEPS
        score = total_profit / theoretical_max if theoretical_max > 0 else 0.001

    return round(max(min(score, 0.999), 0.001), 4)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_key in ["task1", "task2", "task3"]:
        task_name, max_steps, success_threshold = TASK_REGISTRY[task_key]
        env = DeepMatrixHTTPEnv(SPACE_URL)

        history:      List[str]            = []
        rewards:      List[float]          = []
        step_records: List[dict[str, Any]] = []
        steps_taken   = 0
        success       = False
        score         = 0.001

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            result      = env.reset()
            obs         = _get_obs(result)
            done        = obs.get("done", False)
            last_reward = 0.0

            for step in range(1, max_steps + 1):
                if done:
                    break

                action            = get_llm_action(client, step, obs, last_reward, history)
                error: Optional[str] = None
                step_reward       = 0.0

                try:
                    result      = env.step(action)
                    obs         = _get_obs(result)
                    step_reward = float(obs.get("reward", 0.0) or 0.0)
                    done        = bool(obs.get("done", False) or False)
                    metadata    = obs.get("metadata", {}) or {}
                except Exception as exc:
                    error       = str(exc)
                    done        = True
                    metadata    = {}
                    print(f"[DEBUG] env.step() error: {exc}", flush=True)

                rewards.append(step_reward)
                steps_taken = step
                last_reward = step_reward

                step_records.append({
                    "reward":       step_reward,
                    "budget":       obs.get("budget",        0.0),
                    "demand":       obs.get("demand",        [0] * N),
                    "unmet_demand": metadata.get("unmet_demand", [0] * N),
                })

                log_step(
                    step=step,
                    action=action,
                    reward=step_reward,
                    done=done,
                    error=error,
                )

                history.append(
                    f"Week {step}: orders={action} "
                    f"reward={step_reward:+.2f} budget=${obs.get('budget', 0):,.0f}"
                )

                if done:
                    break

            score   = compute_score(task_key, step_records)
            success = score >= success_threshold

        except Exception as exc:
            print(f"[DEBUG] Episode failed: {exc}", flush=True)

        finally:
            env.close()
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()