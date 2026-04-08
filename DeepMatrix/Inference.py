"""
DeepMatrix Inference Script
===================================
MANDATORY ENV VARS:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN / API_KEY
    IMAGE_NAME
"""

import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import OpenAI

# Assuming OpenEnv dynamically provides these or they are imported locally
from models import DeepmatrixAction
from server.DeepMatrix_environment import DeepmatrixEnvironment

IMAGE_NAME = os.getenv("IMAGE_NAME") 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

TASK_NAME = os.getenv("DEEPMATRIX_TASK", "budget_survival")
BENCHMARK = os.getenv("DEEPMATRIX_BENCHMARK", "deepmatrix")
MAX_STEPS = 52  # 52 weeks in a year
TEMPERATURE = 0.1 # Low temperature for more deterministic, logical math
MAX_TOKENS = 150

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert AI Supply Chain Manager. 
    You manage inventory for 5 SKUs. Each week, you must decide how many units of each SKU to order.
    Your goal depends on the task, but generally, you want to maintain a high service level without running out of budget or creating too much waste.
    
    You must respond ONLY with a valid JSON array of 5 integers representing the quantities to buy for SKUs 0 through 4. 
    Do not include markdown formatting, explanations, or any other text.
    Example valid response: [10, 0, 25, 100, 5]
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs: dict) -> str:
    return textwrap.dedent(
        f"""
        Week: {step} / 52
        Current Budget: ${obs.get('budget', 0):.2f}
        Cumulative Profit: ${obs.get('cumulative_profit', 0):.2f}
        Service Level: {obs.get('cumulative_service_level', 0):.2f}
        
        SKU Status (0 to 4):
        Inventory on hand: {obs.get('inventory', [])}
        In Transit: {obs.get('in_transit', [])}
        Buying Prices: {obs.get('buying_price', [])}
        Demand Forecast (Mean): {obs.get('demand_forecast', [])}
        
        Output your order quantities as a JSON array of 5 integers:
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs: dict) -> List[int]:
    user_prompt = build_user_prompt(step, obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Clean up potential markdown formatting the LLM might add
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
            
        action_list = json.loads(text)
        
        # Validate it's a list of 5 integers
        if isinstance(action_list, list) and len(action_list) == 5:
            return [int(x) for x in action_list]
        else:
            raise ValueError("LLM did not return a list of 5 items.")
            
    except Exception as exc:
        print(f"[DEBUG] Model request failed or returned invalid format: {exc}", flush=True)
        return [0, 0, 0, 0, 0] # Safe fallback: order nothing if LLM fails

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize your specific environment (Local or Docker based on OpenEnv setup)
    env = DeepmatrixEnvironment(max_weeks=MAX_STEPS)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        # OpenEnv Reset
        obs = env.reset()
        
        # Using dict representation for easier prompt building
        obs_dict = obs.dict() if hasattr(obs, 'dict') else vars(obs)

        for step in range(1, MAX_STEPS + 1):
            if obs_dict.get('done', False):
                break

            # Get action from LLM
            items_to_buy = get_model_action(client, step, obs_dict)
            action_str = json.dumps(items_to_buy)

            # Step the environment
            try:
                action_obj = DeepmatrixAction(items_to_buy=items_to_buy)
                obs = env.step(action_obj)
                obs_dict = obs.dict() if hasattr(obs, 'dict') else vars(obs)
                error = None
            except Exception as e:
                error = str(e)
                obs_dict['done'] = True # Terminate on env error
            
            # Extract reward (using profit change as step reward)
            current_profit = obs_dict.get('cumulative_profit', 0.0)
            previous_profit = sum(rewards) if rewards else 0.0
            step_reward = current_profit - previous_profit
            
            done = obs_dict.get('done', False)

            rewards.append(step_reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=step_reward, done=done, error=error)

            if done:
                break

        # Calculate final score based on Budget Survival task logic as an example
        weeks_survived = steps_taken
        avg_svc = obs_dict.get('cumulative_service_level', 0.0)
        base_score = min(weeks_survived / 26.0, 1.0)
        bonus = 0.1 if avg_svc >= 0.50 else 0.0
        
        score = min(base_score + bonus, 1.0)
        success = score >= 0.8  # Threshold for success

    finally:
        try:
            # Add cleanup if using docker container
            pass 
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())