from tasks.task1_budget_survival import run_task as run_task1, conservative_agent
from tasks.task2_service_level import run_task as run_task2, safety_stock_agent
from tasks.task3_profit_max import run_task as run_task3, adaptive_agent

TASKS = [
    {
        "id": "task1_budget_survival",
        "name": "Budget Survival",
        "difficulty": "easy",
        "run_task": run_task1,
        "default_agent": conservative_agent,
    },
    {
        "id": "task2_service_level",
        "name": "Service Level Optimizer",
        "difficulty": "medium",
        "run_task": run_task2,
        "default_agent": safety_stock_agent,
    },
    {
        "id": "task3_profit_max",
        "name": "Profit Maximization Under Constraints",
        "difficulty": "hard",
        "run_task": run_task3,
        "default_agent": adaptive_agent,
    },
]

def grade(task_id: str, agent_fn=None, seed: int = 42) -> dict:
    for task in TASKS:
        if task["id"] == task_id:
            fn = agent_fn or task["default_agent"]
            return task["run_task"](fn, seed=seed)
    raise ValueError(f"Unknown task: {task_id}")

if __name__ == "__main__":
    for task in TASKS:
        result = task["run_task"](task["default_agent"])
        print(f"{task['id']}: {result['score']}")