---
title: DeepMatrix Environment
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# DeepMatrix Environment

DeepMatrix is a 10-SKU inventory management environment for OpenEnv.
Each step represents one week of operations. Agents choose order quantities,
then the environment applies lead times, demand realization, waste/expiry,
budget constraints, and reward calculation.

This environment models a real-world supply-chain planning task (not a game).

## Quick Start

```python
from DeepMatrix import DeepmatrixAction, DeepmatrixEnv

with DeepmatrixEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print("Initial budget:", result.observation.budget)

    action = DeepmatrixAction(items_to_buy=[0] * 10)
    result = env.step(action)
    print("Reward:", result.reward)
    print("Service level:", result.observation.cumulative_service_level)
```

## Run Locally

```bash
# From repository root
uvicorn DeepMatrix.server.app:app --reload --host 0.0.0.0 --port 8000
```

## Build Docker Image

```bash
# From repository root
docker build -t deepmatrix-env:latest -f DeepMatrix/server/Dockerfile DeepMatrix
```

## OpenEnv Manifest

The environment manifest is in `openenv.yaml` and includes:
- Canonical environment class path
- Action and observation schema references
- Task file references for all 3 contest tasks

## Action Space

- `items_to_buy: list[int]` with length `10`
- One non-negative integer per SKU each week
- Environment enforces budget caps on effective order quantities

## Observation Space

- `inventory: list[int]` (10)
- `in_transit: list[int]` (10)
- `demand: list[int]` (10)
- `demand_forecast: list[float]` (10)
- `demand_forecast_std: list[float]` (10)
- `buying_price: list[float]` (10)
- `arrival_time: list[int]` (10)
- `expiry_time: list[int]` (10)
- `waste: list[int]` (10)
- `budget: float`
- `cumulative_service_level: float`
- `cumulative_waste_units: int`
- `cumulative_profit: float`
- `reward: float`
- `done: bool`
- `metadata: dict`

## Reward Function

Weekly reward is profit after all operating costs:

`reward = revenue - (purchase_cost + logistics_cost + holding_cost)`

Partial progress signals are always exposed in observations:
- `cumulative_service_level`
- `cumulative_waste_units`
- `cumulative_profit`
- `budget`

## Tasks

- `tasks/task1_budget_survival.py` (easy)
- `tasks/task2_service_level.py` (medium)
- `tasks/task3_profit_max.py` (hard)

Each task script can be run directly:

```bash
python DeepMatrix/tasks/task1_budget_survival.py
python DeepMatrix/tasks/task2_service_level.py
python DeepMatrix/tasks/task3_profit_max.py
```

All tasks produce normalized scores in `[0.0, 1.0]` via their graders.

## Reproducible Baseline Inference

`Inference.py` runs deterministic baseline agents for all tasks across fixed seeds
and writes a reproducible report.

```bash
python DeepMatrix/Inference.py
```

Optional overrides:

```bash
DEEPMATRIX_BASELINE_SEEDS=42,123,999 DEEPMATRIX_BASELINE_OUTPUT=baseline_scores.json python DeepMatrix/Inference.py
```

## Project Structure

```text
DeepMatrix/
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── Inference.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── DeepMatrix_environment.py
│   └── Dockerfile
└── tasks/
    ├── task1_budget_survival.py
    ├── task2_service_level.py
    └── task3_profit_max.py
```

## Hugging Face Spaces Deployment

1. Build and test locally:

```bash
docker build -t deepmatrix-env:latest -f DeepMatrix/server/Dockerfile DeepMatrix
docker run --rm -p 8000:8000 deepmatrix-env:latest
```

2. Validate endpoints:

```bash
curl http://localhost:8000/health
```

3. Push using OpenEnv:

```bash
cd DeepMatrix
openenv push
```

4. On Spaces, verify:
- `/health`
- `/docs`
- `/web`
