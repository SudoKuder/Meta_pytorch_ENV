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
  - reinforcement-learning
  - inventory-management
---

# Meta_pytorch_ENV

DeepMatrix is a 10-SKU inventory management environment for OpenEnv deployed on Hugging Face Spaces.
Each step represents one week of operations. Agents choose order quantities, then the environment applies
lead times, demand realization, waste/expiry, budget constraints, and reward calculation.

## Quick Start

```python
from DeepMatrix import DeepmatrixAction, DeepmatrixEnv

with DeepmatrixEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print("Initial budget:", result.observation.budget)

    action = DeepmatrixAction(items_to_buy=[0] * 10)
    result = env.step(action)
    print("Reward:", result.reward)
```

## Run Locally with Docker

```bash
docker build -t deepmatrix-env:latest .
docker run --rm -p 8000:8000 deepmatrix-env:latest
```

## API Endpoints

- `GET /health` — health check
- `GET /docs` — interactive API docs
- `GET /web` — web interface
- `POST /reset` — reset the environment
- `POST /step` — execute an action
- `WS /ws` — WebSocket for persistent sessions