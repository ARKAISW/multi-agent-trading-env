---
title: Indie Quant Office
sdk: docker
app_port: 7860
---

# Indie Quant Office

Live multi-agent trading desk demo with a cutesy office visualization.

## Local run

Backend:

```bash
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Frontend dev server:

```bash
cd ui
npm install
npm run dev
```

The Vite app proxies `/api/*` to the FastAPI server on port `8000`.

## Hugging Face Spaces

This repository is configured for a Docker Space. The container:

1. Installs the minimal Python runtime for the live demo.
2. Builds the React frontend.
3. Serves the built UI and the simulation API from the same FastAPI process on port `7860`.

## Local policy model

If you want the trader to use a local small language model instead of the fallback rule policy, provide:

- `USE_LOCAL_POLICY=true`
- `LOCAL_MODEL_PATH=/path/to/your/model`

You will also need to install `torch` and `transformers` in the runtime image.
