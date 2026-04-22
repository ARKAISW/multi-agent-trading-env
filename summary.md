# Project Update - 2026-04-22

- Migrated `PortfolioManager` to use standard `openai` client instead of LangChain/Gemini.
- Resolved `Optional` lint issues in `env/state.py`.
- Resolved `openai` import issues in `utils/judge.py`.
- Standardized external API interactions to use the OpenAI client globally as per project requirements.
- Cleaned up `requirements.txt` dependencies.
- Resolved `ModuleNotFoundError` for `openai` and `langchain-openai` in the virtual environment.
- Fixed the demo API/runtime mismatches so `SimulationRunner` uses the current agent return signatures and passes SL/TP through to the environment.
- Corrected trading logic to support fractional position sizing, track average cost directly, and avoid repeatedly penalizing the same historical trades in reward computation.
- Added normalized portfolio metrics to environment info, fixed evaluation metric consistency, and wired training to export JSONL trajectories for local-policy fine-tuning.
- Updated the local policy loader to attempt LoRA adapter inference with a safe heuristic fallback, refreshed the Colab/demo notebooks, and declared missing runtime dependencies in `requirements.txt`.
- Added `tests/smoke_test.py` so the full environment, fast training loop, and demo runner can be validated with one command before the hackathon demo.
- Added `ui/postcss.config.js` so the React visualization in `ui/` compiles Tailwind directives properly instead of shipping raw `@tailwind` CSS.
- Updated `utils/judge.py` to load `.env`, support Groq-compatible credentials via `GROQ_API_KEY`, and fall back cleanly to a neutral reward if the remote judge key is missing or invalid.

### Latest Status
- Ready for professional Hackathon evaluation using local Qwen 2.5 1.5B and remote Llama 3.3 70B Judge.
