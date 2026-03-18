# AGENTS.md – Working Guidelines for **BoxMOT**

> These instructions apply to all directories in this repository.  \
> Nested `AGENTS.md` files (if added later) override rules for their subtrees.

---

## 1. Environment & Tooling

### Python & `uv`

- Use **Python 3.11** (or the version configured in `pyproject.toml`).
- Install `uv` (safe to rerun even if present):

  ```bash
  pip install uv
  ```

- Install dependencies using the existing workflow:

  ```bash
  uv sync --all-extras --all-groups
  ```

- `uv` will create a `.venv` in the project root. Prefer running everything through `uv` so you don’t have to manage activation manually:

  ```bash
  # Generic command wrapper
  uv run <command> [args...]
  ```

#### Running with the package context

Always run Python entry points as modules from the repo root, not as loose scripts, so that `from boxmot...` imports work correctly:

```bash
# ✅ Good – uses package context
uv run python -m boxmot.engine.cli --help

# ❌ Avoid – can break imports (e.g., ModuleNotFoundError: boxmot)
python boxmot/engine/cli.py --help
PYTHONPATH=. python boxmot/engine/cli.py --help
```

If you really need to use the virtualenv directly:

```bash
source .venv/bin/activate
python -m boxmot.engine.cli --help
```

### Local Environment Notes

- The current `.venv` has been validated on the local Blackwell GPU with:

  ```bash
  torch==2.10.0+cu128
  torchvision==0.25.0+cu128
  ```

- `osnet_x0_25_msmt17.pt` is present in the repo root:

  ```bash
  /root/autodl-tmp/boxmot_demo2/osnet_x0_25_msmt17.pt
  ```

- Test-time ReID weights are currently standardized on `osnet_x0_25_msmt17.pt`.
  - `tests/unit/test_trackers.py` and related tracker/performance tests now use this weight by default.
  - If `boxmot/engine/weights/osnet_x0_25_msmt17.pt` is missing but the repo-root file above exists, the ReID backend now reuses the repo-root file instead of attempting a fresh download.

- The local SOMPT22 dataset used for evaluation is under:

  ```bash
  /root/autodl-tmp/boxmot_demo2/train
  ```

- The local MOT20 dataset used for evaluation is under:

  ```bash
  /root/autodl-tmp/boxmot_demo2/MOT20_train/train
  ```

- The historical default cache pattern under `runs/` is not present in the current workspace. If an old cache tree matching the paths below reappears, treat it as stale placeholder data and do **not** use it for current ReID-enabled ByteTrack evaluation:

  ```bash
  runs/dets_n_embs/yolov8m_pretrain_crowdhuman/dets
  runs/dets_n_embs/yolov8m_pretrain_crowdhuman/embs/osnet_x0_25_msmt17
  ```

- The canonical reusable SOMPT22 **person-only** det/emb cache for cross-tracker comparisons is under:

  ```bash
  /root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17
  ```

- The shared cache resolves to these paths:

  ```bash
  /root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17/dets_n_embs/yolov8m_pretrain_crowdhuman/dets
  /root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17/dets_n_embs/yolov8m_pretrain_crowdhuman/embs/osnet_x0_25_msmt17
  ```

- This shared cache is currently backed by:

  ```bash
  /root/autodl-tmp/boxmot_demo2/runs_botsort_sompt22_20260317_120609/dets_n_embs
  ```

  Notes:
  - It was generated with detector `yolov8m_pretrain_crowdhuman`, ReID `osnet_x0_25_msmt17`, and `--classes 0`.
  - The cached SOMPT22 detection files currently contain only class id `0`, i.e. `person`.
  - Use this cache root as `--project` when comparing different trackers on the same detector/ReID/class setup.
  - Keep the detector stem, ReID stem, and `--classes 0` unchanged; vary only `TRACKER` and optionally `--tracker-config`.
  - Successful reuse is indicated by logs such as `Skipping SOMPT22-xx (cached complete...)` followed by `No sequences to process (all cached or no images).`
  - If detector weights, ReID weights, or class filtering change, create a new `--project` root or remove only that root's `dets_n_embs/` subtree before rerunning.

- The latest MOT20 **person-only** det/emb cache generated with detector `yolov8m_pretrain_crowdhuman`, ReID `osnet_x0_25_msmt17`, and `--classes 0` is under:

  ```bash
  /root/autodl-tmp/boxmot_demo2/runs_mot20_bytetrack_improved_20260317_132003
  ```

- The MOT20 cache resolves to these paths:

  ```bash
  /root/autodl-tmp/boxmot_demo2/runs_mot20_bytetrack_improved_20260317_132003/dets_n_embs/yolov8m_pretrain_crowdhuman/dets
  /root/autodl-tmp/boxmot_demo2/runs_mot20_bytetrack_improved_20260317_132003/dets_n_embs/yolov8m_pretrain_crowdhuman/embs/osnet_x0_25_msmt17
  ```

- The corresponding MOT20 tracking results for the improved ByteTrack run are under:

  ```bash
  /root/autodl-tmp/boxmot_demo2/runs_mot20_bytetrack_improved_20260317_132003/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack
  ```

- Latest verified full fresh eval summary on the local SOMPT22 split:

  ```text
  HOTA=53.023
  MOTA=65.710
  IDF1=66.179
  IDSW=815
  Association FPS=18.9
  ```

- Current implementation status for ByteTrack:
  - Step1/Step2 keep the standard ByteTrack high-score / low-score association flow.
  - ReID is only injected into the final zombie-rescue stage.
  - Zombie rescue uses `center-distance hard gate + shape gate + ReID-dominant weighted cost + Hungarian assignment`.
  - Main implementation files:

    ```bash
    /root/autodl-tmp/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py
    /root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_improved.yaml
    /root/autodl-tmp/boxmot_demo2/boxmot/trackers/tracker_zoo.py
    /root/autodl-tmp/boxmot_demo2/boxmot/engine/evaluator.py
    ```

## 2. Workflow

- Create feature branches for work:

  ```bash
  git checkout -b codex/<short-topic>
  ```

- Keep changes focused: one logical change per PR / task.
- Follow the existing structure and conventions of the modules you touch.

## 3. Coding Conventions

- Prefer Python type hints and docstrings for any new or modified functions/classes.
- Keep imports:
  - Sorted.
  - Minimal (remove unused).
- Do not wrap imports in `try/except` unless there is a very specific reason and it’s clearly documented.

**Logging**
- Use the existing logger (e.g., `LOGGER`) rather than `print` in library code.
- It’s fine to `print` in CLI entry points when it improves UX, but prefer consistent logging style.

**Match the surrounding style**
- Naming, spacing, line wrapping, click option style, etc.
- Reuse helper patterns (e.g., decorators like `core_options`, shared parsing helpers).

## 4. CLI-Specific Guidelines

When editing `boxmot/engine/cli.py` or other CLIs:

- Group options logically (e.g., input, inference, output, display), but maintain backwards-compatible option names and defaults where possible.
- Prefer reusable decorators for option groups (`core_options`, `plural_model_options`, etc.).
- Use parsing helpers (e.g., `parse_tuple`, `parse_hw_tuple`) rather than ad-hoc parsing in every command.
- Keep help text accurate and concise; if you change behavior, update:
  - The option help strings.
  - Any CLI examples in `README.md`, `docs/`, or `examples/`.

When adding a new command:

- Reuse `make_args` to build argparse-like namespaces.
- Align with existing subcommands’ style (`track`, `generate`, `eval`, `tune`, `export`).

## 5. Commit & PR Expectations

Commit messages should start with one of:

- `feat:` – new feature
- `fix:` – bug fix
- `refactor:` – internal-only changes / cleanup
- `docs:` – documentation only
- `ci:` – CI / tooling changes
- `perf:` – performance improvements

Each commit should represent a coherent change; avoid mixing unrelated edits.

PR / task descriptions should include:

- A short summary of user-facing changes.
- A Testing section (see below).
- Any follow-up work or known limitations.

## 6. Testing & Verification

**What to run**

- Default: run the pytest suite from the repo root:

  ```bash
  uv run pytest
  ```

- If the full suite is too heavy, at least run the tests relevant to your change, e.g.:

  ```bash
  uv run pytest tests/test_cli.py
  uv run pytest tests/path/to/affected_module_tests.py
  ```

- When touching CLI / engine entry points, it’s useful to smoke-test common commands:

  ```bash
  uv run python -m boxmot.engine.cli --help

  # Example invocations (adjust source/paths as available in your env)
  uv run python -m boxmot.engine.cli track --source <path-or-url> ...
  uv run python -m boxmot.engine.cli generate --source <path-or-url> ...
  uv run python -m boxmot.engine.cli eval --source <path-or-url> ...
  uv run python -m boxmot.engine.cli tune --source <path-or-url> ...
  ```

- For the current ReID-enhanced ByteTrack work, use this focused regression command first:

  ```bash
  uv run pytest tests/unit/test_trackers.py -k "zombie_reid_global_assignment_prefers_appearance or zombie_reid_gate_blocks_wrong_appearance_rescue"
  ```

- Current locally verified tracker test commands:

  ```bash
  uv run pytest tests/unit/test_reidbackend.py -k "resolves_existing_root_weights_before_download"
  uv run pytest tests/unit/test_trackers.py
  ```

  Notes:
  - `tests/unit/test_trackers.py` is currently verified passing locally with `98 passed`.
  - When touching tracker tests, continue using `osnet_x0_25_msmt17.pt` unless there is a very specific reason to validate another ReID weight.

- To run a **fresh** SOMPT22 eval that does not reuse old placeholder embeddings, always write into a new `--project` directory or manually remove that directory's `dets_n_embs/` subtree first:

  ```bash
  RUN_ROOT=/root/autodl-tmp/boxmot_demo2/runs_fullrerun_$(date -u +%Y%m%d_%H%M%S)

  uv run python -m boxmot.engine.cli eval \
    yolov8m_pretrain_crowdhuman \
    osnet_x0_25_msmt17 \
    bytetrack \
    --source /root/autodl-tmp/boxmot_demo2/train \
    --tracker-config /root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_improved.yaml \
    --device 0 \
    --project "$RUN_ROOT" \
    --exist-ok \
    --verbose
  ```

- To reuse the shared SOMPT22 **person-only** dets/embs when comparing trackers, keep the same detector/ReID/class filter and point `--project` to the shared cache root:

  ```bash
  CACHE_ROOT=/root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17

  uv run python -m boxmot.engine.cli eval \
    yolov8m_pretrain_crowdhuman \
    osnet_x0_25_msmt17 \
    bytetrack \
    --source /root/autodl-tmp/boxmot_demo2/train \
    --classes 0 \
    --tracker-config /root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_improved.yaml \
    --device 0 \
    --project "$CACHE_ROOT" \
    --exist-ok \
    --verbose
  ```

  Notes:
  - Swap `bytetrack` and `--tracker-config` for the tracker under test.
  - Reuse works only when `yolov8m_pretrain_crowdhuman`, `osnet_x0_25_msmt17`, and `--classes 0` stay the same.
  - New tracker outputs will be written under `$CACHE_ROOT/mot/` while `dets_n_embs/` is reused unchanged.

- After a fresh eval, inspect the summary output and, if you piped stdout through `tee`, the saved console log:

  ```bash
  tail -n 80 "$RUN_ROOT/console.log"
  sed -n '1,5p' "$RUN_ROOT/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack/person_summary.txt"
  ```

**If tests or commands cannot be run**

Sometimes the provided environment is missing GPUs, large datasets, or external services. In that case:

1. Try the following first:

   ```bash
   uv sync --all-extras --all-groups

   uv run python -m boxmot.engine.cli --help

   uv run pytest
   ```

2. If something still fails for reasons outside your control (e.g., missing CUDA runtime, no network for model downloads, etc.), do not fake test results. Instead, document clearly in your Testing section, for example:

   ```text
   Testing
   - uv run python -m boxmot.engine.cli --help  ✅
   - uv run pytest ❌ (not run)

   Reason: pytest requires GPU / CUDA dependencies that are not available in the current container.
   Please run `uv sync --all-extras --all-groups` and `uv run pytest` in a fully configured environment.
   ```

- Include the exact commands you ran and a brief reason why anything couldn’t be completed.

## 7. Documentation & Examples

- Update docs or examples when behavior or interfaces change, especially:
  - CLI options or defaults.
  - New or removed commands.
- Keep README snippets and CLI help text in sync with code updates.
- When changing data formats or output directories, update any references in:
  - `docs/`
  - `examples/`
  - `tests/`

## 8. Performance & Safety

- Be mindful of model weights and large assets:
  - Do not commit generated artifacts or large binaries.
  - Prefer referencing weights via URLs or documented download steps.
- Where practical:
  - Use deterministic or seeded behavior for tests/examples.
  - Avoid unnecessary heavy computation in unit tests.
