# Project Notes Archive

This folder stores historical experiment notes.

## Status after cleanup

The following one-off experiment scripts have been removed from the repository root:

- `analyze_idsw_events.py`
- `run_bytetrack_ablation.py`
- `run_improved_only.py`
- `run_improved_v2.py`
- `run_mot17_04_assoc_eval_only.py`
- `run_mot17_04_comparison.py`
- `run_mot17_04_param_sweep.py`
- `run_mot20_ablation.py`

Related notes in this folder may still mention those scripts as historical context.

## Recommended workflow

Use the maintained CLI entrypoint instead:

```bash
uv run python -m boxmot.engine.cli --help
uv run python -m boxmot.engine.cli eval yolox_x_MOT17_ablation lmbn_n_duke bytetrack --source MOT17-ablation
```
