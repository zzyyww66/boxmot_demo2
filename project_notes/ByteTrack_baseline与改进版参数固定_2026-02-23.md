# ByteTrack baseline 与改进版参数固定 (2026-02-23)

本文档用于把“baseline(原版) ByteTrack”与“改进版 ByteTrack”的参数固定下来，作为后续所有实验的唯一对齐基准。

说明:

- 本仓库 tracker YAML 是“可调参空间 + default”的格式。
- **评测/推理时实际生效的是每个参数的 `default` 值**。
- 本文档只记录 default 生效值，便于复现与对齐。

## 1. Baseline: 真正原版 ByteTrack (固定)

配置文件:

- `boxmot/configs/trackers/bytetrack_original.yaml`

原版核心阈值 (对齐常见 ByteTrack 默认):

- `track_buffer = 30`
- `track_thresh = 0.5`
- `new_track_thresh = 0.5`
- `match_thresh = 0.8`

明确关闭改进功能 (baseline 必须“干净”):

- `entry_margin = 0` (entry gate 关闭)
- `strict_entry_gate = false` (即使开启 entry gate，也不做中心硬禁止；但 entry gate 本身已关闭)
- `birth_confirm_frames = 1` (新生二次确认关闭)
- `birth_suppress_iou = 0.0` (新生抑制关闭)
- `birth_suppress_center_dist = 0` (新生抑制关闭)
- `zombie_max_history = 0` / `zombie_max_predict_frames = 0` (zombie 关闭)
- `exit_zone_enabled = false` (exit zone 关闭)
- `adaptive_zone_enabled = false` (adaptive zone 关闭)

## 2. Improved: 当前改进版 ByteTrack (固定)

配置文件:

- `boxmot/configs/trackers/bytetrack_improved.yaml`

当前固定的 default 值 (后续实验不要再改动，除非开新版本号):

- `track_buffer = 30`
- `track_thresh = 0.6`
- `new_track_thresh = 0.65`
- `match_thresh = 0.9`

entry/birth 相关:

- `entry_margin = 50`
- `strict_entry_gate = false` (按既有最终记录保持为 false)
- `birth_confirm_frames = 2`
- `birth_suppress_iou = 0.7`
- `birth_suppress_center_dist = 35`

zombie 相关:

- `zombie_max_history = 100`
- `zombie_dist_thresh = 150`
- `zombie_transition_frames = 30`
- `zombie_match_max_dist = 200`
- `zombie_max_predict_frames = 5`

exit zone 相关:

- `exit_zone_enabled = true`
- `exit_zone_margin = 50`
- `exit_zone_remove_grace = 30`

adaptive zone 相关:

- `adaptive_zone_enabled = true`
- `adaptive_zone_update_mode = always_expand`
- `adaptive_zone_expand_trigger = all_high`
- `adaptive_zone_warmup = 10`
- `adaptive_zone_margin = 50`
- `adaptive_zone_padding = 1.2`
- `adaptive_zone_min_box_area = 0`

## 3. 使用方式 (建议统一)

baseline:

```bash
uv run python -m boxmot.engine.cli eval \\
  --tracking-method bytetrack \\
  --tracker-config boxmot/configs/trackers/bytetrack_original.yaml \\
  ...
```

improved:

```bash
uv run python -m boxmot.engine.cli eval \\
  --tracking-method bytetrack \\
  --tracker-config boxmot/configs/trackers/bytetrack_improved.yaml \\
  ...
```

