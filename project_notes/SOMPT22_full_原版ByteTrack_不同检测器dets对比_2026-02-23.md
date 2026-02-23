# SOMPT22-full 原版 ByteTrack 不同检测器 dets 对比 (2026-02-23)

本次对比目标:

- 使用**当前固定的真正原版 ByteTrack 参数**进行追踪评估。
- 仅更换检测来源，比较 `yolov8m` 与 `yolox_x` 产生的 dets 对最终指标的影响。

## 1. 评估设置

数据集:

- `SOMPT22-full`

跟踪器:

- `ByteTrack (baseline / true original)`
- 配置文件: `boxmot/configs/trackers/bytetrack_original.yaml`

原版核心参数 (default):

- `track_thresh = 0.5`
- `new_track_thresh = 0.5`
- `match_thresh = 0.8`
- `track_buffer = 30`

检测来源:

- `yolov8m_pretrain_crowdhuman.pt` 的 dets 缓存
- `yolox_x_MOT17_ablation.pt` 的 dets 缓存

## 2. 结果目录

yolov8m:

- `runs/sompt22_compare_v8m_full_final/mot/yolov8m_pretrain_crowdhuman_lmbn_n_duke_bytetrack_3`

yolox_x:

- `runs/sompt22_full_yoloxx_compare/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_3`

## 3. COMBINED 指标对比

| 指标 | yolov8m dets | yolox_x dets | 差值 (yolox - yolov8m) |
|---|---:|---:|---:|
| HOTA | 53.129 | 51.907 | -1.222 |
| MOTA | 63.852 | 57.853 | -5.999 |
| IDF1 | 64.852 | 63.048 | -1.804 |
| AssA | 52.328 | 51.791 | -0.537 |
| AssRe | 57.034 | 57.311 | +0.277 |
| IDSW | 1255 | 1064 | -191 |
| IDs | 2033 | 1924 | -109 |
| CLR_FP | 77156 | 94689 | +17533 |
| CLR_FN | 115308 | 130113 | +14805 |

## 4. 结论

- 在同一原版 ByteTrack 参数下，`yolox_x` 相比 `yolov8m` 的 `IDSW` 更低:
  - `1255 -> 1064`，减少 `191` 次 (约 `-15.2%`)。
- 但 `yolox_x` 的整体综合指标更低:
  - `HOTA / MOTA / IDF1` 均低于 `yolov8m`。
- 说明在该设置下，`yolox_x` 减少了部分身份切换，但同时引入了更多 `FP/FN`，对整体跟踪质量有负向影响。

