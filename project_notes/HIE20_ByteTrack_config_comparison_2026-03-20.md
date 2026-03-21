# HIE20 ByteTrack 配置专项对比（2026-03-20）

## 1. 结论先行

- 已完成 HIE20 上 5 组 ByteTrack 配置的专项测试：`default`、`original`、`improved`、`sompt22_tuned`、`dual_tuned`。
- 本次专项对比复用了已经跑好的检测与 ReID 缓存，没有重新生成 `dets` 和 `embs`。
- 当前 HIE20 上表现最好的 ByteTrack 配置是 `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml`。
- 该配置在 HIE20 上的核心指标为：`HOTA 44.564`、`MOTA 58.367`、`IDF1 54.786`、`AssA 42.574`、`IDSW 2705`、`Association FPS 325.8`。
- 如果把这次配置专项结果与此前的全算法对比一起看，那么 `ByteTrack + bytetrack_sompt22_tuned.yaml` 已经在 HIE20 上超过此前的 `BotSort`，拿到当前已测方案中的最高 `HOTA` 与最高 `IDF1`；但 `MOTA` 仍略低于 `BotSort`。

## 2. 评测口径

- 数据集：`HIE20/train`
- 检测器权重：`yolov8m_pretrain_crowdhuman.pt`
- ReID 权重：`osnet_x0_25_msmt17.pt`
- 类别过滤：`--classes 0`，即仅评测 `person`
- 运行根目录：`runs_hie20_bytetrack_cfgs_20260320_025718`
- 缓存复用方式：`runs_hie20_bytetrack_cfgs_20260320_025718/dets_n_embs` 为软链接，指向 `runs_hie20_tracker_compare_20260320_014617/dets_n_embs`
- 缓存复用证据：日志中出现 `Skipping HIE20-xx (cached complete ...)`
- 运行状态：5 组配置对应的 `*.exitcode` 全部为 `0`

## 3. 总体结果

| Rank | Config | 配置文件 | HOTA | MOTA | IDF1 | AssA | AssRe | IDSW | IDs | Assoc FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `sompt22_tuned` | `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml` | **44.564** | 58.367 | **54.786** | **42.574** | 48.934 | **2705** | 1836 | 325.8 |
| 2 | `dual_tuned` | `boxmot/configs/trackers/bytetrack_dual_tuned.yaml` | 44.284 | 58.497 | 54.243 | 42.068 | 48.849 | 2771 | 1684 | 338.4 |
| 3 | `improved` | `boxmot/configs/trackers/bytetrack_improved.yaml` | 44.030 | 57.633 | 53.572 | 42.287 | **49.526** | 2760 | **1559** | 331.9 |
| 4 | `original` | `boxmot/configs/trackers/bytetrack_original.yaml` | 40.753 | **59.396** | 46.710 | 34.499 | 37.392 | 5254 | 5547 | **351.9** |
| 5 | `default` | `boxmot/configs/trackers/bytetrack.yaml` | 40.353 | 58.262 | 46.453 | 34.940 | 38.127 | 4619 | 4075 | 343.6 |

## 4. 相对默认配置的增益

这里的默认配置指不传 `--tracker-config` 时使用的 `boxmot/configs/trackers/bytetrack.yaml`。

| Config | ΔHOTA | ΔMOTA | ΔIDF1 | ΔAssA | ΔAssRe | ΔIDSW | ΔIDs | ΔAssoc FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | +0.400 | +1.134 | +0.257 | -0.441 | -0.735 | +635 | +1472 | +8.3 |
| `improved` | +3.677 | -0.629 | +7.119 | +7.347 | +11.399 | -1859 | -2516 | -11.7 |
| `sompt22_tuned` | **+4.211** | +0.105 | **+8.333** | **+7.634** | +10.807 | **-1914** | -2239 | -17.8 |
| `dual_tuned` | +3.931 | +0.235 | +7.790 | +7.128 | +10.722 | -1848 | -2391 | -5.2 |

## 5. 关键观察

- `sompt22_tuned` 是当前 HIE20 上最强的 ByteTrack 配置。它同时拿到了本组中的最高 `HOTA`、最高 `IDF1`、最高 `AssA`，并且 `IDSW` 也是最低。
- `dual_tuned` 非常接近 `sompt22_tuned`。如果后续目标仍然是跨 `SOMPT22 / MOT20 / HIE20` 尽量共用一套参数，它仍然是很有价值的折中方案。
- `improved` 仍然是一条很强的基线。它在 3 个增强版配置中拿到了最高 `AssRe` 和最低 `IDs`，说明找回能力和整体身份碎裂控制都不错，但综合指标略输给 `sompt22_tuned` 与 `dual_tuned`。
- `original` 的现象很典型：`MOTA` 最高，但 `HOTA / IDF1 / AssA / IDSW` 都明显更差，说明它更偏向检测/召回层面的得分，对身份一致性并不友好。
- 从 `default/original` 到 `improved/sompt22_tuned/dual_tuned`，核心收益几乎都来自身份关联能力的大幅提升，而不是单纯依靠更高的检测匹配召回。

## 6. 与此前全算法对比的关系

此前的全算法报告见 `project_notes/HIE20_tracker_comparison_2026-03-20.md`，其中当时的第一名是 `BotSort`：

- `BotSort`: `HOTA 44.251`，`MOTA 58.612`，`IDF1 53.446`
- `ByteTrack + bytetrack_sompt22_tuned.yaml`: `HOTA 44.564`，`MOTA 58.367`，`IDF1 54.786`

因此当前可以这样理解：

- 如果只看当前已经测试过的所有方案，`ByteTrack + bytetrack_sompt22_tuned.yaml` 已经成为 HIE20 上新的 `HOTA / IDF1` 最优方案。
- `BotSort` 仍然保留了更高一点的 `MOTA`，差值为 `0.245`。
- 也就是说，ByteTrack 原来在 HIE20 上落后于 `BotSort`，但换成 tuned 配置以后，这个结论已经发生了变化。

## 7. 主要配置差异速览

为了便于后续继续复验，这里只记录最影响结果的几个配置差异：

| Config | 关键默认值变化 |
| --- | --- |
| `default` | `track_thresh=0.6`，`new_track_thresh=0.6`，`match_thresh=0.9`，`birth_confirm_frames=1`，`lost_reid_enabled=false`，`exit_zone_enabled=false`，`zombie_max_history=0` |
| `original` | 相比 `default` 放宽到 `track_thresh=0.5`、`new_track_thresh=0.5`、`match_thresh=0.8`，但仍未开启 `lost_reid` / `zombie` / `exit_zone` 等增强机制，同时关闭 `adaptive_zone` 与 `spatial_prior` |
| `improved` | 开启 `exit_zone`、`lost_reid`、`spatial_prior_region`、`zombie`，并设置 `birth_confirm_frames=2`、`entry_margin=50`、`zombie_max_history=100`、`zombie_dist_thresh=150` |
| `sompt22_tuned` | 基于 `improved` 进一步调整到 `match_thresh=0.76`、`lost_reid_thresh=0.2`、`zombie_dist_thresh=130`、`zombie_match_cost_thresh=0.38`、`zombie_reid_thresh=0.3` |
| `dual_tuned` | 基于 `improved` 调整到 `match_thresh=0.78`、`birth_suppress_center_dist=25`、`lost_match_max_dist=100`，更偏向跨数据集折中 |

## 8. 复现方式

默认配置：

```bash
uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  bytetrack \
  --source HIE20 \
  --classes 0 \
  --device 0 \
  --project /root/autodl-tmp/boxmot_demo2/runs_hie20_bytetrack_cfgs_20260320_025718 \
  --exist-ok \
  --verbose
```

指定配置文件：

```bash
uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  bytetrack \
  --source HIE20 \
  --classes 0 \
  --tracker-config boxmot/configs/trackers/<config>.yaml \
  --device 0 \
  --project /root/autodl-tmp/boxmot_demo2/runs_hie20_bytetrack_cfgs_20260320_025718 \
  --exist-ok \
  --verbose
```

说明：

- `<config>.yaml` 可替换为 `bytetrack_original.yaml`、`bytetrack_improved.yaml`、`bytetrack_sompt22_tuned.yaml`、`bytetrack_dual_tuned.yaml`
- 因为 `dets_n_embs` 已经复用，所以同 detector / ReID / classes 口径下重新跑不同 ByteTrack 配置时，不需要重新提取检测和外观特征

## 9. 产物路径

- 配置专项运行根目录：`runs_hie20_bytetrack_cfgs_20260320_025718`
- 日志目录：`runs_hie20_bytetrack_cfgs_20260320_025718/logs`
- 结果目录：`runs_hie20_bytetrack_cfgs_20260320_025718/mot`
- 复用缓存链接：`runs_hie20_bytetrack_cfgs_20260320_025718/dets_n_embs`
- 缓存真实位置：`runs_hie20_tracker_compare_20260320_014617/dets_n_embs`
- 本文档：`project_notes/HIE20_ByteTrack_config_comparison_2026-03-20.md`
