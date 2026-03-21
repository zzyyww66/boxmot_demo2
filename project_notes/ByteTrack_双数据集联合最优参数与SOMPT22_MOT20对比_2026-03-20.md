# ByteTrack 双数据集联合最优参数与 SOMPT22 / MOT20 对比（2026-03-20）

## 1. 结论先行

- 当前用于 **SOMPT22 + MOT20 双数据集统一部署** 的最优 ByteTrack 参数配置已经确定，对应配置文件为 `boxmot/configs/trackers/bytetrack_dual_tuned.yaml`。
- 这组参数来自联合调参全流程最终胜出配置 `trial_028_full_from_trial_024`，输出根目录为 `runs_tune_joint_bytetrack_full_20260319_132058`。
- 相比当前默认改进版 `boxmot/configs/trackers/bytetrack_improved.yaml`，它在两个数据集上都实现了 **HOTA / MOTA / IDF1 同时提升**：
  - SOMPT22: `HOTA +0.694`，`MOTA +0.502`，`IDF1 +1.136`
  - MOT20: `HOTA +0.482`，`MOTA +0.455`，`IDF1 +0.595`
- 这组参数不是“单独对 SOMPT22 打榜最强”的参数，但它是当前 **跨 SOMPT22 与 MOT20 的统一最优折中配置**：
  - 在 SOMPT22 上，`ByteTrack-DualTuned` 的 `HOTA / MOTA` 略低于 `ByteTrack-Tuned`（SOMPT22 单集调优版），但 `IDF1` 更高。
  - 在 MOT20 上，`ByteTrack-DualTuned` 拿到了当前对比表中的 **最高 HOTA** 和 **最高 IDF1**。

## 2. 评测与调参口径

- 调参运行目录: `runs_tune_joint_bytetrack_full_20260319_132058`
- 最终汇总文件: `runs_tune_joint_bytetrack_full_20260319_132058/best_summary.json`
- 联合排行榜: `runs_tune_joint_bytetrack_full_20260319_132058/leaderboard.md`
- 检测器固定为 `yolov8m_pretrain_crowdhuman.pt`
- ReID 固定为 `osnet_x0_25_msmt17.pt`
- 类别过滤固定为 `--classes 0`，即仅 `person`
- 对比中其他算法的结果，复用了此前已经确认过的同口径对比文档：
  - SOMPT22: `project_notes/SOMPT22_所有关联算法详细对比_2026-03-18.md`
  - MOT20: `project_notes/MOT20_所有算法复用缓存评测结果_2026-03-19.md`
- 本文中的 `ByteTrack-DualTuned` 行，新增自联合调参最终结果；其他算法行沿用上述两份文档中的同口径结果。

## 3. 最终最优参数

相对 `boxmot/configs/trackers/bytetrack_improved.yaml`，联合最优配置只改了 3 个默认值，其余参数全部保持一致：

| 参数 | `bytetrack_improved.yaml` | `bytetrack_dual_tuned.yaml` | 作用理解 |
| --- | ---: | ---: | --- |
| `match_thresh.default` | 0.80 | **0.78** | 轻微放宽高分匹配阈值，提升跨数据集匹配鲁棒性 |
| `birth_suppress_center_dist.default` | 35 | **25** | 更严格抑制空间上过近的新生轨迹，减少重复出生 |
| `lost_match_max_dist.default` | 120 | **100** | 收紧 recent-lost 恢复的空间上限，降低错误拉回 |

对应配置文件：

- 新配置: `boxmot/configs/trackers/bytetrack_dual_tuned.yaml`
- 调参产物原始导出: `runs_tune_joint_bytetrack_full_20260319_132058/bytetrack_dual_tuned.yaml`

## 4. 相对默认改进版的最终收益

| Dataset | Config | HOTA | MOTA | IDF1 | IDSW |
| --- | --- | ---: | ---: | ---: | ---: |
| SOMPT22 | ByteTrack-Imp | 53.549 | 66.013 | 67.014 | 747 |
| SOMPT22 | ByteTrack-DualTuned | **54.243** | **66.515** | **68.150** | 812 |
| MOT20 | ByteTrack-Imp | 48.944 | 62.464 | 62.405 | 2,322 |
| MOT20 | ByteTrack-DualTuned | **49.426** | **62.919** | **63.000** | 2,374 |

增量解释：

- SOMPT22: `HOTA +0.694`，`MOTA +0.502`，`IDF1 +1.136`，`IDSW +65`
- MOT20: `HOTA +0.482`，`MOTA +0.455`，`IDF1 +0.595`，`IDSW +52`
- 也就是说，这个联合最优配置的核心价值，是 **在两个数据集上同时把综合精度推高**，代价是 `IDSW` 相比默认改进版略有增加。

## 5. SOMPT22 对比表（加入联合最优参数）

说明：本表只保留最核心的 `HOTA / MOTA / IDF1 / IDSW`，更完整的 `RHOTA / AssA / FPS / CLR` 等指标仍可回看 `project_notes/SOMPT22_所有关联算法详细对比_2026-03-18.md`。

| Tracker | Config | HOTA↑ | MOTA↑ | IDF1↑ | IDSW↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| ByteTrack-Tuned | `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml` | **54.313** | **66.681** | 68.008 | 797 |
| ByteTrack-DualTuned | `boxmot/configs/trackers/bytetrack_dual_tuned.yaml` | 54.243 | 66.515 | **68.150** | 812 |
| BoostTrack | default | 53.711 | 62.541 | 67.414 | 993 |
| ByteTrack-Imp | `boxmot/configs/trackers/bytetrack_improved.yaml` | 53.549 | 66.013 | 67.014 | **747** |
| BotSort | default | 53.459 | 66.121 | 65.905 | 939 |
| StrongSort | default | 52.284 | 63.110 | 64.650 | 1,540 |
| DeepOcSort | default | 52.263 | 63.099 | 65.119 | 1,220 |
| OcSort | default | 51.130 | 61.605 | 63.875 | 1,058 |
| ByteTrack-Orig | `boxmot/configs/trackers/bytetrack_original.yaml` | 50.784 | 63.558 | 59.642 | 1,665 |
| HybridSort | default | 49.527 | 60.236 | 60.080 | 2,176 |

SOMPT22 结论：

- 如果只看 SOMPT22 单榜最优，`ByteTrack-Tuned` 仍以 `HOTA 54.313 / MOTA 66.681` 略占上风。
- 但 `ByteTrack-DualTuned` 把 `IDF1` 进一步推到 **68.150**，成为本表中当前最高的身份一致性得分。
- 因此对于需要同时兼顾 MOT20 泛化的统一配置，`ByteTrack-DualTuned` 比单数据集 tuned 版更适合作为项目默认参数。

## 6. MOT20 对比表（加入联合最优参数）

说明：本表同样聚焦核心主指标；更完整的 `RHOTA / AssA / Assoc FPS / CLEAR / Identity` 指标见 `project_notes/MOT20_所有算法复用缓存评测结果_2026-03-19.md`。

| Tracker | Config | HOTA↑ | MOTA↑ | IDF1↑ | IDSW↓ |
| --- | --- | ---: | ---: | ---: | ---: |
| ByteTrack-DualTuned | `boxmot/configs/trackers/bytetrack_dual_tuned.yaml` | **49.426** | 62.919 | **63.000** | 2,374 |
| ByteTrack-Imp | `boxmot/configs/trackers/bytetrack_improved.yaml` | 48.944 | 62.464 | 62.405 | 2,322 |
| BotSort | default | 48.653 | 61.775 | 61.114 | 2,770 |
| ByteTrack-Tuned | `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml` | 48.546 | 62.072 | 61.468 | 2,590 |
| BoostTrack | default | 46.867 | 51.857 | 61.617 | **2,152** |
| DeepOcSort | default | 46.489 | 53.494 | 60.033 | 2,726 |
| ByteTrack-Orig | `boxmot/configs/trackers/bytetrack_original.yaml` | 45.661 | 63.432 | 54.024 | 5,295 |
| StrongSort | default | 43.855 | 52.558 | 55.749 | 4,772 |
| OcSort | default | 42.656 | 48.071 | 55.160 | 2,592 |
| HybridSort | default | 36.076 | **64.166** | 38.471 | 10,876 |

MOT20 结论：

- `ByteTrack-DualTuned` 已经成为当前 MOT20 对比表中的 **HOTA 第一** 与 **IDF1 第一**。
- 相比 `ByteTrack-Imp`，它把 `HOTA` 从 `48.944` 提升到 `49.426`，把 `IDF1` 从 `62.405` 提升到 `63.000`。
- `HybridSort` 虽然 `MOTA` 最高，但它的 `HOTA / IDF1 / IDSW` 明显更差，因此并不是更好的综合方案。
- `BoostTrack` 的 `IDSW` 更低，但它在 `MOTA / HOTA` 上明显落后，不适合作为当前项目的统一主配置。

## 7. 为什么最终推荐这组联合参数

- 联合目标最优分数: `1114.690`
- 相比 baseline full（`bytetrack_improved.yaml`）：
  - `min_delta_hota = +0.482`
  - `avg_delta_hota = +0.588`
  - `min_delta_idf1 = +0.595`
  - `avg_delta_idf1 = +0.866`
  - `avg_delta_mota = +0.478`
- 这说明它不是靠“牺牲一个数据集换另一个数据集”得到最优，而是在 **两个数据集同时正向提升** 的前提下胜出。
- 从工程角度，这比为每个数据集单独维护一套参数更适合当前项目后续复现实验、统一部署与持续横向比较。

## 8. 推荐使用方式

```bash
uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  bytetrack \
  --tracker-config boxmot/configs/trackers/bytetrack_dual_tuned.yaml \
  --classes 0 \
  --source <dataset_root> \
  --device 0 \
  --project <run_root> \
  --exist-ok \
  --verbose
```

如果后续目标改变，可以这样理解选择：

- 想要 **双数据集统一最优配置**：优先 `boxmot/configs/trackers/bytetrack_dual_tuned.yaml`
- 想要 **SOMPT22 单数据集极限最优**：优先 `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml`
- 想要 **默认稳健、IDSW 更低的改进版基线**：使用 `boxmot/configs/trackers/bytetrack_improved.yaml`

## 9. 相关产物路径

- 联合调参根目录: `runs_tune_joint_bytetrack_full_20260319_132058`
- 联合最优汇总: `runs_tune_joint_bytetrack_full_20260319_132058/best_summary.json`
- 联合排行榜: `runs_tune_joint_bytetrack_full_20260319_132058/leaderboard.md`
- 联合最优原始导出配置: `runs_tune_joint_bytetrack_full_20260319_132058/bytetrack_dual_tuned.yaml`
- 仓库内落地配置: `boxmot/configs/trackers/bytetrack_dual_tuned.yaml`
- SOMPT22 对比原文: `project_notes/SOMPT22_所有关联算法详细对比_2026-03-18.md`
- MOT20 对比原文: `project_notes/MOT20_所有算法复用缓存评测结果_2026-03-19.md`
