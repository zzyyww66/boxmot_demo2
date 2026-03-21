# ByteTrack SOMPT22 最优参数逐项对照

更新时间: 2026-03-18

## 1. 固定版本与来源

- 固定配置文件: `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml`
- 对照基线配置: `boxmot/configs/trackers/bytetrack_improved.yaml`
- 最优试验来源: `runs_tune_sompt22_bytetrack_20260317_142900/trial_configs/trial_070_full_from_trial_060.yaml`
- 调参汇总文件: `runs_tune_sompt22_bytetrack_20260317_142900/best_summary.json`
- 说明: 当前固定版配置已和最优 full run 试验配置逐项校验一致。
- 参数结论: 共 67 个参数中，仅 6 个默认值发生变化，其余 61 个保持与 `bytetrack_improved.yaml` 一致。

## 2. 当前固定最优结果

- 最优 full trial: `trial_070` / `full_from_trial_060`
- 最优配置输出: `/root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml`
- 最优实验目录: `/root/autodl-tmp/boxmot_demo2/runs_tune_sompt22_bytetrack_20260317_142900/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack_72`
- 本轮系统调参总 trial 数: `74`

### 2.1 Headline 指标对照

| Metric | Baseline | 当前固定最优 | Delta | 说明 |
| --- | ---: | ---: | ---: | --- |
| HOTA | 53.549 | 54.313 | +0.764 | 越大越好 |
| MOTA | 66.013 | 66.681 | +0.668 | 越大越好 |
| IDF1 | 67.014 | 68.008 | +0.994 | 越大越好 |
| IDSW | 747 | 797 | +50 | 越小越好 |
| IDs | 766 | 901 | +135 | 供参考，反映输出轨迹 ID 数量 |

> 备注: 当前“最优”是按本轮调参输出的 full run 最优配置固定下来的，综合结果明显提升在 HOTA / MOTA / IDF1；但 `IDSW` 与 `IDs` 并不是单独最优，因此相对 baseline 有所增加。

## 3. 发生变化的 6 个参数

| 参数 | Improved 默认值 | Tuned 默认值 | 变化 | 解释 |
| --- | ---: | ---: | --- | --- |
| `match_thresh` | 0.8 | 0.76 | 下调 | 第一阶段高分框关联的最大成本阈值从 0.80 下调到 0.76，关联门更严格。 |
| `birth_suppress_center_dist` | 35 | 25 | 下调 | 新生轨迹的近邻抑制半径从 35px 缩到 25px，降低了过度抑制造成的漏生。 |
| `zombie_dist_thresh` | 150 | 130 | 下调 | Zombie rescue 的中心距离硬门从 150px 收紧到 130px，先压掉远距离误救。 |
| `zombie_reid_thresh` | 0.35 | 0.3 | 下调 | Zombie rescue 的外观成本阈值从 0.35 下调到 0.30，ReID 放行条件更严格。 |
| `zombie_match_cost_thresh` | 0.45 | 0.38 | 下调 | Zombie rescue 的全局匹配总成本阈值从 0.45 下调到 0.38，最终接受条件更严格。 |
| `lost_reid_thresh` | 0.25 | 0.2 | 下调 | Recent-lost 恢复的外观成本阈值从 0.25 下调到 0.20，短时恢复也更偏保守。 |

## 4. 全量逐参数对照

### 4.1 基础 ByteTrack 关联参数

| 参数 | 类型 | 搜索空间 / 选项 | Improved 默认值 | Tuned 默认值 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| `min_conf` | uniform | `[0.1, 0.3]` | 0.1 | 0.1 | 保持 |
| `track_thresh` | uniform | `[0.4, 0.7]` | 0.5 | 0.5 | 保持 |
| `new_track_thresh` | uniform | `[0.4, 0.9]` | 0.65 | 0.65 | 保持 |
| `track_buffer` | qrandint | `[10, 61, 10]` | 30 | 30 | 保持 |
| **`match_thresh`** | uniform | `[0.7, 0.9]` | 0.8 | 0.76 | 变化 |
| `frame_rate` | choice | `25, 30` | 30 | 30 | 保持 |

### 4.2 出生控制 / Zombie / Lost 恢复参数

| 参数 | 类型 | 搜索空间 / 选项 | Improved 默认值 | Tuned 默认值 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| `entry_margin` | qrandint | `[0, 101, 10]` | 50 | 50 | 保持 |
| `strict_entry_gate` | choice | `true, false` | false | false | 保持 |
| `birth_confirm_frames` | qrandint | `[1, 4, 1]` | 2 | 2 | 保持 |
| `birth_suppress_iou` | uniform | `[0, 0.9]` | 0.7 | 0.7 | 保持 |
| **`birth_suppress_center_dist`** | qrandint | `[0, 101, 5]` | 35 | 25 | 变化 |
| `zombie_max_history` | qrandint | `[50, 201, 10]` | 100 | 100 | 保持 |
| **`zombie_dist_thresh`** | qrandint | `[50, 301, 10]` | 150 | 130 | 变化 |
| `zombie_transition_frames` | qrandint | `[20, 61, 5]` | 30 | 30 | 保持 |
| `lost_max_history` | qrandint | `[0, 401, 20]` | 0 | 0 | 保持 |
| `zombie_match_max_dist` | qrandint | `[100, 301, 20]` | 200 | 200 | 保持 |
| `zombie_max_predict_frames` | qrandint | `[3, 11, 1]` | 5 | 5 | 保持 |
| `zombie_reid_enabled` | choice | `true, false` | true | true | 保持 |
| `zombie_reid_weight` | uniform | `[0.4, 0.95]` | 0.75 | 0.75 | 保持 |
| `zombie_motion_weight` | uniform | `[0, 0.4]` | 0.2 | 0.2 | 保持 |
| `zombie_shape_weight` | uniform | `[0, 0.2]` | 0.05 | 0.05 | 保持 |
| **`zombie_reid_thresh`** | uniform | `[0.1, 0.6]` | 0.35 | 0.3 | 变化 |
| **`zombie_match_cost_thresh`** | uniform | `[0.2, 0.8]` | 0.45 | 0.38 | 变化 |
| `zombie_shape_max_ratio` | uniform | `[1.2, 4]` | 2 | 2 | 保持 |
| `zombie_reid_min_box_area` | qrandint | `[0, 4097, 256]` | 1024 | 1024 | 保持 |
| `lost_reid_enabled` | choice | `true, false` | true | true | 保持 |
| `lost_match_max_dist` | qrandint | `[60, 241, 10]` | 120 | 120 | 保持 |
| `lost_reid_max_frames` | qrandint | `[3, 31, 1]` | 15 | 15 | 保持 |
| `lost_reid_weight` | uniform | `[0.4, 0.9]` | 0.7 | 0.7 | 保持 |
| `lost_motion_weight` | uniform | `[0.05, 0.4]` | 0.25 | 0.25 | 保持 |
| `lost_shape_weight` | uniform | `[0, 0.2]` | 0.05 | 0.05 | 保持 |
| **`lost_reid_thresh`** | uniform | `[0.1, 0.5]` | 0.25 | 0.2 | 变化 |
| `lost_match_cost_thresh` | uniform | `[0.15, 0.6]` | 0.35 | 0.35 | 保持 |
| `lost_shape_max_ratio` | uniform | `[1.2, 3]` | 1.8 | 1.8 | 保持 |
| `lost_reid_min_box_area` | qrandint | `[0, 4097, 256]` | 1024 | 1024 | 保持 |

### 4.3 退出区参数

| 参数 | 类型 | 搜索空间 / 选项 | Improved 默认值 | Tuned 默认值 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| `exit_zone_enabled` | choice | `true, false` | true | true | 保持 |
| `exit_zone_margin` | qrandint | `[0, 101, 10]` | 50 | 50 | 保持 |
| `exit_zone_remove_grace` | qrandint | `[1, 31, 1]` | 30 | 30 | 保持 |

### 4.4 自适应有效区参数

| 参数 | 类型 | 搜索空间 / 选项 | Improved 默认值 | Tuned 默认值 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| `adaptive_zone_enabled` | choice | `true, false` | true | true | 保持 |
| `adaptive_zone_update_mode` | choice | `always_expand, warmup_once` | warmup_once | warmup_once | 保持 |
| `adaptive_zone_expand_trigger` | choice | `all_high, outside_high, unmatched_high` | outside_high | outside_high | 保持 |
| `adaptive_zone_entry_mode` | choice | `outside_only, margin_inside` | outside_only | outside_only | 保持 |
| `adaptive_zone_warmup` | qrandint | `[5, 31, 1]` | 10 | 10 | 保持 |
| `adaptive_zone_margin` | qrandint | `[20, 101, 10]` | 50 | 50 | 保持 |
| `adaptive_zone_padding` | uniform | `[1, 1.5]` | 1.2 | 1.2 | 保持 |
| `adaptive_zone_min_box_area` | qrandint | `[0, 5001, 100]` | 0 | 0 | 保持 |

### 4.5 空间先验参数

| 参数 | 类型 | 搜索空间 / 选项 | Improved 默认值 | Tuned 默认值 | 状态 |
| --- | --- | --- | ---: | ---: | --- |
| `spatial_prior_enabled` | choice | `true, false` | true | true | 保持 |
| `spatial_prior_grid_w` | qrandint | `[16, 97, 8]` | 48 | 48 | 保持 |
| `spatial_prior_grid_h` | qrandint | `[9, 65, 6]` | 27 | 27 | 保持 |
| `spatial_prior_sigma` | uniform | `[0.5, 3.5]` | 1.5 | 1.5 | 保持 |
| `spatial_prior_decay` | uniform | `[0.95, 1]` | 0.999 | 0.999 | 保持 |
| `spatial_prior_birth_commit_age` | qrandint | `[1, 11, 1]` | 3 | 3 | 保持 |
| `spatial_prior_birth_commit_hits` | qrandint | `[1, 11, 1]` | 3 | 3 | 保持 |
| `spatial_prior_support_min_age` | qrandint | `[1, 11, 1]` | 2 | 2 | 保持 |
| `spatial_prior_entry_mode` | choice | `bias_only, strict_region` | bias_only | bias_only | 保持 |
| `spatial_prior_recovery_cooldown` | qrandint | `[0, 21, 1]` | 5 | 5 | 保持 |
| `spatial_prior_region_enabled` | choice | `true, false` | true | true | 保持 |
| `spatial_prior_region_conf` | uniform | `[0.1, 10]` | 1 | 1 | 保持 |
| `spatial_prior_region_walk` | uniform | `[0, 1]` | 0.05 | 0.05 | 保持 |
| `spatial_prior_region_birth` | uniform | `[0.05, 1]` | 0.85 | 0.85 | 保持 |
| `spatial_prior_region_birth_grow` | uniform | `[0, 1]` | 0.6 | 0.6 | 保持 |
| `spatial_prior_region_grow_max_steps` | qrandint | `[0, 9, 1]` | 3 | 3 | 保持 |
| `spatial_prior_region_component_mean_ratio` | uniform | `[0, 1]` | 0.45 | 0.45 | 保持 |
| `spatial_prior_region_component_max_area` | qrandint | `[0, 257, 8]` | 0 | 0 | 保持 |
| `spatial_prior_entry_band_radius` | qrandint | `[0, 7, 1]` | 2 | 2 | 保持 |
| `spatial_prior_entry_support_threshold` | qrandint | `[20, 401, 20]` | 100 | 100 | 保持 |
| `spatial_prior_entry_birth_threshold` | qrandint | `[2, 31, 1]` | 8 | 8 | 保持 |

## 5. 推荐使用方式

- 复用当前 SOMPT22 人类类别 dets/embs 缓存，建议固定使用下面这份 tuned 配置。
- 保持 detector / reid / classes 不变，只替换 tracker config。

```bash
CACHE_ROOT=/root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17

uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  bytetrack \
  --source /root/autodl-tmp/boxmot_demo2/train \
  --classes 0 \
  --tracker-config /root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml \
  --device 0 \
  --project "$CACHE_ROOT" \
  --exist-ok \
  --verbose
```

## 6. 校验结论

- 已校验 `bytetrack_sompt22_tuned.yaml` 与 `trial_070_full_from_trial_060.yaml` 完全一致。
- 已校验相对 `bytetrack_improved.yaml` 仅有 6 个默认参数变化。
- 因此这份配置可以作为当前 SOMPT22 固定最优参数版本继续复用。
