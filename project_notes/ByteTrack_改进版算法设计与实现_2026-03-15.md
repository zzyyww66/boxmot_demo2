# ByteTrack 改进版算法设计与实现说明

更新时间: 2026-03-15

适用代码基线:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/trackers/bytetrack/spatial_prior.py`
- `boxmot/trackers/bytetrack/basetrack.py`
- `boxmot/configs/trackers/bytetrack_improved.yaml`
- `boxmot/trackers/tracker_zoo.py`
- `tests/unit/test_trackers.py`

本文档只描述当前仓库里真实运行的 ByteTrack 改进实现，不沿用旧文档里的旧命名和旧结构假设。当前实现仍然挂在 `ByteTrack` 追踪器中，通过配置项启用各个增强模块。

## 1. 两个核心先验

### 1.1 先验一: 边缘离开倾向与中心非新生倾向

在固定城市监控场景中，下面两个现象通常同时成立，但都只是统计倾向，不是绝对规则:

- 一个行人在画面边缘附近消失，更大概率是真正离开监控视野，而不是即将在长时间后重新回到画面内部。
- 一个未匹配检测如果首次出现在长期稳定的画面中心区域，它更大概率不是“真正的新生目标”，而是旧轨迹的断检回归、遮挡后重现，或者此前生命周期管理过于激进导致的 ID 断裂。

更正式地说，当前算法假设:

- `P(permanent_exit | track_lost_near_boundary)` 通常高于 `P(recover_later_inside_scene | track_lost_near_boundary)`
- `P(true_birth | unmatched_detection_in_long_term_core)` 通常低于 `P(true_birth | unmatched_detection_in_entry_region)`

这里必须强调“通常”而不是“总是”。因此实现上采用的是**偏置型生命周期策略**，而不是绝对硬编码规则:

- 边缘丢失轨迹会被优先按“离开画面”解释，并进入 exit-zone grace + remove 流程。
- 中心区域未匹配高分检测会被优先按“旧 ID 可复活”解释，先尝试 zombie rescue，再决定是否允许新生。
- 这个偏置可以被显式放松:
  - `strict_entry_gate=False` 时，中心区在复活失败后仍然允许创建新 ID。
  - `adaptive_zone` 的 outside-before-expand 放行逻辑会允许“刚刚扩展到的新可见区域”立即新生。
  - `spatial prior` 学到的 entry hotspot 也可能出现在画面中央附近，因此“中心不新生”并不是几何意义上的绝对中心封锁，而是对“长期核心区”的统计性约束。

### 1.2 先验二: 新生位置与持久遮挡结构的长时稳定性

在固定城市监控场景中，行人的 birth 事件空间位置，以及长期存在的遮挡物对可见区域造成的空间结构，在长时间尺度上通常保持基本稳定。

更正式地说，当前算法假设:

- 已确认轨迹的脚点支持分布 `support(x, y)` 在长时间尺度上会稳定刻画“人通常出现/经过”的可行走区域。
- 已确认新生事件的脚点分布 `birth(x, y)` 在长时间尺度上会稳定刻画“人通常从哪里进入或重新可见”的热点区域。
- 持久遮挡物、道路边界、护栏、树池、站台立柱等静态结构，会通过长期统计把画面自然分解为“高支持核心区”和“高出生热点区”。

这个先验在实现中被编码为一个低分辨率概率场:

- `support_count`: 由稳定轨迹脚点持续累积。
- `birth_count`: 由确认后的 birth 事件脚点累积。
- 当样本量达到门限后，从概率场中显式构造 `entry mask` 和 `core mask`，将纯几何的边缘带门控替换为场景自适应的 entry/core 语义区域。

## 2. 适用范围与目标

当前改进版不是“任何场景都更好”的通用 MOT 改法，它主要针对:

- 固定摄像头
- 城市监控
- 以行人为主
- 中低到中等密度遮挡
- 希望减少 ID 断裂、重复新生和错误复活

优化目标按优先级可以概括为:

1. 在不破坏 ByteTrack 主干稳定性的前提下，降低 `IDs` 和 `IDSW`。
2. 尽量维持或提升 `IDF1 / HOTA / MOTA`。
3. 让“新生”和“离场”两个生命周期事件更符合固定监控场景的真实统计规律。
4. 把外观信息限制在高风险但低频的 zombie rescue 阶段，避免把 ReID 扩散到 Step1/Step2 主干造成回归风险。

## 3. 相对原始 ByteTrack 的总体改动

当前实现保留了 ByteTrack 最核心的两段式关联主干:

1. Step1: 高分检测与 `tracked + lost` 的 IoU 关联。
2. Step2: 低分检测与剩余 `tracked` 的二次 IoU 关联。

在此基础上，增加了四类场景先验驱动模块:

1. 新生控制:
   - `new_track_thresh`
   - `birth_confirm_frames`
   - duplicate suppression
2. 区域化生命周期管理:
   - `entry zone`
   - `exit zone`
   - `adaptive effective zone`
3. 长时记忆与复活:
   - `lost -> zombie` 转移
   - 位置冻结
   - center-zone zombie rescue
4. 长时空间先验学习:
   - `SpatialPriorField`
   - `support/birth` 概率图
   - `entry/core` 显式掩码

因此，当前算法可以概括为:

- 主干关联仍然是 ByteTrack
- 出生与离场使用区域化生命周期先验
- 长时间断链恢复使用 zombie memory + ReID
- 长时固定场景结构由 spatial prior 自动学习

## 4. 算法结构

### 4.1 运行时状态

`ByteTrack` 维护以下几类轨迹集合:

- `active_tracks`: 当前处于 `Tracked` 状态并参与输出的轨迹
- `lost_stracks`: 短时丢失但仍保留在常规恢复窗口内的轨迹
- `zombie_stracks`: 超过常规 lost 窗口但仍保留身份记忆的轨迹
- `removed_stracks`: 已彻底移除的轨迹
- `pending_births`: 待确认的新生候选

### 4.2 单轨迹对象 `STrack`

`STrack` 在原有检测框和 KF 状态之外，额外保存:

- `curr_feat`: 当前帧外观特征
- `smooth_feat`: 指数平滑后的稳定外观特征
- `features`: 历史外观队列
- `spatial_birth_frame`: 轨迹出生帧
- `spatial_birth_point`: 轨迹初始脚点
- `spatial_birth_committed`: 该 birth 是否已经写入概率场
- `lost_frame_id`: 轨迹进入 `Lost` 的帧号
- `frozen_mean`: 长时丢失后冻结的位置状态

辅助接口里最重要的两个是:

- `footpoint()`: 直接从当前状态计算脚点底部中心
- `get_tlwh_for_matching(frame_id, max_predict_frames)`: 在 zombie 阶段优先使用冻结位置或当前 KF 状态做匹配

### 4.3 概率场 `SpatialPriorField`

概率场是一个低分辨率网格统计器，核心状态为:

- `support_count`
- `birth_count`
- `_decay_scale`

它提供两类输出:

- 概率图:
  - `walkable`
  - `birth`
  - `confidence`
- 区域构造用度量图:
  - `support_density`
  - `birth_density`
  - `birth_ratio`

## 5. 每帧更新流程

下面的流程严格对应 `ByteTrack.update()` 当前实现。

### 5.1 输入预处理

输入检测 `dets` 是 `[x1, y1, x2, y2, conf, cls]`，进入 `update()` 后会补上 `det_ind`，内部统一使用:

`[x1, y1, x2, y2, conf, cls, det_ind]`

随后:

- `frame_count += 1`
- 清空 `_outside_zone_det_inds`
- 清理过期 `pending_births`
- 记录图像宽高
- 如果启用 `spatial_prior`，则先 `configure_image()`，再做一次 `step()` 时间衰减

### 5.2 检测划分与特征准备

当前帧检测被拆成两组:

- 高分检测: `conf > track_thresh`
- 次高分检测: `min_conf < conf < track_thresh`

对高分检测:

- 如果 tracker 带 ReID 且没有预先提供 `embs_first`，则在线抽取外观特征
- 每个检测构造为一个 `STrack`

注意:

- 低分检测阶段当前不使用 ReID，只按原始 ByteTrack 方式做 IoU 关联
- ReID 的主设计意图是只服务于 zombie rescue，而不是替换主干 IoU 流程

### 5.3 自适应有效区预更新

如果启用了 `adaptive_zone_enabled`，会在正式关联前先调用:

- `_update_effective_zone(detections, phase="pre")`

在 `always_expand` 模式下，顺序是:

1. 先基于上一帧有效区，标记哪些高分检测在扩展前位于有效区外，记入 `_outside_zone_det_inds`
2. 再根据触发策略扩展 `_effective_zone`

这个顺序很关键，因为它保留了“本帧原本在旧有效区外”的事实，供 Step4 放行真正的新生目标。

### 5.4 Step1: 高分检测主关联

这一步完全保留 ByteTrack 主干逻辑:

1. `strack_pool = tracked_stracks + lost_stracks`
2. 对 `strack_pool` 做 KF 预测
3. 计算 `iou_distance(strack_pool, detections)`
4. 用 `fuse_score()` 融合检测置信度
5. 用 `linear_assignment(..., thresh=match_thresh)` 做全局匹配

匹配成功后:

- `Tracked` 轨迹调用 `update()`
- `Lost` 轨迹调用 `re_activate(new_id=False)`

这一步的设计原则是:

- 短时遮挡恢复仍然优先依赖原始 ByteTrack 的运动 + IoU 主干
- 不把 ReID 扩散进主干，以控制回归风险

### 5.5 Step2: 次高分检测补关联

对 Step1 未匹配的 `Tracked` 轨迹，继续与次高分检测做第二轮 IoU 关联:

- 匹配阈值固定为 `0.5`
- 不使用 ReID

未匹配的 tracked 轨迹会被标记为 `Lost`。如果启用了 `exit_zone_enabled`，在进入 `Lost` 前会先判断其是否落在 exit zone，并设置 `exit_pending`。

### 5.6 Unconfirmed 轨迹处理

只出现过一次、尚未稳定的未确认轨迹，会和剩余高分检测再做一次匹配:

- 成功匹配则转为有效轨迹
- 失败则直接移除

这部分保留了 ByteTrack 对“新生后一帧确认”的原始语义，因此很多测试里会看到:

- 非首帧新轨迹在 `activate()` 当帧未必立刻出现在输出中
- 往往需要下一帧 `update()` 之后才进入最终输出

### 5.7 Step4: 新生管理与 zombie rescue

这是当前改进版最关键的一步，也是与原始 ByteTrack 差异最大的部分。

#### 5.7.1 首帧

第 1 帧没有历史上下文，所有超过 `new_track_thresh` 的未匹配高分检测直接激活为新轨迹，并登记空间出生元数据。

#### 5.7.2 非首帧的未匹配高分检测

对每一个剩余高分检测 `det_track`:

1. 若 `conf < new_track_thresh`，直接忽略，不允许新生。
2. 若 `adaptive_zone` 使用 `unmatched_high` 或 `outside_high`，在 Step4 可继续局部扩展有效区。
3. 若该检测在本帧扩展前位于旧有效区外，即 `det_ind in _outside_zone_det_inds`:
   - 视为“新进入画面”或“新可见区域”优先放行
   - 直接走 `_try_activate_new_track()`
4. 否则计算它是否位于 entry zone:
   - 在 entry zone 内:
     - 直接走新生逻辑
     - `skip_confirmation=True`
     - 即保持 ByteTrack 的即时出生语义
   - 不在 entry zone 内:
     - 先作为 center-zone detection 收集起来
     - 统一进入 zombie rescue

#### 5.7.3 center-zone zombie rescue

所有中心区未匹配高分检测会和 `zombie_stracks` 做一次全局匹配:

- `_match_zombie_tracks(center_zone_detections, zombie_stracks)`
- 内部通过 `_build_zombie_match_cost()` 构造 gated cost matrix
- 再用 Hungarian 做全局最优匹配

匹配成功的 zombie 轨迹会:

- 调用 `re_activate(new_id=False)`
- 回到 `refind_stracks`
- 从 `zombie_stracks` 中移除

匹配失败的中心区检测:

- 如果 `strict_entry_gate=True` 且 `entry_margin > 0`，则禁止新生
- 否则回退到 `_try_activate_new_track()`，允许创建新 ID

换句话说，中心区检测的处理顺序是:

1. 先假设它是旧 ID 回归
2. 只有复活失败时，才考虑它是否允许新生

这正是第一个核心先验的主要落点。

### 5.8 Step5/Step6: Lost、Exit、Zombie 三段式生命周期

Step4 之后，生命周期管理继续分三段进行。

#### 5.8.1 新增 lost

Step2 失败的 tracked 轨迹先进入 `lost_stracks`。

#### 5.8.2 长时丢失位置冻结

如果启用了 zombie 模式且 `zombie_max_predict_frames > 0`，当一条 lost 轨迹丢失时间超过该阈值时:

- 不再无限制依赖 KF 外推
- 把当前位置写入 `frozen_mean`

后续 zombie 匹配时，`get_tlwh_for_matching()` 会优先使用冻结状态，避免长时间漂移后的位置预测失真。

#### 5.8.3 exit-zone 优先删除

对于 `exit_pending=True` 的 lost 轨迹:

- 先经过 `exit_zone_remove_grace` 帧宽限期
- 宽限期结束后直接 `Removed`
- 不再进入 zombie 池

这体现了“边缘消失通常就是离场”的先验。

#### 5.8.4 lost -> zombie 转移

如果未命中 exit-zone 删除逻辑，且 zombie 功能开启:

- 当 `frames_lost >= zombie_transition_frames`
- 该轨迹从 `lost_stracks` 转入 `zombie_stracks`

这部分是对原始 ByteTrack 的核心扩展:

- 原始 ByteTrack 中这类轨迹通常会被直接移除
- 当前实现保留其身份记忆，供后续中心区复活

### 5.9 Step7/Step8/Step9: 历史裁剪与集合清理

末尾会执行:

- 删除已复活的 zombies
- 限制 `zombie_max_history`
- 限制 `lost_max_history`
- `joint_stracks / sub_stracks / remove_duplicate_stracks`

最后更新:

- `active_tracks`
- `lost_stracks`
- `removed_stracks`

并把已激活轨迹输出为:

`[x1, y1, x2, y2, track_id, conf, cls, det_ind]`

### 5.10 Step10: 空间先验学习更新

每帧末尾调用:

- `_update_spatial_prior_tracks()`
- `_update_spatial_prior_stage()`

这里做两件事:

1. 对稳定轨迹写入 support 点
2. 对满足稳定条件的新轨迹，把其初始脚点记为确认 birth 事件

## 6. 关键子模块实现细节

### 6.1 新生门控 `_try_activate_new_track`

这个函数统一处理“是否真的要创建新 ID”。

它包含两层控制:

1. duplicate suppression
2. temporal confirmation

#### 6.1.1 duplicate suppression

参考轨迹集合会从下面几组中去重收集:

- `active_tracks`
- `lost_stracks`
- `zombie_stracks`
- 本帧已激活和已复活轨迹

随后按配置做两种抑制:

- IoU 抑制: `birth_suppress_iou`
- 中心距离抑制: `birth_suppress_center_dist`

只要与已有轨迹太近，就不允许再创建一个新 ID。

#### 6.1.2 temporal confirmation

当 `birth_confirm_frames > 1` 时，新生不是立刻发生，而是先进入 `pending_births`:

- 首次命中: 建立 pending，`hits=1`
- 后续帧如果与 pending 候选足够接近:
  - `hits += 1`
  - 达到 `birth_confirm_frames` 后才真正 `activate()`
- 如果连续漏掉过多帧:
  - `_prune_pending_births()` 会将其删除

需要注意:

- entry-zone births 当前使用 `skip_confirmation=True`
- 这意味着边缘进入仍然保持更敏捷的原始 ByteTrack 语义
- center-zone fallback births 才更依赖多帧确认与 duplicate suppression

### 6.2 zombie rescue 代价构造

`_build_zombie_match_cost()` 的策略是:

1. 先做硬门控
2. 再做代价融合
3. 最后做 Hungarian 全局匹配

#### 6.2.1 硬门控

一个 zombie-detection 配对只有同时满足以下条件才进入代价计算:

- 检测框面积不小于 `zombie_reid_min_box_area`，否则直接跳过 ReID 复活
- 中心距离 `<= max_dist`
- 宽高比例差不超过 `zombie_shape_max_ratio`
- 如果启用 ReID:
  - `reid_cost <= zombie_reid_thresh`

任何一个条件不满足，该配对的代价都会被置为无效大值，禁止匹配。

#### 6.2.2 代价融合

如果 ReID 可用，总代价为:

`total_cost = (w_r * reid_cost + w_m * motion_cost + w_s * shape_cost) / (w_r + w_m + w_s)`

其中默认权重为:

- `w_r = 0.75`
- `w_m = 0.20`
- `w_s = 0.05`

如果 ReID 不可用，则退化为 motion + shape 的归一化加权代价。

各项定义:

- `reid_cost = 1 - cosine_similarity`
- `motion_cost = min(1, center_distance / max_dist)`
- `shape_cost = 0.5 * (|log(width_ratio)| + |log(height_ratio)|)`，再裁剪到 `[0, 1]`

实现含义很明确:

- ReID 主导最终选择
- motion 与 shape 只负责把明显不合理的候选压掉，并在近邻冲突时提供轻量辅助

### 6.3 自适应有效区 `adaptive_zone`

当前支持两种模式:

- `warmup_once`
- `always_expand`

当前默认更偏向 `always_expand`，原因是它更适合长期运行的固定监控:

- 初期没有标注 ROI 时，也能逐步形成有效活动区
- 后续只扩不缩，避免区域抖动

关键工程点是 `_outside_zone_det_inds`:

- 它记录的是“本帧扩展前位于旧有效区外”的检测
- 这类检测在 Step4 会被优先允许新生
- 从而避免“先扩展后门控”导致真正新进入目标被误判为中心区复活对象

### 6.4 概率场学习与显式 entry/core 区域

#### 6.4.1 support 写入

对每条处于 `Tracked` 状态的活跃轨迹:

- 当年龄达到 `spatial_prior_support_min_age`
- 将当前 `footpoint()` 写入 `support_count`

support 的含义不是“出生”，而是“稳定存在/经过的可行走支持证据”。

#### 6.4.2 birth 写入

新轨迹在激活时只登记 `spatial_birth_point`，并不会立即写入 `birth_count`。

只有当轨迹满足以下至少一个稳定条件时，才提交 birth:

- `age >= spatial_prior_birth_commit_age`
- `hits >= spatial_prior_birth_commit_hits`

这一步的目的是避免把单帧误检或短命轨迹错误写成长期 birth hotspot。

#### 6.4.3 概率场成熟阶段

当前只定义了一个关键成熟阶段:

- `learn_only`
- `entry_only`

当以下两个计数都达到阈值时:

- `spatial_prior_support_samples >= spatial_prior_entry_support_threshold`
- `spatial_prior_birth_events >= spatial_prior_entry_birth_threshold`

系统进入 `entry_only` 阶段，允许用概率场显式构建 `entry/core` 区域。

#### 6.4.4 entry/core mask 构造

`_refresh_spatial_region_masks()` 的逻辑可以概括为:

1. 从概率场取出:
   - `confidence`
   - `walkable`
   - `birth_density`
2. 用 `confidence` 和 `walkable` 得到 `eligible` 可行走区域
3. 对 `eligible` 做腐蚀，形成 outer band 候选
4. 在 outer band 内按 `birth_density` 的高分位数选 seed
5. 再按较低分位数选 candidate
6. 用连通域 + 有限步数生长，得到 entry 区域
7. `core = eligible - entry`

因此当前的 entry/core 不是简单的“离边缘近就是 entry”，而是:

- 先有长期支持区域
- 再在其中找长期稳定的出生热点
- 最终得到场景自适应的 entry/core 划分

这正是第二个核心先验的主要实现落点。

### 6.5 概率场优先级高于几何 entry margin

`_is_in_entry_zone()` 的判定顺序是:

1. 如果 `spatial_entry_mask / spatial_core_mask` 已经就绪，优先使用概率场区域标签
2. 否则退回:
   - 固定 `entry_margin`
   - 或 `adaptive effective zone + margin`

这意味着:

- 在概率场成熟后，entry/core 的最终语义来自长期学习的场景统计，而不是固定边缘带
- 即使几何上接近中心，只要它长期是 birth hotspot，也可以被当作 entry

## 7. 当前默认参数要点

以下是 `bytetrack_improved.yaml` 中与当前设计最相关的一组默认值:

- `track_thresh = 0.5`
- `new_track_thresh = 0.65`
- `birth_confirm_frames = 2`
- `birth_suppress_iou = 0.7`
- `birth_suppress_center_dist = 35`
- `entry_margin = 50`
- `strict_entry_gate = false`
- `exit_zone_enabled = true`
- `exit_zone_margin = 50`
- `exit_zone_remove_grace = 30`
- `adaptive_zone_enabled = true`
- `adaptive_zone_update_mode = always_expand`
- `adaptive_zone_expand_trigger = all_high`
- `zombie_transition_frames = 30`
- `zombie_max_history = 100`
- `zombie_match_max_dist = 200`
- `zombie_dist_thresh = 150`
- `zombie_max_predict_frames = 5`
- `zombie_reid_enabled = true`
- `zombie_reid_weight = 0.75`
- `zombie_motion_weight = 0.20`
- `zombie_shape_weight = 0.05`
- `zombie_reid_thresh = 0.35`
- `zombie_match_cost_thresh = 0.45`
- `spatial_prior_enabled = true`
- `spatial_prior_decay = 0.999`
- `spatial_prior_region_enabled = true`
- `spatial_prior_region_birth = 0.85`
- `spatial_prior_region_birth_grow = 0.6`
- `spatial_prior_entry_support_threshold = 100`
- `spatial_prior_entry_birth_threshold = 8`

这些默认值组合出来的策略可以概括为:

- 主干关联尽量保持 ByteTrack 原样
- 新生比原始 ByteTrack 更保守
- 中心区优先复活旧 ID
- 边缘区更容易即时新生，也更容易被解释为离场
- 概率场在积累足够统计量后，逐步接管 entry/core 语义

## 8. 当前实现与原始 ByteTrack 的本质差异

如果只抓核心差异，可以总结为五条:

1. 原始 ByteTrack 把未匹配高分检测几乎都视为潜在新生，当前实现会先按区域解释它们的生命周期含义。
2. 原始 ByteTrack 的长时 lost 轨迹会被移除，当前实现会把它们转成 zombie memory，供中心区复活。
3. 原始 ByteTrack 没有 exit-zone 偏置，当前实现会把边缘丢失更偏向解释为真实离场。
4. 原始 ByteTrack 的 entry 语义主要是几何上的边缘带，当前实现可以用长期学习到的概率场显式定义 entry/core。
5. 原始 ByteTrack 不区分“短时恢复”和“长时复活”的代价结构，当前实现把 ReID 限定到 zombie rescue 阶段，形成更保守的分层关联策略。

## 9. 测试覆盖与实现对应关系

`tests/unit/test_trackers.py` 中已经覆盖了当前设计的关键语义:

- `zombie_reid_global_assignment_prefers_appearance`
  - 验证 zombie rescue 在多候选冲突时由外观主导全局匹配
- `zombie_reid_gate_blocks_wrong_appearance_rescue`
  - 验证外观门控能阻断错误复活
- `birth_confirm_requires_two_hits`
  - 验证多帧新生确认
- `entry_zone_birth_skips_pending_confirmation`
  - 验证 entry-zone birth 保持即时激活语义
- `exit_zone_remove_grace_delays_removal`
  - 验证 exit-zone grace + remove
- `adaptive_zone_always_expand_grows_monotonically`
  - 验证有效区只扩不缩
- `outside_before_expand_keeps_new_id_creation`
  - 验证 outside-before-expand 放行机制
- `spatial_prior_*`
  - 验证 support/birth 提交、lazy decay、entry/core mask 构造与中心区覆盖/阻断逻辑

## 10. 一句话总结当前算法

当前实现不是“用 ReID 替换 ByteTrack”，而是:

在保留 ByteTrack 两段式 IoU 主干的前提下，利用固定监控场景中的两类长时稳定先验，对新生、离场和长时断链复活进行区域化、分层化、保守化管理。
