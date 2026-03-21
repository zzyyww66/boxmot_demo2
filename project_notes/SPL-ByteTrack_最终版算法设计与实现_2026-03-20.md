# SPL-ByteTrack 最终版算法设计与实现（2026-03-20）

## 1. 文档定位与命名说明

本文不是对 2026-03-15 旧方案的修补说明，而是基于当前仓库代码重新整理的一份**现行实现版**技术文档。

为了便于表述，本文将仓库中当前这套最终版改进 ByteTrack 统称为 **SPL-ByteTrack**：

- `S` = `Spatial`
- `P` = `Prior`
- `L` = `Lifecycle-aware`

也就是说，**SPL-ByteTrack 不是代码里的另一个类名**，代码中的实际 tracker 仍然是 `boxmot/trackers/bytetrack/bytetrack.py` 里的 `ByteTrack`，只是它已经不再是“原始 ByteTrack”，而是一套融合了：

1. 场景空间先验学习（Spatial Prior）
2. 分层生命周期管理（Tracked / Lost / Zombie / Pending Birth）
3. 受控的新生门控（birth gating）
4. recent-lost 恢复
5. zombie 复活
6. adaptive effective zone
7. exit zone 延迟移除
8. 几何缓存与向量化代价构造

的综合版本。

本文描述的核心代码来源于以下文件：

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/trackers/bytetrack/spatial_prior.py`
- `boxmot/trackers/bytetrack/basetrack.py`
- `boxmot/trackers/tracker_zoo.py`
- `boxmot/engine/evaluator.py`
- `tests/unit/test_trackers.py`
- `boxmot/configs/trackers/bytetrack_improved.yaml`
- `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml`
- `boxmot/configs/trackers/bytetrack_dual_tuned.yaml`

因此，本文的结论以**当前代码事实**为准，而不是以历史文档、旧实验计划或旧版本实现为准。

---

## 2. 一句话概括当前算法

当前版 SPL-ByteTrack 的本质可以概括为：

> 在保持 ByteTrack 主干高分 / 低分两阶段 IoU 关联框架不变的前提下，把“新 ID 该不该生、旧 ID 该不该救、哪些位置更可信、哪些轨迹该尽快删、哪些轨迹值得长期记忆”全部纳入一个统一的生命周期控制系统里。

它不是简单地“给 ByteTrack 加 ReID”，而是把整个关联流程拆成了**分层决策**：

- 主干关联仍然优先依赖 ByteTrack 最稳定的 IoU 流程；
- 新生（birth）不再无条件开放，而要经过 entry / effective zone / spatial prior / confirmation / duplicate suppression 多级过滤；
- 短时断链恢复前移到 `recent-lost` 阶段；
- 长时断链恢复后移到 `zombie rescue` 阶段；
- 场景固定结构不再靠人工规则硬编码，而是通过 `SpatialPriorField` 被动学习；
- 删除策略不再只看时间，而会结合 `exit zone` 和 `zombie history` 来区分“真正离场”和“可能还会回来”。

因此，SPL-ByteTrack 是一套**保守地主干、前移短时恢复、后置长时复活、对新生做强约束、对场景做被动学习**的 ByteTrack 改进体系。

---

## 3. 相对原始 ByteTrack 的根本变化

与原始 ByteTrack 相比，当前实现最重要的差异不是某一个参数，而是下面 8 个结构性变化：

### 3.1 新生不再“检测够高就立刻生”

原始 ByteTrack 中，未匹配高分检测往往很容易直接变成新轨迹。当前实现则把“新生”分成了几类：

- 几何 entry 区新生
- adaptive effective zone 外扩产生的新生
- spatial prior entry hotspot 允许的新生
- 中心区未匹配高分检测在 recover 失败后的新生
- 首帧新生

这些新生在当前实现里**不是同一种置信来源**，因此享受不同策略：

- 有些会跳过确认门（`skip_confirmation=True`）
- 有些必须经过 `birth_confirm_frames`
- 有些即使满足新生条件，也还要过 duplicate suppression

### 3.2 旧 ID 恢复被拆成“短时恢复”和“长时复活”两层

当前实现不再把所有恢复都丢给同一条通道，而是明确分成两层：

1. `recent-lost recovery`
   - 面向近期丢失轨迹
   - 发生在 Step4 的前段
   - 使用 ReID + motion + shape 的 gated global assignment
   - 目标是把本该很快找回的目标尽量前移找回

2. `zombie rescue`
   - 面向已经从 lost 池转入 zombie 池的长期记忆轨迹
   - 同样采用 gated global assignment
   - 但语义上是“长时间断链后的旧 ID 复活”

### 3.3 引入了“Zombie”长期身份记忆池

原始 ByteTrack 中，lost 轨迹超时后会被移除。当前实现则允许：

- lost 轨迹在达到 `zombie_transition_frames` 后转入 `zombie_stracks`
- 转入 zombie 后仍保留身份与外观信息
- 可以在后续帧通过 zombie rescue 被重新 `re_activate`

### 3.4 引入 exit zone，把“离场”与“短暂消失”区分开

如果一条轨迹是在图像边缘消失，它更像是“真的走出画面”而不是“中心区被遮挡”。

因此当前实现中：

- 轨迹在边缘消失会被标记 `exit_pending`
- 在宽限期 `exit_zone_remove_grace` 内仍可保留
- 超过宽限期后优先被删除
- 不再盲目转 zombie

### 3.5 引入 adaptive effective zone，减少“无效区域误生”

当前实现不再只依赖图像边缘矩形 entry zone，而是维护一个**有效活动区**：

- 可通过 `warmup_once` 在预热阶段一次性估计
- 也可通过 `always_expand` 单调外扩
- entry 判定不再只看是否靠边，还会看是否在有效活动区外 / 内边界附近

### 3.6 引入被动学习的 spatial prior

`SpatialPriorField` 学习的是固定机位下两类统计：

- support：目标稳定活动过的地方
- birth：目标真实诞生过的地方

基于这两个统计，系统可以把场景划成：

- `entry region`
- `core region`

它不是直接替代矩形 entry gate，而是在当前默认配置 `spatial_prior_entry_mode=bias_only` 下，作为一种**保守加权偏置**来补充几何规则。

### 3.7 引入 Pending Birth 队列与重复新生抑制

为了避免短暂噪声点或同一目标在相邻位置被反复生出多个 ID，当前实现增加了：

- `pending_births`
- `birth_confirm_frames`
- `birth_suppress_iou`
- `birth_suppress_center_dist`
- `BirthReferenceCache`

### 3.8 工程实现层面做了大量缓存与向量化

当前 ByteTrack 改进版不只是算法逻辑变复杂了，工程上也做了明显优化：

- `STrack` 几何缓存
- frozen state 缓存
- batched detection geometry 预计算
- vectorized lost / zombie cost builder
- birth reference cache
- duplicate remove fast path

单元测试中专门保留了“legacy loop”和“fast path”的等价性校验，说明当前实现已经不是原始逐对循环版本，而是一个**经过性能重构的最终版**。

---

## 4. 算法核心先验

SPL-ByteTrack 的设计背后有几条非常明确的先验假设，这些先验决定了代码为何长成现在这样。

### 4.1 主干关联必须尽量稳，不轻易被 ReID 污染

Step1 / Step2 仍然保留 ByteTrack 的高分 / 低分 IoU 主流程：

- Step1：高分检测 + `tracked/lost` 池做 IoU 关联
- Step2：低分检测 + 未匹配 tracked 池做 IoU 关联

这说明当前实现的态度很明确：

- ByteTrack 的主干优势仍然有效；
- ReID 不应该粗暴地替代主干；
- 外观更适合在“高风险但低频”的恢复场景里使用。

### 4.2 新生比恢复更危险

在固定机位人群场景里，错误新生会直接带来：

- 身份碎裂
- 同人多 ID
- 后续 zombie / lost 恢复空间变窄

所以系统把大量逻辑压在“新生该不该放行”上，而不是一味让未匹配高分检测直接生新轨迹。

### 4.3 短时恢复应尽量前移，长时复活应更保守

一个目标刚丢失几帧时：

- 几何漂移还不大
- 历史状态还可信
- ReID 也更容易稳定发挥

所以 recent-lost recovery 被放在 zombie rescue 之前。

而 zombie rescue 处理的是更久远的恢复，因此必须更保守：

- 有中心距离硬门
- 有 shape gate
- 有 ReID 阈值门
- 最后再做 global assignment

### 4.4 场景结构应被“被动学习”，而不是“强行主宰”

当前空间先验模块的定位很克制：

- 它先学习，不急着参与决策；
- 满足 `support` 和 `birth` 的成熟阈值后才进入 `entry_only` 阶段；
- 默认 `bias_only`，说明它更像“偏置项”，不是“绝对裁判”。

### 4.5 空间统计不能被错误恢复污染

当前实现中，birth prior 并不是从所有新生都学习，而只从一部分**可信 birth source** 学习：

- `geometric_entry`
- `outside_expand`

这点非常关键。因为如果 `center_birth`、`spatial_entry`、`frame1` 之类来源也被无条件写回 prior，系统就会把偶然误生当成“稳定入口”，形成自激污染。

### 4.6 长时 lost 轨迹的位置不应无限漂移

一条轨迹丢失太久后，Kalman 继续预测反而可能把位置带偏。因此当前实现引入：

- `zombie_max_predict_frames`
- `frozen_mean`
- `get_tlwh_for_matching()`

这意味着：

- 丢失初期仍使用 live KF 状态；
- 超过一定帧数后冻结位置；
- zombie matching 优先匹配“冻结位置”而不是“长期漂移后的预测框”。

---

## 5. 代码级系统结构

### 5.1 核心类与模块

#### 5.1.1 `STrack`

位置：`boxmot/trackers/bytetrack/bytetrack.py`

它不只是原始 ByteTrack 的轨迹容器，当前版本里已经承载了大量生命周期状态：

- 基础几何状态：`xywh`、`tlwh`、`xyah`
- Kalman 状态：`mean`、`covariance`
- 外观状态：`curr_feat`、`smooth_feat`、`features`
- 空间先验元数据：
  - `spatial_birth_frame`
  - `spatial_birth_point`
  - `spatial_birth_committed`
  - `spatial_birth_trustworthy`
  - `spatial_birth_source`
- 恢复元数据：`last_recovery_frame`
- lost / zombie 生命周期状态：
  - `lost_frame_id`
  - `frozen_mean`
  - `exit_pending`
- 几何缓存：
  - `_cached_xyxy`
  - `_cached_tlwh_live`
  - `_cached_footpoint_live`
  - `_cached_tlwh_frozen`

#### 5.1.2 `ByteTrack`

位置：`boxmot/trackers/bytetrack/bytetrack.py`

当前的 `ByteTrack` 已经是整个 SPL-ByteTrack 的主控制器，负责：

- 常规 ByteTrack 两阶段关联
- effective zone 更新
- unmatched-high 路由
- recent-lost recovery
- zombie rescue
- birth confirmation / suppression
- exit zone 处理
- lost/zombie 迁移
- spatial prior 更新

#### 5.1.3 `SpatialPriorField`

位置：`boxmot/trackers/bytetrack/spatial_prior.py`

它负责维护低分辨率空间统计场：

- `support_count`
- `birth_count`

并提供：

- 概率图 `get_probability_maps()`
- 度量图 `get_metric_maps()`
- 点投影 `point_to_index()`
- 懒惰衰减 `step()`
- 高斯 splat 写入 `_splat_many()`

#### 5.1.4 `tracker_zoo.create_tracker`

位置：`boxmot/trackers/tracker_zoo.py`

负责：

- 读取 yaml 默认参数
- 将 `reid_weights / device / half / per_class` 注入 tracker
- 动态实例化 `ByteTrack`
- 如果模型存在则执行 `warmup()`

#### 5.1.5 `process_sequence`

位置：`boxmot/engine/evaluator.py`

负责：

- 用 `MOT17DetEmbDataset` 读取缓存的 dets/embs/img
- 创建 tracker
- 对每一帧调用 `tracker.update(dets, img, embs)`
- 把输出转成 MOT 结果

这也说明：在常见评测路径里，ByteTrack 拿到的是**预提取好的 embeddings**，tracker 侧主要负责关联而非在线 ReID 提取。

---

## 6. 生命周期状态机

当前实现中的轨迹生命周期可以概括为：

```text
New candidate
  -> Pending Birth         (如果 birth_confirm_frames > 1 且未满足跳过确认条件)
  -> Tracked               (激活成功)
  -> Lost                  (主干关联失败)
  -> Zombie                (丢失时间达到 zombie_transition_frames)
  -> Tracked               (recent-lost 或 zombie rescue 成功 re_activate)
  -> Removed               (exit-zone 宽限后移除 / baseline 超时移除 / history 裁剪)
```

### 6.1 当前实现中维护的轨迹集合

`ByteTrack` 里显式维护了 5 个集合：

- `active_tracks`
  - 当前活跃跟踪的轨迹
- `lost_stracks`
  - 常规 lost 池
- `zombie_stracks`
  - 长时记忆池
- `pending_births`
  - 待确认新生候选
- `removed_stracks`
  - 已移除轨迹

### 6.2 重要状态字段

#### `lost_frame_id`

表示轨迹何时进入 lost 状态，用于：

- 计算 `frames_lost`
- 判断 recent-lost 恢复窗口
- 判断何时转 zombie
- 判断 exit-zone grace 是否已到

#### `frozen_mean`

表示 zombie 匹配阶段使用的冻结位置。其设计目的不是替代 KF，而是限制长期漂移。

#### `exit_pending`

表示该 lost 轨迹是“疑似已离场”的对象。满足宽限期后优先删除。

#### `last_recovery_frame`

记录最近一次 `re_activate()` 的帧号，供 spatial prior 的 support 冷却逻辑使用，防止“刚恢复的轨迹”立刻污染 support field。

### 6.3 生命周期状态清理

`STrack.clear_lost_lifecycle_state()` 是一个非常重要的安全函数。它在：

- `activate()`
- `update()`
- `re_activate()`

中都会被调用，用于清除：

- `lost_frame_id`
- `frozen_mean`
- `exit_pending`

这保证了轨迹一旦回到 Tracked 状态，就不会携带上一轮 lost / zombie 周期的脏状态。对应的单元测试也明确覆盖了这一点。

---

## 7. 每帧处理总流程

下面按 `ByteTrack.update()` 的实际代码顺序，把整套算法流程完整展开。

### 7.1 Step 0：输入预处理与检测分层

输入 `dets` 的原始格式是：

```text
[x1, y1, x2, y2, conf, cls]
```

在进入 update 之后，代码先为每个检测追加一个 `det_ind`：

```text
[x1, y1, x2, y2, conf, cls, det_ind]
```

这个索引会在后续多处使用：

- unmatched-high routing
- outside-zone 检测标记
- 输出结果保留检测索引

随后根据置信度把检测切成两层：

1. 高分检测：`conf > track_thresh`
2. 低分检测：`min_conf < conf < track_thresh`

这仍然保留了 ByteTrack 最经典的 score-split 结构。

### 7.2 Step 0.5：ReID embedding 的准备策略

当前实现里：

- 如果外部已经传入 `embs`，则直接使用高分检测对应的 embedding；
- 如果没传入 embedding，但 `with_reid=True` 且有模型，则只对**高分检测**在线提特征；
- 低分检测 `detections_second` 不提 embedding。

这意味着：

- Step1 / Step4 使用的是高分检测特征；
- Step2 低分补关联是纯几何的；
- current code 的外观信息重点服务于 high-conf unmatched 恢复，而不是 low-conf second match。

### 7.3 Step 0.75：空间先验与 effective zone 的帧级更新

在每帧开始时，系统会：

1. 更新 `frame_count`
2. 清空 `_outside_zone_det_inds`
3. 清理过期 `pending_births`
4. 记录图像尺寸
5. 如果启用 spatial prior：
   - `configure_image()`
   - `step()` 执行一次时间衰减
   - 若 region 已成熟，则把 region mask 标记为 dirty，等待按需刷新
6. 如果启用 adaptive zone，则在正式关联前用高分检测执行一次 `_update_effective_zone(detections, phase='pre')`

---

## 8. Step1：高分检测主关联

### 8.1 参与者

高分主关联的轨迹池是：

```python
strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
```

这点非常关键：

- 当前 tracked 轨迹会参与
- 历史 lost 轨迹也会参与

也就是说，**短时断链恢复的第一机会并不是 recent-lost ReID，而仍然是 ByteTrack 自带的 IoU 主干恢复**。

### 8.2 预测

系统对 `strack_pool` 做 `STrack.multi_predict()`：

- 对 `Tracked` 轨迹正常预测
- 对非 `Tracked` 轨迹会把速度项清零后再预测

### 8.3 距离与匹配

Step1 的代价仍然是典型 ByteTrack 风格：

- `iou_distance(strack_pool, detections)`
- 再通过 `fuse_score()` 融合检测分数
- 最后 `linear_assignment(..., thresh=self.match_thresh)`

### 8.4 匹配后的状态更新

- 如果匹配到的是 `Tracked` 轨迹：调用 `update()`
- 如果匹配到的是 `Lost` 轨迹：调用 `re_activate()`

两种情况都会：

- 重新变成有效 tracked
- 清掉 lost 生命周期残留
- 清掉 `exit_pending`

因此，**Step1 本身已经是一次“最近丢失轨迹恢复”的主战场**。

---

## 9. Step2：低分检测补关联

Step2 只让 **Step1 未匹配且状态仍为 Tracked 的轨迹** 参与：

```python
r_tracked_stracks = [
    strack_pool[i]
    for i in u_track
    if strack_pool[i].state == TrackState.Tracked
]
```

然后用低分检测 `detections_second` 做第二次 IoU 关联：

- `iou_distance(r_tracked_stracks, detections_second)`
- `linear_assignment(..., thresh=0.5)`

这一步没有用 ReID，也没有用 spatial prior。

它的职责非常纯粹：

- 尽量利用低分框延续当前已经存在的轨迹
- 保住被压低置信度但其实位置正确的目标

### 9.1 Step2 失败后的去向

如果某个 tracked 轨迹在 Step2 后仍未匹配到：

1. 若启用 `exit_zone_enabled`，先根据当前 box 判断是否进入 exit zone；
2. 设置 `exit_pending=True/False`；
3. `mark_lost()`；
4. `lost_frame_id = current_frame`；
5. 暂存到本帧局部 `lost_stracks`。

注意这里的 exit 判断使用的是**当前 live track state**，而不是初始框；单元测试也专门覆盖了这一点。

---

## 10. Step3：未确认轨迹（unconfirmed）清理

`active_tracks` 中那些还没正式激活成功的轨迹会被放进 `unconfirmed`。

这些轨迹会再与 Step1 剩下的未匹配高分检测做一次 IoU + score 匹配：

- 匹配成功：`update()` 并转活跃
- 匹配失败：直接移除

这部分逻辑基本保持 ByteTrack 原味，目标是处理“刚开始只出现过一次的半成品轨迹”。

---

## 11. Step4：第三关联阶段 —— unmatched high 的显式路由

这是 SPL-ByteTrack 与原始 ByteTrack 差异最大的部分。

### 11.1 为什么 Step4 是整个系统的中枢

Step1 / Step2 结束后，剩下的是：

- 没被任何现有 tracked/lost 轨迹解释掉的高分检测

这些检测是最危险的一类：

- 它们可能是真新生
- 也可能是 recent-lost 该恢复的旧 ID
- 也可能是 zombie 该复活的旧 ID
- 还可能只是位于“中心区、不该直接生”的高风险检测

因此当前实现没有直接 `activate()`，而是先显式分类，再按路由执行。

### 11.2 unmatched-high 路由器：`_classify_unmatched_high_detection()`

它会给每个 unmatched high detection 产出一个 `UnmatchedHighDecision`，包含：

- `det_index`
- `route`
- `skip_confirmation`
- `birth_source`

### 11.3 路由规则

#### 路由 1：低于 `new_track_thresh`，直接不处理

如果高分检测虽然进了 Step1，但还没高到允许新生：

- `conf < new_track_thresh`
- 返回 `None`
- 既不新生，也不进入后续 recover 路由

#### 路由 2：outside-before-expand，新生但不跳过确认

如果使用 `adaptive_zone_enabled=True`、`always_expand`，并且该检测在本轮扩张前位于旧 effective zone 外：

- 路由为 `birth`
- `birth_source = outside_expand`
- `skip_confirmation = False`

这是一个很细的实现点：

- 这种 birth 被视为“可能真的来自新区”，但不是强几何入口，因此**仍要过 birth confirmation**。

#### 路由 3：entry zone 允许的新生，直接跳过确认

如果 `_get_entry_zone_info()` 判定该检测属于 entry：

- 路由为 `birth`
- `skip_confirmation = True`
- `birth_source` 可能是：
  - `geometric_entry`
  - `spatial_entry`

这体现出当前系统的偏好：

- 真正可信的入口点，不需要再等两帧确认；
- 它们更像“合理入场”，不是“中心区突然冒出一个目标”。

#### 路由 4：中心区检测，先恢复，恢复失败后禁止新生

如果检测不在 entry，并且：

- `strict_entry_gate=True`
- `entry_margin > 0`

则路由为：

- `recover_then_block`

含义是：

- 可以尝试把它解释成旧 ID（recent-lost / zombie）
- 但如果旧 ID 恢复失败，就**不允许**在中心区生新 ID

#### 路由 5：中心区检测，先恢复，恢复失败后允许新生

否则路由为：

- `recover_then_birth`

含义是：

- 先尝试 old-id recovery
- 若 recover 不成功，则允许它走中心区新生流程

### 11.4 先处理直接 birth 的 unmatched high

Step4 中，所有被路由成 `birth` 的检测会先调用 `_try_activate_new_track()`。

这意味着 Step4 的顺序是：

1. 先把显然属于新生的检测处理掉
2. 剩下的中心区高风险检测再进入 recover 通道

这个顺序有助于：

- 降低 recover 阶段的干扰候选数
- 提前把本帧新激活轨迹加入 birth reference cache
- 防止后面的 birth 再和已激活对象重复生 ID

---

## 12. 新生机制（Birth Control）详解

### 12.1 `_try_activate_new_track()` 的职责

它统一处理：

- duplicate suppression
- temporal confirmation
- 轨迹激活
- spatial birth 元数据注册
- exit_pending 清理

### 12.2 两类新生：立即激活 vs 待确认激活

#### 立即激活

触发条件：

- `skip_confirmation=True`，或
- `birth_confirm_frames <= 1`

立即激活前仍然要先过 `_is_birth_suppressed()`。

#### 待确认激活

否则会进入 `pending_births`。

### 12.3 `pending_births` 的工作方式

每个 pending 条目记录：

- `track`
- `hits`
- `last_frame`

当前实现中还有两个内部常量：

- `_birth_confirm_iou = 0.3`
- `_birth_pending_max_miss = 1`

其语义是：

- 新一帧候选要和已有 pending 候选有足够 IoU 或足够近的中心距离，才算同一个 birth candidate；
- 超过 1 帧没连续命中就会被 `_prune_pending_births()` 删除。

因此 `birth_confirm_frames=2` 的真实效果不是“任意两帧见过就行”，而是：

- 必须连续出现
- 位置还要足够一致

### 12.4 duplicate suppression：防止同一目标瞬间生多个 ID

`_is_birth_suppressed()` 会检查当前候选是否与参考轨迹过近。参考集合来自：

- `active_tracks`
- `lost_stracks`
- `zombie_stracks`
- 本帧 `activated_starcks`
- 本帧 `refind_stracks`
- 本帧新产生的 `lost_stracks`

并通过 `BirthReferenceCache` 把这些参考轨迹的：

- `tlwh`
- `center`

一次性缓存下来。

### 12.5 为什么 `BirthReferenceCache` 很重要

如果不把**本帧刚激活的轨迹**也立刻加入参考池，那么同一帧后面一个很相似的 detection 仍可能再次生新 ID。

当前实现通过 `_append_birth_reference_track()` 在每次成功新生后更新 cache，保证：

- 同帧后续的候选也会避开刚生成的 ID

这是当前实现中非常关键的一个“同帧防重复”设计。

---

## 13. recent-lost recovery：短时恢复前移

### 13.1 进入条件

Step4 中，所有中心区 unmatched high detection 会先进入 recent-lost recovery。

候选轨迹来自：

```python
recent_lost_candidates = joint_stracks(self.lost_stracks, lost_stracks)
recent_lost_candidates = sub_stracks(recent_lost_candidates, refind_stracks)
```

这说明 recent-lost recovery 会同时覆盖：

- 历史 lost 池里的轨迹
- 本帧刚刚因为 Step2 失败而掉入 lost 的轨迹

然后再按 `lost_reid_max_frames` 做时间裁剪。

### 13.2 cost builder：`_build_lost_match_cost()`

当前实现里，recent-lost recovery 的代价矩阵是**先 hard gate，再 soft cost**。

只有同时满足下列条件，一个 lost-detection 对才会得到有效 cost：

1. 检测框面积 `>= lost_reid_min_box_area`
2. 中心距离 `<= lost_match_max_dist`
3. 宽高比例门 `<= lost_shape_max_ratio`
4. ReID cost `<= lost_reid_thresh`

若任一条件不满足，就会被赋予：

```text
invalid_cost = lost_match_cost_thresh + 1.0
```

也就是无论如何都不会通过最终 Hungarian assignment 的那种“大无效代价”。

### 13.3 recent-lost 的总成本公式

有效候选的总成本为：

```text
reid_cost   = 1 - cosine_similarity
motion_cost = min(1, center_distance / lost_match_max_dist)
shape_cost  = min(1, 0.5 * (|log(width_ratio)| + |log(height_ratio)|))

total_cost =
    (lost_reid_weight   * reid_cost +
     lost_motion_weight * motion_cost +
     lost_shape_weight  * shape_cost)
    / (lost_reid_weight + lost_motion_weight + lost_shape_weight)
```

在 `bytetrack_improved.yaml` 中，对应默认权重是：

- `lost_reid_weight = 0.70`
- `lost_motion_weight = 0.25`
- `lost_shape_weight = 0.05`

可以看出：

- 外观是主项
- 几何是辅助项
- shape 是稳定性修饰项

### 13.4 assignment 与恢复动作

构造完代价矩阵后，调用：

```python
linear_assignment(cost_matrix, thresh=self.lost_match_cost_thresh)
```

匹配成功的 recent-lost 轨迹会执行：

- `re_activate(det_track, frame_id, new_id=False)`

这意味着：

- 保留原 ID
- 更新 Kalman 状态
- 更新外观特征
- 清除 lost/zombie 残留状态
- 记录 `last_recovery_frame`

### 13.5 这一层的设计意义

这一步是当前代码相对旧版思路最关键的升级之一：

- 很多原本要拖到 zombie 才能“勉强复活”的 case
- 现在会在 recent-lost 阶段先被找回

这比单纯继续增强 zombie rescue 更符合真实问题结构。

---

## 14. zombie rescue：长时复活

### 14.1 zombie 的来源

当 `zombie_enabled=True` 时，lost 轨迹在满足：

```text
frames_lost >= zombie_transition_frames
```

后，会从 `lost_stracks` 转移到 `zombie_stracks`。

### 14.2 zombie 的意义

zombie 不是“活跃轨迹”，而是：

- 已经脱离常规 short-term lost 管理
- 但仍保留身份和外观的长期记忆对象

它相当于告诉系统：

> 这条轨迹正常来说应该删掉了，但在当前场景里，我们认为它仍有价值被后续中心区高分检测复活。

### 14.3 冻结位置：`frozen_mean`

如果启用 zombie 且 `zombie_max_predict_frames > 0`，那么 lost 轨迹在丢失一定帧数后会冻结位置：

- 若 `frames_lost >= zombie_max_predict_frames`
- 且 `frozen_mean is None`
- 则 `frozen_mean = mean.copy()`

之后 `get_tlwh_for_matching()` 会优先返回 frozen tlwh，而不是持续漂移后的 live KF 位置。

这使 zombie rescue 的位置假设更稳定。

### 14.4 cost builder：`_build_zombie_match_cost()`

zombie rescue 的思想与 recent-lost 类似，但语义上更保守。有效候选必须通过：

1. 若启用 ReID，则检测面积 `>= zombie_reid_min_box_area`
2. 中心距离 `<= effective_zombie_gate_dist`
3. `max(width_ratio, height_ratio) <= zombie_shape_max_ratio`
4. 若启用 ReID，则 `reid_cost <= zombie_reid_thresh`

这里的 `effective_zombie_gate_dist` 来自：

```python
_zombie_gate_dist(max_dist)
```

其实现是：

- 先取 `max_dist`（默认 `zombie_match_max_dist`）
- 如果 `zombie_dist_thresh > 0`，再取两者最小值

所以当前代码里，真正生效的硬门距离通常是：

```text
min(zombie_match_max_dist, zombie_dist_thresh)
```

### 14.5 zombie 总成本公式

有 ReID 时：

```text
total_cost =
    (zombie_reid_weight   * reid_cost +
     zombie_motion_weight * motion_cost +
     zombie_shape_weight  * shape_cost)
    / (zombie_reid_weight + zombie_motion_weight + zombie_shape_weight)
```

无 ReID 时：

```text
total_cost =
    (zombie_motion_weight * motion_cost +
     zombie_shape_weight  * shape_cost)
    / (zombie_motion_weight + zombie_shape_weight)
```

默认 improved 配置中：

- `zombie_reid_weight = 0.75`
- `zombie_motion_weight = 0.20`
- `zombie_shape_weight = 0.05`

这说明 zombie rescue 明确是**ReID 主导**的。

### 14.6 global assignment 的意义

zombie rescue 不是“逐个找最近邻”，而是：

- 对 zombie 轨迹和剩余中心区检测构造完整 cost matrix
- 用 `linear_assignment()` 做一次全局最优匹配

这可以避免在多目标交叉场景里出现“局部最近邻正确但全局身份交换”的问题。单测 `test_bytetrack_zombie_reid_global_assignment_prefers_appearance()` 就在验证这一点。

### 14.7 zombie rescue 失败后的去向

如果某个中心区 unmatched high detection：

- recent-lost recovery 失败
- zombie rescue 也失败

则它是否允许继续新生，取决于前面的路由决策：

- `recover_then_block`：到此为止，直接放弃
- `recover_then_birth`：进入中心区新生逻辑

---

## 15. exit zone 机制

### 15.1 基本逻辑

`_is_in_exit_zone()` 只看图像边缘矩形区域，不依赖 adaptive zone，也不依赖 spatial prior。

这一点非常有意图：

- entry 可以借助 learned prior 做偏置
- 但 exit 判定保持几何规则，避免 learned prior 干扰“离场”判断

### 15.2 何时标记 `exit_pending`

在 Step2 结束后，如果一个 tracked 轨迹仍未匹配：

- 若启用 `exit_zone_enabled`
- 且当前 box 位于 exit zone
- 则 `exit_pending=True`

### 15.3 何时真正删除

在 Step6 处理 lost -> zombie / remove 时：

- 若 `exit_pending=True`
- 且 `frames_lost >= exit_zone_remove_grace`
- 直接 `mark_removed()`
- 不进入 zombie 池

### 15.4 设计意义

这相当于在生命周期里显式引入“疑似离场”语义，减少：

- 边缘离场目标继续长期保存在 zombie 池中
- 后续又被错误拉回中心区

---

## 16. adaptive effective zone

### 16.1 目的

它不是在做语义分割，也不是在学背景，而是在估计：

> 当前画面中“合理会出现目标的有效活动区域”大概在哪里。

这可以减少：

- 画面无效区域误生 ID
- 因镜头边界过大导致的松散 entry 判定

### 16.2 两种工作模式

#### `warmup_once`

- 前 `adaptive_zone_warmup` 帧收集 detections
- 到阈值时一次性估计 `_effective_zone`
- 后续不再缩放或更新

#### `always_expand`

- 每帧都可依据候选检测扩展 effective zone
- 只允许**单调外扩**，不允许收缩

### 16.3 effective zone 的几何计算

`_compute_effective_zone()` 会：

1. 汇总 warmup detections 的 tlwh
2. 转成整体 xyxy 包围盒
3. 以包围盒中心为轴做 `adaptive_zone_padding`
4. 最后裁剪到图像范围内

### 16.4 entry 判定与 effective zone 的关系

在 adaptive mode 下，`_is_in_rect_entry_zone()` 的规则是：

- 如果 box 完全在 effective zone 外，则视为 entry；
- 若 `adaptive_zone_entry_mode='outside_only'`，则 zone 内部都不是 entry；
- 若 `adaptive_zone_entry_mode='margin_inside'`，则 zone 内部靠边缘一定 margin 的区域仍算 entry。

improved 默认使用：

- `adaptive_zone_update_mode = warmup_once`
- `adaptive_zone_entry_mode = outside_only`

说明当前默认策略更保守：

- 有效活动区一旦估好，就不轻易改；
- 在有效活动区内部，不因为“靠近 zone 边界”就轻易开放 birth。

### 16.5 outside-before-expand 保护机制

在 `always_expand` 模式下，如果不记录“扩张前它在 zone 外”，就会出现：

- 先因为这个检测把 zone 扩大
- 再回过头发现它已经在 zone 内
- 最终失去“它本来来自 zone 外”的信息

当前实现用 `_outside_zone_det_inds` 专门保存这个事实。这个细节正是 `test_bytetrack_outside_before_expand_keeps_new_id_creation()` 在验证的内容。

---

## 17. Spatial Prior：被动学习的固定机位场景先验

### 17.1 `SpatialPriorField` 学什么

它维护两个二维栅格场：

- `support_count`
  - 目标稳定存在过的支持证据
- `birth_count`
  - 目标可信诞生过的入口证据

两者都不是点计数，而是通过高斯 splat 投影到低分辨率网格上。

### 17.2 懒惰衰减（lazy decay）

`SpatialPriorField.step()` 并不会每帧直接把整张图逐元素乘 `decay`，而是：

- 先把全局 `_decay_scale *= decay`
- 直到 scale 很小才真正 materialize 到 field

这样做可以显著减少每帧大数组乘法开销。

### 17.3 概率图与度量图

#### 概率图 `get_probability_maps()`

返回：

- `walkable`
- `birth`
- `confidence`

其中：

- `walkable` 反映 support 密度归一化后的可行走性
- `birth` 用 `(birth + alpha) / (support + alpha + beta)` 形成带先验的条件概率
- `confidence` 近似表示该区域统计是否足够成熟

#### 度量图 `get_metric_maps()`

返回：

- `support_density`
- `birth_density`
- `birth_ratio`

`birth_density` 是 `3x3 local sum` 后的局部出生密度，对显式 entry/core mask 的构造非常重要。

### 17.4 spatial prior 何时开始介入决策

当前实现使用分阶段策略：

- 初始阶段：`learn_only`
- 成熟阶段：`entry_only`

只有在同时满足：

- `spatial_prior_support_samples >= spatial_prior_entry_support_threshold`
- `spatial_prior_birth_events >= spatial_prior_entry_birth_threshold`

后，系统才认为 prior 已经成熟到可以生成显式 entry/core 区。

在那之前，即便 prior 已经开始累积统计，也**不会**直接拿来做 entry region 判定。

### 17.5 support 与 birth 如何写入

#### support 写入

`_update_spatial_prior_tracks()` 会遍历 `active_tracks`，对满足条件的 tracked 轨迹写入 support：

- 轨迹必须已激活
- 状态必须是 `Tracked`
- 年龄 `>= spatial_prior_support_min_age`
- 若有 recovery cooldown，则必须距离最近一次 `re_activate()` 足够久

#### birth 写入

轨迹只有在其 birth 元数据尚未 committed 时，才会等待条件成熟后写入 birth：

- `age >= spatial_prior_birth_commit_age`，或
- `hits >= spatial_prior_birth_commit_hits`

并且还必须满足：

- `spatial_birth_trustworthy=True`
- `spatial_birth_point is not None`

这是一套非常严格的反污染机制。

### 17.6 什么样的新生会被视为 trustworthy birth

当前代码只认可以下 source：

- `geometric_entry`
- `outside_expand`

默认不信任：

- `frame1`
- `center_birth`
- `spatial_entry`
- `unknown`

这说明 spatial prior 的 birth 学习非常保守：

- 只从几何上可信的入场现象中学习入口热点
- 不从自身 override 结果反向强化自己

### 17.7 entry/core mask 的生成逻辑

`_refresh_spatial_region_masks()` 的过程可以分解为：

1. 用 `confidence` 和 `walkable` 得到 `eligible`
2. 对 `eligible` 做腐蚀，得到 core seed
3. 用 `eligible - core_seed` 得到 outer band / entry band
4. 在 entry band 上找 `birth_density > 0` 的候选区域
5. 取高分位数阈值作为 seed
6. 取较低分位数阈值作为 grow candidate
7. 沿连通分量进行有约束生长
8. 最后把：
   - 生长出的区域记为 `spatial_entry_mask`
   - 剩余 `eligible` 区域记为 `spatial_core_mask`

### 17.8 为什么不是“所有高 birth_ratio 的点都算 entry”

因为那样容易出现两类问题：

- 一条弱桥把整个 outer band 吞进去
- 一块偶然高值噪声扩成大片入口区

当前实现通过以下设计限制这种风险：

- 只在 walkable + confident 的区域里操作
- 只在 outer band 上找 entry
- 用连通分量过滤
- 用 `component_mean_ratio` 过滤弱组件
- 用 `grow_max_steps` 限制长尾扩张
- 可选 `component_max_area` 防止超大 entry 区

### 17.9 `bias_only` 与 `strict_region`

`_get_entry_zone_info()` 的逻辑表明 spatial prior 不是永远强裁判。

#### `bias_only`

- `entry` 区可以直接放行
- `core` 区不会否掉几何 edge entry
- 也就是说：prior 只增加入口，不强行盖过矩形规则

#### `strict_region`

- 如果点落在 `core` 区，则认为不是 entry
- 这是更强的 veto 版本

当前 improved 默认用的是：

- `spatial_prior_entry_mode = bias_only`

这与当前算法整体风格一致：**让 prior 提供加性偏置，而不是绝对裁决。**

---

## 18. 关键几何与外观公式

### 18.1 IoU

`_calculate_iou()` 与 `_calculate_iou_many()` 都使用 tlwh 格式计算交并比。

### 18.2 中心距离

`_calculate_center_distance()` / `_calculate_center_distance_many()` 计算的是 box center 的欧氏距离。

### 18.3 shape ratio 与 shape cost

```text
width_ratio  = max(w1, w2) / max(min(w1, w2), 1e-6)
height_ratio = max(h1, h2) / max(min(h1, h2), 1e-6)

shape_cost = min(1.0,
                 0.5 * (|log(width_ratio)| + |log(height_ratio)|))
```

其直觉是：

- 完全一致时 cost 约为 0
- 尺寸 / 宽高比差异越大，cost 越高
- 且始终被截断到 `[0, 1]`

### 18.4 ReID cost

当前实现用的是归一化特征的余弦距离：

```text
reid_cost = 1 - clip(feature_a @ feature_b, -1, 1)
```

其中：

- `STrack.update_features()` 会先做 `L2 normalize`
- `smooth_feat` 采用 EMA 平滑
- `alpha = 0.9`

因此：

- zombie / lost 的轨迹侧更多使用平滑后的 `smooth_feat`
- detection 侧使用 `curr_feat`

---

## 19. `STrack` 的实现细节与工程优化

### 19.1 几何缓存

当前 `STrack` 不再每次都临时从 `mean` 反算几何，而是维护：

- live xyxy
- live tlwh
- live footpoint
- frozen tlwh

并用 `_invalidate_geom_cache()` / `_invalidate_frozen_cache()` 控制失效时机。

### 19.2 为什么 footpoint 被单独缓存

因为空间先验模块使用的是底部中心点（footpoint），而不是 box center：

```text
[x_center, y_center + 0.5 * height]
```

这对固定机位场景更合理，因为它更接近“人在地面上的投影位置”。

### 19.3 `get_tlwh_for_matching()` 的双态行为

它有两种返回：

- 正常返回 live tlwh
- 若满足 frozen 条件，则返回 frozen tlwh

因此整个系统不需要在 zombie / lost cost builder 里到处写 if/else，只需统一调用这一个接口。

### 19.4 geometry batching

当前实现还把 detection geometry 的构造批处理化了：

- `_prepare_detection_geometry()` 一次性得到 `xywh_batch / tlwh_batch / xyah_batch`
- 避免每个检测分别调用转换函数

### 19.5 vectorized cost builder

`_build_lost_match_cost()` 和 `_build_zombie_match_cost()` 都已经是向量化实现。单元测试里保留了 legacy loop 版本，并要求：

- 输出 shape 一致
- 数值逐元素完全一致

这说明当前实现并不是“近似优化”，而是**严格等价的工程重构**。

---

## 20. 参数族与当前推荐理解

当前同一套 SPL-ByteTrack 代码有几组典型参数包：

- `boxmot/configs/trackers/bytetrack.yaml`
- `boxmot/configs/trackers/bytetrack_original.yaml`
- `boxmot/configs/trackers/bytetrack_improved.yaml`
- `boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml`
- `boxmot/configs/trackers/bytetrack_dual_tuned.yaml`

### 20.1 `bytetrack_improved.yaml`：当前改进版基线

这份配置最适合作为“算法默认形态”的说明对象，因为它把整套机制都打开了，而且参数上相对平衡。

#### 基础阈值

- `min_conf = 0.1`
- `track_thresh = 0.5`
- `new_track_thresh = 0.65`
- `match_thresh = 0.8`
- `track_buffer = 30`
- `frame_rate = 30`

#### Birth 控制

- `entry_margin = 50`
- `strict_entry_gate = false`
- `birth_confirm_frames = 2`
- `birth_suppress_iou = 0.7`
- `birth_suppress_center_dist = 35`

#### Zombie / Lost 生命周期

- `zombie_max_history = 100`
- `zombie_transition_frames = 30`
- `lost_max_history = 0`
- `zombie_match_max_dist = 200`
- `zombie_dist_thresh = 150`
- `zombie_max_predict_frames = 5`

#### Zombie rescue

- `zombie_reid_enabled = true`
- `zombie_reid_weight = 0.75`
- `zombie_motion_weight = 0.20`
- `zombie_shape_weight = 0.05`
- `zombie_reid_thresh = 0.35`
- `zombie_match_cost_thresh = 0.45`
- `zombie_shape_max_ratio = 2.0`
- `zombie_reid_min_box_area = 1024`

#### recent-lost recovery

- `lost_reid_enabled = true`
- `lost_match_max_dist = 120`
- `lost_reid_max_frames = 15`
- `lost_reid_weight = 0.70`
- `lost_motion_weight = 0.25`
- `lost_shape_weight = 0.05`
- `lost_reid_thresh = 0.25`
- `lost_match_cost_thresh = 0.35`
- `lost_shape_max_ratio = 1.8`
- `lost_reid_min_box_area = 1024`

#### Exit zone

- `exit_zone_enabled = true`
- `exit_zone_margin = 50`
- `exit_zone_remove_grace = 30`

#### Adaptive zone

- `adaptive_zone_enabled = true`
- `adaptive_zone_update_mode = warmup_once`
- `adaptive_zone_expand_trigger = outside_high`
- `adaptive_zone_entry_mode = outside_only`
- `adaptive_zone_warmup = 10`
- `adaptive_zone_margin = 50`
- `adaptive_zone_padding = 1.2`
- `adaptive_zone_min_box_area = 0`

#### Spatial prior

- `spatial_prior_enabled = true`
- `spatial_prior_grid_w = 48`
- `spatial_prior_grid_h = 27`
- `spatial_prior_sigma = 1.5`
- `spatial_prior_decay = 0.999`
- `spatial_prior_birth_commit_age = 3`
- `spatial_prior_birth_commit_hits = 3`
- `spatial_prior_support_min_age = 2`
- `spatial_prior_entry_mode = bias_only`
- `spatial_prior_recovery_cooldown = 5`
- `spatial_prior_region_enabled = true`
- `spatial_prior_region_conf = 1.0`
- `spatial_prior_region_walk = 0.05`
- `spatial_prior_region_birth = 0.85`
- `spatial_prior_region_birth_grow = 0.6`
- `spatial_prior_region_grow_max_steps = 3`
- `spatial_prior_region_component_mean_ratio = 0.45`
- `spatial_prior_region_component_max_area = 0`
- `spatial_prior_entry_band_radius = 2`
- `spatial_prior_entry_support_threshold = 100`
- `spatial_prior_entry_birth_threshold = 8`

### 20.2 `bytetrack_sompt22_tuned.yaml`：单数据集更保守的恢复版

相对于 improved，只改了少数关键阈值：

- `match_thresh: 0.80 -> 0.76`
- `birth_suppress_center_dist: 35 -> 25`
- `zombie_dist_thresh: 150 -> 130`
- `zombie_reid_thresh: 0.35 -> 0.30`
- `zombie_match_cost_thresh: 0.45 -> 0.38`
- `lost_reid_thresh: 0.25 -> 0.20`

可以把它理解为：

- 主干匹配略放松
- 最近恢复和 zombie 复活都更严格
- 更偏向压误救、压误匹配

### 20.3 `bytetrack_dual_tuned.yaml`：跨 SOMPT22 / MOT20 的折中配置

相对于 improved，它主要改动：

- `match_thresh: 0.80 -> 0.78`
- `birth_suppress_center_dist: 35 -> 25`
- `lost_match_max_dist: 120 -> 100`

可以理解为：

- 维持同一套 SPL-ByteTrack 结构不变
- 只针对跨数据集泛化做轻量折中

### 20.4 重要理解

这三份配置的关系不是“完全不同的算法”，而是：

> 同一套 SPL-ByteTrack 代码骨架上的三组参数形态。

也就是说，本文描述的是**算法结构层面的最终版**；而 tuned / dual tuned 只是它的参数化实例。

---

## 21. 单元测试已经固化的关键算法行为

`tests/unit/test_trackers.py` 实际上把当前 SPL-ByteTrack 的很多行为都写死成了“契约”，这些测试非常值得作为算法理解依据。

### 21.1 effective zone 契约

- 单调外扩不收缩
- outside-before-expand 信息不能丢
- `outside_only` 和 `margin_inside` 语义不同

### 21.2 birth 契约

- entry birth 可以跳过 confirmation
- `new_track_thresh` 会拦截低于阈值的新生
- `birth_confirm_frames=2` 要求连续两次命中
- duplicate suppression 不能让同目标生双 ID
- pending birth 会因中断而过期

### 21.3 recovery 契约

- zombie global assignment 在冲突场景下应由 appearance 主导
- wrong appearance 不应被 zombie rescue 错救
- recent-lost recovery 应先于 zombie
- wrong appearance 不应通过 recent-lost 恢复

### 21.4 spatial prior 契约

- stable birth / support 才会被写入
- 首帧 birth 不参与学习
- lazy decay 数学上要保持有效质量
- entry/core mask 的成长、长尾限制、中心 override 等行为都被测试覆盖
- spatial override 不能反过来污染 birth learning
- recovery 后的轨迹要经过 cooldown 才能继续贡献 support
- spatial prior 必须先成熟，成熟前只能回退到矩形 entry 规则

### 21.5 生命周期清理契约

- `update()` / `re_activate()` 后必须清空 lost lifecycle 残留
- `frozen_mean` 不能跨 lost 周期复用旧值

### 21.6 工程等价性契约

- geometry cache 与旧公式完全一致
- detection geometry batching 与原始逐个构造一致
- birth cache 与旧逻辑一致
- lost / zombie 向量化 cost builder 与旧循环逐元素一致
- duplicate remove fast path 与 legacy 逻辑一致

这意味着：当前版 SPL-ByteTrack 已经不仅是“试验性逻辑”，而是**被单测严密钉死过行为边界的实现版系统**。

---

## 22. 从调用链看整个系统如何落地

### 22.1 参数加载

`tracker_zoo.create_tracker()` 会：

1. 读取对应 yaml
2. 取每个参数的 `default`
3. 注入 `reid_weights`、`device`、`half`
4. 实例化 `ByteTrack`

### 22.2 评测数据流

`evaluator.process_sequence()` 中：

1. 用 `MOT17DetEmbDataset` 读取某序列所有帧
2. 每帧拿到：
   - `dets`
   - `embs`
   - `img`
3. 调用 `tracker.update(dets, img, embs)`
4. 结果写成 MOTChallenge 格式

### 22.3 为什么这很重要

它说明当前仓库的“评测标准路径”是：

- detector / ReID 特征可被提前缓存
- tracker update 的主要工作是关联与生命周期控制
- 因此在评测中看到的“association FPS”更接近 tracker 算法本身，而不是 detector / ReID 全链路耗时

---

## 23. SPL-ByteTrack 的整体伪代码

下面给出一个尽量贴近当前代码的伪代码描述：

```text
for each frame:
    append det_ind to detections
    split detections into high-score and low-score
    prepare embeddings for high-score detections if needed

    update image shape
    prune pending births
    step spatial prior decay
    update adaptive effective zone (pre)

    tracked_stracks = confirmed active tracks
    unconfirmed = inactive active tracks

    # Step1: standard ByteTrack high-score association
    strack_pool = tracked_stracks + lost_stracks
    KF predict on strack_pool
    match high-score detections with strack_pool by IoU + score
    matched tracked -> update
    matched lost    -> re_activate

    # Step2: low-score association for unmatched tracked only
    match remaining tracked with low-score detections by IoU
    matched tracked -> update
    unmatched tracked -> mark lost, maybe set exit_pending

    # Step3: handle unconfirmed tracks
    match unconfirmed with remaining unmatched high detections
    unmatched unconfirmed -> remove

    # Step4: explicit unmatched-high routing
    classify each unmatched high detection:
        - birth
        - recover_then_block
        - recover_then_birth
        - or drop if below new_track_thresh

    process direct birth detections first
    build birth reference cache

    center_zone_detections -> try recent-lost recovery
    remaining center_zone_detections -> try zombie rescue
    still remaining detections:
        if route == recover_then_birth:
            try activate as center birth
        else:
            block

    # Step5: append newly lost tracks
    append frame-local lost tracks into self.lost_stracks
    freeze long-lost positions if needed

    # Step6: lost -> zombie or remove
    for each lost track:
        if exit_pending and grace expired:
            remove
        elif zombie enabled and lost long enough:
            move to zombie pool
        elif baseline mode and over max_time_lost:
            remove
        else:
            keep in lost pool

    # Step7/8/9: cleanup and history limits
    remove rescued zombies from zombie pool
    cap zombie history
    cap lost history
    merge active / refound tracks
    subtract duplicates and removed tracks

    update spatial prior support/birth
    update spatial prior stage
    output current active tracks
```

---

## 24. 这套最终版算法的本质总结

如果只看代码结构而不看具体参数，当前 SPL-ByteTrack 的本质有 5 个关键词：

1. **Conservative mainline**
   - 主干不乱改，仍以 ByteTrack 的高低分 IoU 关联为核心

2. **Lifecycle splitting**
   - 把轨迹分成 tracked / lost / zombie / pending birth，不同状态走不同恢复与删除逻辑

3. **Birth control**
   - 把“是否生新 ID”从隐含动作提升为显式模块，进行多级门控

4. **Recovery layering**
   - recent-lost 前移，zombie 后置，短时恢复和长时复活不再混用

5. **Passive scene prior**
   - 场景先验先学习后介入，默认只做 bias，不抢主决策权

从工程实现上看，它又额外体现出两个特征：

- **反污染意识很强**：不让错误新生、刚恢复轨迹、spatial override 结果去反向污染 prior
- **性能意识很强**：核心几何与代价计算都做了缓存和向量化，同时通过测试保证语义不变

因此，当前仓库中的最终版 SPL-ByteTrack 不是“在 ByteTrack 上零散打补丁”，而是一套已经成体系的：

- 结构上分层
- 决策上保守
- 恢复上前移/后置分工明确
- 场景上可被动学习
- 工程上可验证、可复现、可评测

的完整多目标跟踪算法实现。

---

## 25. 建议如何阅读源码

如果后续还要继续深入这套实现，最推荐的阅读顺序是：

1. 先读 `boxmot/trackers/bytetrack/bytetrack.py` 的 `ByteTrack.update()`
   - 抓住每帧主流程
2. 再读以下 6 个辅助模块：
   - `_classify_unmatched_high_detection()`
   - `_try_activate_new_track()`
   - `_build_lost_match_cost()`
   - `_build_zombie_match_cost()`
   - `_update_spatial_prior_tracks()`
   - `_refresh_spatial_region_masks()`
3. 再读 `STrack`
   - 理解 geometry cache / frozen state / lifecycle reset
4. 最后读 `SpatialPriorField`
   - 理解 support / birth 的数学结构
5. 并结合 `tests/unit/test_trackers.py`
   - 把“代码怎么写”映射成“系统保证了什么行为”

这样最容易把这套最终版 SPL-ByteTrack 的算法结构、实现逻辑和工程约束一起看明白。
