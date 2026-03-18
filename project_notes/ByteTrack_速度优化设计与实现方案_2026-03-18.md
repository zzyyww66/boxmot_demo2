# ByteTrack 速度优化设计与实现方案

更新时间: 2026-03-18

适用工程:

- `/root/autodl-tmp/boxmot_demo2`

适用代码基线:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/trackers/bytetrack/basetrack.py`
- `boxmot/trackers/bytetrack/spatial_prior.py`
- `boxmot/utils/matching.py`
- `boxmot/utils/ops.py`
- `boxmot/engine/evaluator.py`
- `tests/unit/test_trackers.py`

相关现状文档:

- `project_notes/ByteTrack_改进版算法设计与实现_2026-03-15.md`
- `project_notes/ByteTrack_最终版全面诊断与整改方案_2026-03-16.md`
- `project_notes/ByteTrack_SOMPT22_最优参数逐项对照_2026-03-18.md`


## 1. 文档目标

这份文档只回答一个问题:

- **在完全不改变测试指标、关联质量、MOT 输出结果的前提下，如何只提升 ByteTrack 的 Association FPS。**

这里的约束是严格的，不是“大体不变”，也不是“允许轻微波动”。

本文档采用的唯一正确优化目标是:

1. `MOTA / HOTA / IDF1 / IDSW / IDs / Frag` 等评估指标不变。
2. 最终逐序列 `mot/*.txt` 跟优化前保持一致，最好做到逐文件字节级一致。
3. 只通过实现层提速，不通过算法退化换速度。
4. 最终只提高 `Association FPS`。


## 2. 严格边界与不可触碰项

### 2.1 本轮允许做的事

本轮只允许以下三类改动:

1. **缓存 / 记忆化**
   - 把同一帧内反复重复计算的几何量缓存起来。
   - 把同一帧内反复构造的中间数组只构造一次。

2. **完全等价的数组化 / 向量化**
   - 公式不变。
   - 阈值不变。
   - 轨迹和检测的顺序不变。
   - 仅把 Python for-loop 改成 NumPy 广播或批量数组计算。

3. **只读型辅助结构**
   - 每帧构造一份 reference cache、bbox cache、center cache、footpoint cache。
   - 不改变状态机，不改变生命周期语义，不改变激活/丢失/恢复规则。

### 2.2 本轮明确禁止做的事

以下事项一律视为**不符合本轮目标**:

1. **禁止调参**
   - 不改 `match_thresh`
   - 不改 `zombie_reid_thresh`
   - 不改 `zombie_match_cost_thresh`
   - 不改 `lost_reid_thresh`
   - 不改 `birth_suppress_*`
   - 不改任何 YAML 默认值

2. **禁止功能退化换速度**
   - 不关 `lost_reid_enabled`
   - 不关 `zombie_reid_enabled`
   - 不关 `spatial_prior_enabled`
   - 不关 `spatial_prior_region_enabled`
   - 不关 `adaptive_zone_enabled`
   - 不关 `exit_zone_enabled`
   - 不减少 `zombie_max_history`
   - 不缩短 `lost_reid_max_frames`

3. **禁止改关联决策面**
   - 不改 Step1 / Step2 / Step4 / Step5 / Step6 的控制流语义
   - 不改候选集
   - 不改 gating 条件
   - 不改 Hungarian/LAP 的输入顺序
   - 不改 `lap.lapjv` 求解器

4. **禁止改数值精度路径**
   - 不把 NumPy `float32` 换成 `float16`
   - 不把关联阶段迁移到近似 GPU kernel
   - 不引入 ANN / top-k 近似截断

5. **禁止不受控地改变顺序**
   - 不允许随意用 `set` / `dict` 打乱 `tracks` / `detections` 顺序
   - 不允许改变 `joint_stracks()`、`sub_stracks()`、`remove_duplicate_stracks()` 的输出顺序语义


## 3. 当前速度问题的真实来源

### 3.1 关键判断

从当前代码结构看，很多人第一反应会认为性能瓶颈在:

- zombie rescue
- recent-lost recovery
- spatial prior

但对当前改进版 ByteTrack 的 `update()` 热路径做局部 profiling 后，可以确认:

- **新增模块确实有额外开销，但不是唯一主瓶颈。**
- 更大的公共开销来自于:
  - bbox 几何格式重复转换
  - `track.xyxy` / `track.tlwh` / `track.footpoint()` 的重复计算
  - `iou_distance()` 前的对象属性展开
  - 每帧大量 `STrack` 对象构造
  - duplicate removal 的重复 IoU 准备
  - birth suppression 的 reference 反复收集

换句话说:

- **现在最值得优化的是“实现层公共开销”，不是“删掉某个增强模块”。**

### 3.2 当前主流程中的热点位置

当前追踪主流程位于:

- `boxmot/trackers/bytetrack/bytetrack.py:1449`

其中主要热点区域是:

1. 高分 / 低分 `STrack` 构造
2. `iou_distance()` 调用前后的 bbox 准备
3. `STrack.xyxy`、`STrack.get_tlwh()`、`STrack.footpoint()`
4. `remove_duplicate_stracks()`
5. `_update_spatial_prior_tracks()`
6. `_is_birth_suppressed()` 与 `_collect_birth_reference_tracks()`

### 3.3 一个重要现实

文档里统计的 `Association FPS`，实际计时点在:

- `boxmot/engine/evaluator.py:896`
- `boxmot/engine/evaluator.py:899`

也就是:

- `tracker.update(dets, img, embs)` 这一段

因此本轮优化的第一优先级应该是:

- `ByteTrack.update()` 内部热路径提速

而不是:

- 数据加载
- 结果写盘
- TrackEval

这些外层 I/O 优化虽然有意义，但不会直接抬高当前文档统计口径下的 `Association FPS`。


## 4. 总体优化策略

本方案把速度优化分成两个层级:

### 4.1 第一层级: 强等价优化

这是本轮必须优先做的部分，特点是:

- 只做缓存
- 只做中间数组复用
- 不改公式
- 不改顺序
- 不改阈值
- 不改求解器

这一层应当作为**默认落地方案**。

### 4.2 第二层级: 受控等价向量化

这是第二批可做项，特点是:

- 数学公式不变
- 结果应当等价
- 但会改变运算组织方式
- 对临界阈值 pair 的浮点舍入更敏感

这一层不是不能做，而是必须在第一层全部完成并通过回归之后，再一项一项进入。


## 5. 具体设计与实现

---

## 5.1 P-speed-1: `STrack` 几何缓存

### 5.1.1 问题

当前 `STrack` 中下面几个接口会被高频调用:

- `xyxy`
- `get_tlwh()`
- `get_tlwh_for_matching()`
- `footpoint()`

这些值本质上都来自同一份状态:

- `self.mean`
- `self.xywh`
- `self.tlwh`
- `self.frozen_mean`

但当前实现里，经常会在同一帧内被反复从头计算，重复走:

- `xywh2xyxy`
- `xywh2tlwh`
- `tlwh2xyah`

这会产生大量不必要的 NumPy copy 和小数组临时对象。

### 5.1.2 设计原则

必须保持:

1. 公式完全一致
2. dtype 路径一致，继续使用当前 `np.float32` / NumPy 标量体系
3. 对外接口不变，避免影响现有单元测试和上层调用

### 5.1.3 具体实现

在 `STrack` 中新增几何缓存字段:

- `_cached_xyxy`
- `_cached_tlwh_live`
- `_cached_footpoint_live`
- `_cached_tlwh_frozen`
- `_geom_cache_valid`
- `_frozen_cache_valid`

新增缓存控制方法:

- `_invalidate_geom_cache()`
- `_invalidate_frozen_cache()`
- `_refresh_live_geom_cache()`
- `_refresh_frozen_tlwh_cache()`

失效点必须覆盖:

1. `__init__`
2. `activate()`
3. `re_activate()`
4. `update()`
5. `predict()`
6. `multi_predict()` 中对每个 track 的 `mean/covariance` 写回后
7. `frozen_mean` 被创建或覆盖时

### 5.1.4 对外接口约束

以下方法对调用方保持原语义:

- `xyxy`
- `get_tlwh()`
- `get_tlwh_for_matching()`
- `footpoint()`

也就是说:

- 外部代码无需知道缓存存在
- 仍然得到与当前实现相同的返回值

### 5.1.5 预期收益

这一步是全局收益项，因为几乎所有关联路径都依赖 bbox 几何量。

它不会改变:

- 匹配关系
- birth / recover / zombie 判定
- MOT 输出

因此是本轮最安全、优先级最高的一刀。

---

## 5.2 P-speed-2: `iou_distance()` 的数组快路径

### 5.2.1 问题

当前 `boxmot/utils/matching.py:46` 的 `iou_distance()` 会在对象输入时做:

- `atlbrs = [track.xyxy for track in atracks]`
- `btlbrs = [track.xyxy for track in btracks]`

这意味着:

1. 每次 IoU 前都会触发一轮对象属性访问
2. 每次属性访问又可能触发 bbox 重新计算
3. Step1 / Step2 / unconfirmed / duplicate removal 全都会重复这套流程

### 5.2.2 设计原则

必须保持:

1. 继续调用同一个 `AssociationFunction.iou_batch`
2. 不改 IoU 公式
3. 不改 cost matrix 的 shape 与行列顺序

### 5.2.3 具体实现

在 `boxmot/utils/matching.py` 中增加内部 helper，例如:

- `_to_xyxy_array_fast(tracks_or_boxes)`

行为要求:

1. 如果输入已经是 `np.ndarray`，直接复用
2. 如果输入是 `STrack` 列表，则直接读取缓存后的 `xyxy`
3. 输出数组顺序与原列表顺序完全一致

然后让 `iou_distance()` 改成:

1. 先统一取出 `atlbrs`
2. 再统一取出 `btlbrs`
3. 仍然调用同一个 `AssociationFunction.iou_batch`

### 5.2.4 风险控制

这里**绝不能**:

- 改成新的 IoU 实现
- 改成不同精度
- 改行列顺序

因为这会直接影响 Hungarian 的输入，进而影响最终关联。

---

## 5.3 P-speed-3: `remove_duplicate_stracks()` 只读取数组缓存

### 5.3.1 问题

`remove_duplicate_stracks()` 位于:

- `boxmot/trackers/bytetrack/bytetrack.py:1846`

它当前每帧都会对:

- `active_tracks`
- `lost_stracks`

再做一次 IoU 去重。

该函数本身语义不能变，但它内部再次调用 `iou_distance()` 时，又会重复展开 track bbox。

### 5.3.2 具体实现

优化方式:

1. 在 `remove_duplicate_stracks()` 内部一次性取出两侧 `xyxy` 数组
2. 直接送入 `iou_distance()` 的数组路径
3. 保持后续:
   - `pairs = np.where(pdist < 0.15)`
   - 生存时间比较
   - 删哪一侧的规则
   - 输出 list 顺序
   全部不变

### 5.3.3 关键约束

绝不能:

- 改 `0.15` 阈值
- 改 duplicate 判定条件
- 改较长轨迹优先保留规则

因为这会直接影响最终输出 txt。

---

## 5.4 P-speed-4: Step4 新生抑制的 reference cache

### 5.4.1 问题

在 `update()` 的 Step4 里，每个 unmatched-high detection 都会重新做:

1. `_collect_birth_reference_tracks()`
2. `_is_birth_suppressed()`

但同一帧内 reference track 集合通常几乎不变。

因此当前存在两类重复:

1. 同一帧重复收集 refs
2. 同一帧对同一批 refs 重复计算 `tlwh_for_matching`、center、distance

### 5.4.2 设计目标

只做“同一帧一次性准备 reference cache”，不改变以下任何逻辑:

- 哪些 track 会进入 refs
- duplicate suppression 的阈值
- 对同一 det 的 suppress 判断结果

### 5.4.3 具体实现

新增一个按帧构造的 reference cache，例如:

- `birth_ref_tracks`
- `birth_ref_tlwh`
- `birth_ref_centers`
- `birth_ref_xyxy`（可选）

处理流程:

1. Step4 开始时收集一次 refs
2. 只在 `activated_starcks` / `refind_stracks` / `lost_stracks` 有变化后，局部刷新或重新构造
3. `_is_birth_suppressed()` 接收 cache，而不是在内部重复取 track 几何量

### 5.4.4 是否允许向量化

允许，但要求如下:

1. 比较公式与当前实现一致
2. `IoU >= birth_suppress_iou` 规则不变
3. `center_dist <= birth_suppress_center_dist` 规则不变
4. 返回值仍然只是 bool

因为 suppress 结果只取 `True/False`，所以在保持公式一致时，数组化不会改变语义。

---

## 5.5 P-speed-5: Spatial prior 的只读缓存与批量收集

### 5.5.1 问题

当前 `_update_spatial_prior_tracks()` 位于:

- `boxmot/trackers/bytetrack/bytetrack.py:465`

其主要开销不是 region 规则本身，而是:

1. 每帧遍历 active tracks
2. 反复算 `footpoint()`
3. 分别构造 `support_points` 与 `birth_points`

### 5.5.2 设计原则

空间先验对当前 SOMPT22 tuned 版本是有效模块，因此:

- 不能关
- 不能降级
- 不能改变 entry/core region 的生成逻辑

### 5.5.3 具体实现

只做以下实现层优化:

1. `footpoint()` 读取 `STrack` 的 live geometry cache
2. `support_points` / `birth_points` 用批量数组收集
3. `_refresh_spatial_region_masks()` 继续保持“dirty 才刷新”的机制

不做以下改动:

- 不改 `spatial_prior_stage`
- 不改 `entry/core` 掩码构建公式
- 不改 `confidence/walkable/birth_density` 的阈值

### 5.5.4 预期收益

收益中等，但非常安全。

它属于“每帧都有、但不改决策”的实现层提速。

---

## 5.6 P-speed-6: 高分 / 低分 detection 预计算与轻量构造

### 5.6.1 问题

当前 high / low detection 在构造 `STrack` 时都会重复做:

- `xyxy2xywh`
- `xywh2tlwh`
- `tlwh2xyah`

这在密集序列中开销并不小。

### 5.6.2 设计目标

不改变 `STrack` 语义，只减少重复转换。

### 5.6.3 具体实现

可以新增一个批量预计算 helper，例如:

- `_prepare_detection_geometry(dets_xyxy)`

一次性生成:

- `xywh_batch`
- `tlwh_batch`
- `xyah_batch`

然后在构造 `STrack` 时直接填入这些值，而不是每个对象单独再做三次转换。

### 5.6.4 风险控制

这里必须继续使用与 `boxmot/utils/ops.py` 完全相同的公式。

更稳妥的实现方式是:

- 直接批量调用与当前公式完全一致的 NumPy 运算
- 或者在 helper 中复用现有 `ops.py` 的同等公式

不能引入不同 rounding 路径。

---

## 5.7 P-speed-7: Recent-lost / Zombie cost builder 的受控向量化

### 5.7.1 适用位置

- `boxmot/trackers/bytetrack/bytetrack.py:1041` `_build_zombie_match_cost()`
- `boxmot/trackers/bytetrack/bytetrack.py:1129` `_build_lost_match_cost()`

### 5.7.2 为什么放在第二阶段

这两处虽然数学上非常适合向量化，但相比缓存优化，风险更高:

1. 会改变浮点运算组织顺序
2. 临界阈值 pair 上理论上存在边界抖动风险

因此本方案建议:

- **先完成 P-speed-1 ~ P-speed-6 并验证输出完全一致**
- 再决定是否推进这一层

### 5.7.3 如果要做，必须满足的约束

1. 保持 `invalid_cost` 完全一致
2. 保持 gate 先后语义一致
3. 保持 `reid_cost > thresh -> continue` 语义一致
4. 保持 `motion_cost` / `shape_cost` / `total_cost` 公式一致
5. 保持 cost matrix 的行列顺序一致

### 5.7.4 推荐策略

如果推进这一层，建议拆成两步:

1. **只向量化 recent-lost**
2. recent-lost 稳定后，再向量化 zombie rescue

每一步都单独做完整输出一致性回归。


## 6. 明确不建议做的“伪优化”

为了避免后续再次走偏，这里把不建议做的方向单列出来。

### 6.1 禁止通过关模块换速度

这些做法虽然会提速，但一定不满足本轮要求:

- 关闭 zombie rescue
- 关闭 lost recovery
- 关闭 spatial prior
- 关闭 birth suppress
- 关闭 adaptive zone

它们不是优化，是退化。

### 6.2 禁止通过缩候选集换速度

例如:

- 最近邻 top-k 截断
- 按距离先砍掉大量 pair
- 只保留部分 zombie / lost 候选

这些策略会改变合法匹配空间，不符合“指标完全不动”的要求。

### 6.3 禁止改变列表顺序

尤其是以下位置必须非常谨慎:

- `joint_stracks()`
- `sub_stracks()`
- `remove_duplicate_stracks()`
- Step4 unmatched high 的处理顺序

原因是:

- Hungarian 输入顺序一旦变化，在 tie case 下就可能改变分配结果。


## 7. 推荐实施顺序

为了最大化“收益 / 风险比”，建议按以下顺序推进。

### 7.1 第一批: 必做

1. `STrack` 几何缓存
2. `iou_distance()` 数组快路径
3. `remove_duplicate_stracks()` 数组化
4. birth suppression reference cache
5. spatial prior footpoint/batch cache

这批改动的共同特点是:

- 完全不改决策面
- 只减少重复计算

### 7.2 第二批: 选做

6. detection geometry 批量预计算
7. `_build_lost_match_cost()` 向量化
8. `_build_zombie_match_cost()` 向量化

这批一定要建立在第一批输出完全一致之后。


## 8. 具体落地文件与职责划分

### 8.1 `boxmot/trackers/bytetrack/bytetrack.py`

本文件负责:

- `STrack` 几何缓存
- high / low detection 预计算
- birth reference cache
- duplicate removal 数组路径
- recent-lost / zombie cost builder 的受控向量化

### 8.2 `boxmot/utils/matching.py`

本文件负责:

- `iou_distance()` 的数组快路径
- `fuse_score()` 的小型广播优化

注意:

- 这里只允许做完全等价改写，禁止更换 IoU 核心逻辑。

### 8.3 `boxmot/trackers/bytetrack/spatial_prior.py`

本文件**原则上不改算法语义**。

除非后续确认有纯实现层冗余，否则不建议修改该文件的概率场规则。

### 8.4 `tests/unit/test_trackers.py`

本文件应补充以下回归测试:

1. `STrack` 几何缓存返回值与原公式一致
2. `remove_duplicate_stracks()` 快路径结果与原逻辑一致
3. birth suppression cache 与逐 track 循环版结果一致
4. recent-lost / zombie 向量化版与旧版 cost matrix 一致（若推进第二批）


## 9. 回归验证方案

这是本方案最重要的一部分。

### 9.1 判定标准

“指标完全不动”不能只看 headline summary，必须采用更强标准:

1. **优先标准: MOT 输出逐文件一致**
2. 如果逐文件一致，则 TrackEval 指标必然一致
3. 在逐文件一致的前提下，只比较 `Association FPS`

### 9.2 推荐验证层级

#### 第一层: 单元测试

至少运行:

```bash
uv run pytest tests/unit/test_trackers.py
```

如果修改了 matching 公共层，建议补充:

```bash
uv run pytest tests/unit/test_trackers.py -k "bytetrack or spatial_prior or zombie or lost"
```

#### 第二层: 共享 cache 下的输出一致性验证

使用相同 detector / ReID / class 过滤，只更换代码，不更换任何配置。

建议流程:

```bash
CACHE_ROOT=/root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17
BASE_RUN=/root/autodl-tmp/boxmot_demo2/runs_speedcheck_baseline_$(date -u +%Y%m%d_%H%M%S)
OPT_RUN=/root/autodl-tmp/boxmot_demo2/runs_speedcheck_opt_$(date -u +%Y%m%d_%H%M%S)

mkdir -p "$BASE_RUN" "$OPT_RUN"
ln -s "$CACHE_ROOT/dets_n_embs" "$BASE_RUN/dets_n_embs"
ln -s "$CACHE_ROOT/dets_n_embs" "$OPT_RUN/dets_n_embs"
```

然后分别在优化前代码与优化后代码下运行:

```bash
uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  bytetrack \
  --source /root/autodl-tmp/boxmot_demo2/train \
  --classes 0 \
  --tracker-config /root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml \
  --device 0 \
  --project "$BASE_RUN" \
  --exist-ok \
  --verbose

uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  bytetrack \
  --source /root/autodl-tmp/boxmot_demo2/train \
  --classes 0 \
  --tracker-config /root/autodl-tmp/boxmot_demo2/boxmot/configs/trackers/bytetrack_sompt22_tuned.yaml \
  --device 0 \
  --project "$OPT_RUN" \
  --exist-ok \
  --verbose
```

#### 第三层: 序列输出逐文件对比

先比较 tracking 输出文件:

```bash
BASE_MOT="$BASE_RUN/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack"
OPT_MOT="$OPT_RUN/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack"

for f in "$BASE_MOT"/*.txt; do
  name="$(basename "$f")"
  cmp -s "$f" "$OPT_MOT/$name" || echo "DIFF: $name"
done
```

或者更强一点，直接比较 hash:

```bash
sha256sum "$BASE_MOT"/*.txt | sort > "$BASE_RUN/sha256.txt"
sha256sum "$OPT_MOT"/*.txt | sort > "$OPT_RUN/sha256.txt"
diff -u "$BASE_RUN/sha256.txt" "$OPT_RUN/sha256.txt"
```

如果这里完全一致，则可以认为:

- `MOTA / HOTA / IDF1 / IDSW / IDs / Frag` 全部一致

#### 第四层: 指标与 FPS 对照

最后再看:

- `person_summary.txt`
- 控制台里的 `Association FPS`

要求:

1. `person_summary.txt` 完全一致
2. `Association FPS` 高于 baseline

### 9.3 如果逐文件不一致怎么办

只要序列输出 txt 有任何差异，就不能宣称“指标完全不动”。

此时处理原则是:

1. 回退最近一个优化点
2. 缩小改动范围
3. 重新做逐文件对比

不能只因为 headline summary 接近，就继续往前推进。


## 10. 建议新增的保障测试

为了让后续提速不再反复触碰指标，建议补以下测试。

### 10.1 几何缓存一致性测试

目标:

- 对同一个 `STrack`，缓存版 `xyxy / tlwh / footpoint` 与旧公式逐项一致。

### 10.2 duplicate removal 一致性测试

目标:

- 同一组 `active/lost` 输入下，优化前后保留 / 删除的轨迹集合完全一致。

### 10.3 birth suppression 一致性测试

目标:

- 在同一组 reference tracks 与 detection 下，优化前后 suppress bool 完全一致。

### 10.4 lost/zombie cost matrix 一致性测试

仅在推进向量化时增加:

- cost matrix shape 完全一致
- 每个元素在 `float32` 路径下与旧版一致


## 11. 预期收益判断

在不改任何决策逻辑的前提下，本轮可期待的收益来源主要有:

1. bbox 几何缓存
2. IoU 前处理去重
3. duplicate removal 数组化
4. birth reference cache
5. detection 预计算

这类优化的共同特点是:

- 对密集序列收益更明显
- 对所有序列都有效
- 不依赖是否触发某个特殊模块

因此它们比“专门优化 zombie rescue 单点”更稳、更值得优先做。


## 12. 最终验收标准

本轮优化完成后，只有在同时满足下面四条时，才能算成功。

### 12.1 功能正确

- `uv run pytest tests/unit/test_trackers.py` 通过

### 12.2 输出完全一致

- SOMPT22 共享 cache 复跑后，逐序列 `mot/*.txt` 一致

### 12.3 指标完全一致

- `person_summary.txt` 一致
- `HOTA / MOTA / IDF1 / IDSW / IDs` 一致

### 12.4 速度明确提升

- `Association FPS` 高于 baseline


## 13. 结论

当前 ByteTrack 改进版要做的，不是“继续删功能换速度”，而是:

- **在不改变任何关联决策的前提下，把实现层重复开销做掉。**

最安全、最符合本轮目标的路线是:

1. 先做 `STrack` 几何缓存
2. 再做 `iou_distance()` 和 duplicate removal 的数组快路径
3. 再做 birth suppression reference cache
4. 再做 spatial prior 的缓存化
5. 最后才考虑 recent-lost / zombie cost builder 的受控向量化

并且整个过程必须以:

- **逐文件 MOT 输出一致**

作为唯一可信验收标准，而不是只看 headline metric 是否“差不多”。

这条线如果严格执行，才符合本轮真正目标:

- **指标完全不动，只增加 Association FPS。**
