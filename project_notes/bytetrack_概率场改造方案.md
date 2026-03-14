# ByteTrack 概率场改造方案（固定城市场景行人 MOT）

## 1. 结论先行

不要把当前“矩形有效区”直接替换成“单张概率图”。

更合适的方案是引入一个**多头空间先验模块**，至少维护 4 张低分辨率概率场：

1. `walkable_field`：人可稳定运动/出现的区域；
2. `birth_field`：新 ID 真实出生热点；
3. `exit_field`：真实离场热点；
4. `occlusion_field`：持久遮挡导致的消失热点。

当前改进版 ByteTrack 中：

- 新生判定主要落在 Step4，核心依赖 `_is_in_entry_zone()`；
- 离开判定主要落在 lost 后的 `_is_in_exit_zone()`；
- 矩形有效区由 `_update_effective_zone()` 单调扩张。

这些逻辑集中在 [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L285) 到 [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L516)，以及 Step4/5/6 的 [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L859) 到 [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L980)。

建议保留 ByteTrack 的主关联流程，只替换“空间门控”和“lost 后处置”两块，不动 Step1/2 的标准关联主干。

## 2. 当前矩形方案的根本缺陷

### 2.1 表达能力不够

当前 `_effective_zone` 只有一个矩形，最多表达：

- “大概哪些地方出现过人”；
- “边缘带更像入口”。

它表达不了：

- 哪一段边缘才是真入口；
- 中心区域哪一块是长期遮挡物前沿；
- 哪些区域虽然在大矩形内，但实际上几乎不会有人走；
- 哪些位置的消失更像“被遮挡”，哪些更像“真正离场”。

### 2.2 单调扩张会把错误永久写进先验

当前 `always_expand` 下，误检或偶发异常框会永久扩张有效区。

这和你的先验相冲突。你的先验是“长期稳定”，不是“只要出现过一次就永久合法”。

### 2.3 birth / lost 的空间统计被混在一起

“新建 ID 多”不一定都代表入口。

如果一个遮挡区边缘经常切断轨迹，也会造成误 birth。反过来，“删除 tracker 多”也不一定都代表出口，其中一部分是长期遮挡前沿。

所以必须把：

- 真出生；
- 真离场；
- 遮挡导致的暂时消失

拆开建模，否则概率场会学偏。

## 3. 最合适的整体结构

### 3.1 不做单场，做多头空间先验

新增模块：

- `SpatialPriorField`

建议新文件：

- [boxmot/trackers/bytetrack/spatial_prior.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/spatial_prior.py)

模块内部至少维护：

- `support_count`: 经过该处的稳定轨迹支持数；
- `birth_count`: 真实 birth 事件计数；
- `exit_count`: 真实 exit 事件计数；
- `occlusion_count`: 遮挡消失事件计数；
- `local_confidence`: 该区域先验是否足够可信。

最终导出：

- `P_walk(x)`
- `P_birth(x)`
- `P_exit(x)`
- `P_occ(x)`
- `C(x)`：局部置信度

### 3.2 观测点不要用整框，优先用行人脚点

对固定监控的行人场景，空间先验的观测位置建议用：

- `footpoint = ((x1+x2)/2, y2)`

而不是整框外接矩形。

原因：

- 行人脚点在图像平面上的空间语义更稳定；
- bbox 上边界受姿态、检测抖动影响大；
- 入口、出口、遮挡边缘，本质上更接近地面接触点的拓扑。

可以对脚点做高斯撒点，而不是硬落格子。

## 4. 事件定义必须“延迟确认”

这是整个方案最关键的地方。

### 4.1 birth 不能在激活当下立刻写入 `birth_field`

应当先记成 provisional birth，只有该 ID 满足以下条件后再提交：

- 连续存活 `birth_commit_age` 帧以上；
- 或累计命中 `birth_commit_hits` 次以上；
- 且没有很快和旧轨迹 merge / duplicate。

否则误检造成的假新生会污染入口图。

### 4.2 lost 也不能立即定性为 exit

当轨迹消失时，先记 provisional disappearance：

- 记录最后脚点；
- 记录消失帧号；
- 记录是否靠近边界；
- 记录后续是否被同 ID re-activate。

只有在观察窗口结束后再归因：

1. 若同 ID 在 `reactivate_window` 内复活，则提交到 `occlusion_field`；
2. 若未复活且末位置接近图像边界/高 `P_exit` 区，则提交到 `exit_field`；
3. 若未复活但也不在边界，则提交到 `unknown_loss`，先不强写进任何主场。

否则“被遮挡”和“走出画面”会混在一起。

## 5. 概率场如何估计

### 5.1 采用低分辨率网格 + 高斯累积

建议不要做全分辨率图，直接用低分辨率网格，例如：

- `grid_w = 48`
- `grid_h = 27`

对于每次轨迹脚点或事件点：

- 找到对应网格中心；
- 以 `sigma_cells` 做高斯核累积到邻域；
- 维护计数图，而不是直接维护概率。

### 5.2 概率不要只看绝对次数，要看条件概率

建议用下面这种形式：

- `P_birth(x) = (birth_count(x) + a) / (support_count(x) + a + b)`
- `P_exit(x)  = (exit_count(x)  + a) / (support_count(x) + a + b)`
- `P_occ(x)   = (occlusion_count(x) + a) / (support_count(x) + a + b)`

理由：

- 只看绝对计数，主通道区域会因为人多导致什么都高；
- 用“事件 / 支持”的条件概率，才能把“经常经过”和“经常在此 birth/lost”区分开。

### 5.3 要有慢速遗忘，不要永久累积

固定监控场景虽然稳定，但不是永远不变。

建议每张计数图都加 EMA 或衰减：

- `count <- decay * count + event`

`decay` 建议在 `0.995 ~ 0.999` 量级，按帧率折算。

这样既体现长期先验，也能缓慢适应场景变化。

## 6. 如何与现有 ByteTrack 融合

## 6.1 冷启动阶段：完全退化为当前改进版

这个阶段不要强用概率场，只学习，不决策。

继续使用现有：

- `_is_in_entry_zone()`
- `_is_in_exit_zone()`
- `birth_confirm_frames`
- `birth_suppress_*`
- zombie rescue

也就是保留现在的 [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L675) 到 [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L980) 的主流程。

### 6.2 进入混合阶段，不要硬切换

不要到某一帧后把矩形逻辑完全关闭。

建议定义一个全局成熟度：

- `global_ready`
- `blend_alpha in [0, 1]`

例如由以下条件共同决定：

- 已提交的 confirmed birth 数量足够；
- 已提交的 confirmed disappearance 数量足够；
- `support_field` 覆盖的有效区域足够；
- 最近一段时间地图变化很小，说明先验趋于稳定。

当 `blend_alpha` 从 0 缓慢升到 1 时：

- `alpha = 0`：完全是当前改进版；
- `0 < alpha < 1`：矩形规则和概率场融合；
- `alpha = 1`：概率场主导，仅在低置信区域回退。

### 6.3 Step4 新生逻辑的推荐改法

现有 Step4 在这里：

- [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L859)

建议把 `_is_in_entry_zone()` 的布尔判定，改成一个连续分数：

- `rect_entry_score`
- `field_birth_score`
- `field_walk_score`
- `field_occ_score`

最终得到：

`spawn_score = (1-alpha) * rect_score + alpha * field_score`

其中一个可落地的 `field_score` 是：

`field_score = w1 * P_birth(x) + w2 * P_walk(x) - w3 * P_occ(x) - w4 * P_exit(x)`

决策规则建议：

1. `C(x)` 低时，直接回退当前矩形逻辑；
2. `P_walk(x)` 很低且 `C(x)` 高时，拒绝新建；
3. `P_birth(x)` 高时，即使不在旧矩形边缘，也允许新建；
4. `P_occ(x)` 高时，不要立刻新建，优先：
   - 先尝试 zombie / lost 复活；
   - 提高 `birth_confirm_frames`；
   - 或进入更严格的 pending birth。

这一步是减少误切 ID 的核心。

### 6.4 Step5/6 lost 处置的推荐改法

现有 lost 和 remove 逻辑在：

- [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L952)

建议按最后脚点位置，给每条 lost 轨迹一个 disappearance type：

- `exit_like`
- `occlusion_like`
- `unknown`

判定可用：

- `P_exit(last_point)`
- `P_occ(last_point)`
- 是否靠近图像边界

然后动态调整：

- `exit_like`: 更短 remove grace，更少 zombie 保留；
- `occlusion_like`: 更长 lost 保留，更大 zombie match 距离，更晚删除；
- `unknown`: 保持默认。

也就是说，不要再只靠统一 `exit_zone_margin` 和统一 `exit_zone_remove_grace`。

### 6.5 zombie rescue 也要接概率场

当前 zombie 只看中心距，见：

- [boxmot/trackers/bytetrack/bytetrack.py](/home/zyw/code/boxmot_demo2/boxmot/trackers/bytetrack/bytetrack.py#L561)

建议改成：

- 在高 `P_occ` 区域，提高 zombie 匹配容忍度；
- 在高 `P_birth` 区域，降低把 unmatched detection 硬匹配回旧 zombie 的倾向；
- 对明显不可行走区域，直接降低匹配优先级。

第一版不需要引入复杂路径搜索，只需要把阈值做成位置相关即可。

## 7. 最关键的一点：把“区域分类”改成“事件归因”

你当前的直觉是对的，但还差一步。

真正该学的不是“这个矩形是不是入口”，而是：

- 这里如果发生 unmatched detection，更像真出生还是误切；
- 这里如果发生 lost，更像遮挡还是离场；
- 这里本身是不是长期可通行。

所以概率场不应当直接替代矩形区域，而应当替代“空间决策的依据”。

## 8. 推荐的状态机

每条轨迹新增两个延迟事件状态：

1. `provisional_birth`
2. `provisional_disappearance`

推荐流程：

1. 新 ID 激活时，记录 `birth_point`，不立刻写 `birth_field`；
2. 轨迹稳定后，回填 `birth_field`；
3. 轨迹 lost 时，记录 `loss_point` 和 `loss_frame`；
4. 若后续 re-activate，则把该事件记为 `occlusion_field`；
5. 若窗口结束仍未回来，则再记为 `exit_field` 或丢到 `unknown_loss`。

这一步比“直接统计新建和删除位置”更稳健。

## 9. 建议的工程改造顺序

不要一次把所有逻辑改完。最稳的顺序如下。

### 阶段 A：先把概率场作为旁路学习器接入

目标：

- 不改变当前输出；
- 只记录统计；
- 支持可视化。

要做的事：

1. 新增 `SpatialPriorField`；
2. 在 `update()` 中记录 confirmed track 脚点；
3. 对 birth / lost / re-activate 做延迟提交；
4. 输出调试图：
   - walkable heatmap
   - birth heatmap
   - exit heatmap
   - occlusion heatmap

只有这一步做完，你才知道先验学得对不对。

### 阶段 B：只用概率场改造新生门控

目标：

- 先控制误 birth / 误切 ID；
- 不碰主关联。

做法：

1. 保留原 `_try_activate_new_track()`；
2. 在其前面加一个 `score_new_track_candidate()`；
3. 根据位置动态决定：
   - 是否允许新建；
   - 使用几帧确认；
   - 是否优先 zombie rescue。

### 阶段 C：再改造 lost/remove/zombie

目标：

- 用 `P_occ` / `P_exit` 控制 lost 生命周期；
- 进一步减少遮挡导致的删轨和重建。

### 阶段 D：最后再废掉矩形有效区

只有当概率场稳定、可视化结果正确、指标收益明确后，才建议逐步下掉：

- `_effective_zone`
- `_is_in_entry_zone()`
- `_update_effective_zone()`

在这之前，矩形逻辑应该保留为 fallback。

## 10. 配置建议

建议新增一组参数，而不是直接复用 `adaptive_zone_*`：

- `spatial_prior_enabled`
- `spatial_prior_grid_w`
- `spatial_prior_grid_h`
- `spatial_prior_decay`
- `spatial_prior_sigma`
- `spatial_prior_min_support`
- `spatial_prior_min_birth_events`
- `spatial_prior_min_loss_events`
- `spatial_prior_reactivate_window`
- `spatial_prior_birth_commit_age`
- `spatial_prior_local_conf_thresh`
- `spatial_prior_blend_frames`
- `spatial_prior_use_rect_fallback`

不要把概率场继续命名成 `adaptive_zone`，因为它已经不是“区域框”问题了。

## 11. 我建议你避免的两个坑

### 11.1 不要直接用原始检测更新概率场

应优先用：

- confirmed tracks；
- 已被验证的 birth / occlusion / exit 事件。

否则 detector 的瞬时误检会直接污染长期先验。

### 11.2 不要把高 lost 区域直接视为出口

这是最容易学歪的地方。

必须用“是否 later reappear”来区分：

- 高 lost + 高频 reappear = 遮挡区；
- 高 lost + 低 reappear + 靠边 = 出口；
- 高 lost + 低 reappear + 不靠边 = 先记 unknown，不急着定性。

## 12. 对你这个场景最值得先做的版本

如果只做一版最值当、风险最低的改造，我建议是：

1. 用脚点建立 `walkable/birth/occlusion/exit` 四场；
2. 事件全部延迟确认后再写图；
3. 先只改 Step4 新生门控；
4. `P_occ` 高的位置延长确认、优先复活旧 ID；
5. `P_birth` 高的位置放宽新建；
6. 低置信区域继续退化到当前改进版；
7. 等可视化确认先验学对，再去改 lost/remove 策略。

这是最符合你给出的核心先验、同时又不容易把 ByteTrack 主体搞坏的路线。
