# ByteTrack SOMPT22 达标整改总方案

更新时间: 2026-03-15

适用代码基线:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/trackers/bytetrack/basetrack.py`
- `boxmot/trackers/bytetrack/spatial_prior.py`
- `boxmot/configs/trackers/bytetrack_improved.yaml`
- `boxmot/trackers/tracker_zoo.py`
- `boxmot/engine/evaluator.py`
- `tests/unit/test_trackers.py`

本文档是面向“把当前固定城市场景版 ByteTrack 在 SOMPT22 上推进到目标指标”的主整改方案。它不是单点 bug 记录，也不是纯思路草稿，而是后续若干轮开发、实验、验证的执行说明书。

## 1. 目标与当前现实

### 1.1 最终目标

当前目标是让系统在 SOMPT22 上尽量逼近并稳定达到以下区间:

- `HOTA >= 54`
- `MOTA >= 67`
- `IDF1 >= 67`
- `IDSW < 600`
- 纯关联 `FPS > 100`

### 1.2 当前已验证基线

当前 fresh eval 的已验证结果为:

- `HOTA = 53.023`
- `MOTA = 65.710`
- `IDF1 = 66.179`
- `IDSW = 815`
- `Association FPS = 18.9`

对应 summary 文件:

- `runs_fullrerun_20260315_094907/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack/person_summary.txt`

### 1.3 必须先讲清楚的一件事

这组目标里，`MOTA 67` 不是一个“只靠少做 IDSW 就能达到”的目标。

从当前 summary 可知:

- `CLR_FN = 137202`
- `CLR_FP = 45746`
- `IDSW = 815`
- `GT_Dets = 535904`

MOTA 公式本质上由:

- `FN`
- `FP`
- `IDSW`

共同决定。

把 `IDSW` 从 `815` 降到 `600`，只会带来约 `0.04` 个 MOTA 点的提升，远不足以把 `65.710` 推到 `67.0`。因此:

- 想达标，必须同时改善身份保持和召回/误检
- 不能把问题只看成“zombie ReID 不够强”

这件事决定了后续整个整改方向:

- `IDSW/IDF1/HOTA` 主要靠关联与身份管理改善
- `MOTA` 还必须靠 `FN/FP` 的系统性下降来实现

## 2. 当前算法是什么

当前实现不是原始 ByteTrack，而是一个“ByteTrack 主干 + 场景生命周期偏置 + 长时记忆复活 + 在线空间先验”的组合系统。

可以分成四层:

### 2.1 主干关联层

保留了 ByteTrack 的基本结构:

1. 高分检测与 `tracked + lost` 做第一轮 IoU 关联
2. 低分检测与剩余 `tracked` 做第二轮 IoU 关联

这一层决定大多数短时跟踪稳定性。

### 2.2 生命周期控制层

新增:

- `entry zone`
- `exit zone`
- `birth confirm`
- `birth suppression`
- `adaptive effective zone`

这一层主要决定:

- 新 ID 何时创建
- 目标何时解释为离场
- 未匹配检测是当作“新生”还是“旧 ID 回归”

### 2.3 长时复活层

新增:

- `lost -> zombie` 转移
- `frozen_mean`
- `zombie rescue`
- ReID 只用于 zombie 阶段

这一层主要负责:

- 原始 ByteTrack 会丢掉的长时断链身份是否能被恢复

### 2.4 在线空间先验层

新增:

- `SpatialPriorField`
- `support_count`
- `birth_count`
- learned `entry/core` masks

这一层的目标是:

- 在固定监控场景中，从长期统计上学习“人通常从哪里出现”和“哪些区域是长期核心区”

## 3. 当前算法的总体评判

### 3.1 做对了什么

当前版本有几个方向是明确正确的。

#### 3.1.1 没有直接把 ReID 扩散到整个 ByteTrack 主干

这是对的。

原因:

- 纯 appearance 接管主干关联，回归风险大
- 固定监控场景里的误差往往不是“所有地方都需要外观”，而是某些高风险阶段需要外观

当前“先保 ByteTrack 主干稳定，再在高风险环节加 ReID”的基本原则是合理的。

#### 3.1.2 试图把 fixed-scene 先验显式引入生命周期

这也是对的。

固定城市场景里，下面两个事实通常都成立:

- 真正新生目标更可能出现在特定入口区域
- 边缘或遮挡边界附近消失更可能是离场或临时不可见

因此把:

- new birth
- exit
- long-gap reappearance

从统一的一套通用逻辑里拆开，本身是正确方向。

#### 3.1.3 尝试用空间统计代替纯几何边缘带

这也是正确方向。

原因:

- 固定摄像头下的真实 entry region 不一定等于图像边缘带
- 场景内部也可能存在“重新可见热点”
- 使用 support/birth 分布学习 entry/core 比写死 margin 更有潜力

### 3.2 根本问题不在单个参数，而在“系统没有形成统一控制流”

当前实现最大的问题不是某个阈值错了，而是:

- ByteTrack 主干
- adaptive zone
- learned spatial prior
- birth confirm
- zombie ReID

这些东西都在发挥作用，但没有形成一个完全自洽、层次清晰、职责明确的系统。

结果是:

- 某些地方过于保守，导致漏检和漏新生
- 某些地方又过于容易新生，导致重复 ID
- 某些地方本该先复活旧 ID，却被其他规则抢先放行成新 ID
- 某些场景下空间先验会把 tracker 自己的错误继续强化

## 4. 当前版本的主要症状

基于现有结果和代码检查，当前版本至少有五类明显症状。

### 4.1 身份错误不只是长时断链问题

当前 zombie ReID 的设计主要针对:

- 长时 lost
- center-zone unmatched high

但 SOMPT22 中大量真正影响 `IDSW` 的 case 很可能来自:

- 5 到 20 帧的中短时遮挡
- 近邻交叉
- 密集人群中的短时漂移
- 低分续命后再次稳定

这些错误在进入 zombie 阶段之前就已经发生，因此单独继续改 final zombie ReID，收益上限有限。

### 4.2 漏检问题对 MOTA 影响远大于 IDSW

当前 MOTA 的主要误差来源是:

- `FN` 为主
- `FP` 次之
- `IDSW` 占比很小

这意味着后续算法改造如果只关注:

- old ID rescue
- ID switch reduction

而不同时改善:

- medium-confidence reappearance
- hard sequence 里的 birth / recall

那么 `MOTA 67` 基本不现实。

### 4.3 hard sequence 呈现两种不同失败模式

从分序列结果看，当前问题不是单一模式。

大致可分成两类:

#### 类型 A: 以身份问题为主

典型序列:

- `SOMPT22-08`
- `SOMPT22-10`

这些序列 `IDSW` 很高，说明:

- crowded / interaction 密度高
- 中短时混淆严重
- 仅靠 IoU 主干不够

#### 类型 B: 以召回问题为主

典型序列:

- `SOMPT22-07`
- `SOMPT22-13`

这些序列 `MOTA` 很差，但不完全是 `IDSW` 导致，更像:

- high/medium confidence 检测比例不足
- 新生和重现被过度保守地压制
- 漏检和漏恢复更严重

也就是说:

- 当前算法不是只有一个瓶颈
- 它同时卡在“身份保持”和“召回保守”两端

## 5. 当前实现中的关键问题总表

下面按优先级和性质分组。

### 5.1 P0: 状态正确性问题

这是最优先修复项。否则后续所有实验都不可靠。

#### 5.1.1 `track.tlwh` 状态一致性风险

当前 `STrack.tlwh` 在初始化时写入，但 `update()` 与 `re_activate()` 中没有明确同步更新。

若 `tlwh` 不是动态属性，而只是旧值，则会直接影响:

- exit-zone 判定
- birth suppression
- 依赖 tlwh 的几何逻辑

特别是 exit-zone，如果基于旧框而不是当前框:

- 该进 zombie 的轨迹可能被误删
- 该删除的轨迹可能被误保留

这是 correctness 问题，不是调参问题。

#### 5.1.2 `frozen_mean` 生命周期残留风险

当前 `frozen_mean` 被写入后，在重新激活时没有看到统一清空逻辑。

这意味着:

- 轨迹经历过一次长时丢失
- 后面重新激活
- 再次 lost 时可能继续用旧 frozen state

这种残留状态会直接破坏 zombie 阶段的几何门控。

#### 5.1.3 lost/zombie 专用状态没有统一 reset

除了 `frozen_mean` 之外，以下字段也需要在重新激活时重新定义语义:

- `lost_frame_id`
- `exit_pending`
- 可能还包括与 spatial birth 相关的局部状态

当前实现更像是“依靠局部逻辑维持正确”，而不是“通过统一 reset 保证状态干净”。

### 5.2 P1: 目标分解与流程设计问题

#### 5.2.1 ReID 介入得太晚

当前 ReID 只在 zombie rescue 中启用，这会漏掉大量真正重要的中短时身份错误。

这不是说要把 ReID 直接灌进 Step1 主干，而是说:

- 应当在 `late-lost` 阶段增加一层窄而保守的 ReID rescue
- 不能等到 `lost >= 30` 才第一次允许 appearance 参与

#### 5.2.2 `lost` 和 `tracked` 共享同一轮纯 IoU 匹配

当前高分主关联把:

- `tracked`
- `lost`

直接合并成一个池子做 IoU + score 匹配。

对于追求低 `IDSW` 的 fixed-scene pedestrian MOT 来说，这太粗了。

问题在于:

- recent lost 与 current tracked 的误抢检测风险高
- crowded scene 里最近 lost 的轨迹会与当前邻近 tracked 发生竞争
- 纯 IoU 下很容易出现身份错抢

#### 5.2.3 low-score 路径只保生命周期，不保 appearance

Step2 里的低分检测不带特征，这导致:

- 低分观测能延续 track
- 但不能更新外观记忆

这会让后续 ReID 使用的特征滞后于真实外观变化。

#### 5.2.4 当前算法没有为不同时间尺度定义不同的恢复策略

目前基本只有两档:

- 短时: 纯 IoU 恢复
- 长时: zombie ReID 恢复

缺少中间层:

- medium-gap rescue

而 medium-gap 往往是 fixed-scene crowded pedestrian MOT 里最关键的身份错误来源。

### 5.3 P2: 区域逻辑互相纠缠

#### 5.3.1 adaptive zone 没有在 spatial prior 成熟后真正退出

当前并不是“概率场成熟前用 adaptive zone，成熟后完全切换到概率场”。

真实情况是:

- `_is_in_entry_zone()` 在 learned region ready 后优先使用概率场
- 但 `adaptive zone` 还在继续更新 effective zone
- `outside-before-expand` 依然继续生效

这意味着是“部分接管 + 部分并存”，不是完全切换。

#### 5.3.2 `outside-before-expand` 会绕过 learned spatial prior

一旦 detection 被标成 `_outside_zone_det_inds`:

- 它会直接尝试新生
- 不会优先经过概率场的 `entry/core` 判断
- 也不会优先尝试 old ID rescue

因此:

- spatial prior 认为是 core 的点
- adaptive zone 仍可能因为 outside-before-expand 把它直接放行成新生

这属于控制流冲突。

#### 5.3.3 entry 与 exit 语义不对称

当前:

- entry 可以学
- exit 仍然完全几何化

但 fixed-scene 里真实 exit 不一定只在边缘带。

这会导致:

- 新生逻辑越来越场景化
- 离场逻辑依然比较粗糙

生命周期解释仍然不平衡。

### 5.4 P3: 概率场逻辑存在自增强风险

#### 5.4.1 birth map 来自 tracker 自己的激活结果

当前 birth 事件不是来自 GT，而是来自 tracker 成功激活并稳定存活的 track。

一旦某个区域因为误恢复失败而错误地产生了新 ID，并且活过 commit 门限:

- 该区域就会被写入 birth map
- 后续更容易被解释成合法 entry 区域

这会让错误具备“越错越像对”的自增强特征。

#### 5.4.2 spatial prior 成熟门槛过低

默认:

- `support >= 100`
- `birth >= 8`

就进入 `entry_only`

这对长序列来说过早，尤其在前期 tracker 本身还不稳定时。

#### 5.4.3 `confidence` 语义更像计数而不是真置信度

概率场里 `confidence = support_count + birth_count`

这意味着:

- 它更像累计质量
- 而不是归一化置信度

若直接拿这个量做阈值，会让阈值解释依赖序列长度和前期偏差。

### 5.5 P4: zombie ReID 本身也不够强

#### 5.5.1 现在的 zombie rescue 不是 appearance-led recall

当前 zombie 匹配流程是:

1. 面积门控
2. 距离门控
3. shape 门控
4. ReID 门控
5. 代价融合排序

因此 ReID 更像“末端裁决”，不是“主要召回手段”。

#### 5.5.2 小框直接无法复活

当前 `zombie_reid_min_box_area` 的语义不是“跳过 ReID，回退 motion-only”，而更像:

- 这个 pair 直接不参与 zombie rescue

这会系统性损失远处和小尺度目标的恢复机会。

#### 5.5.3 只用单个 `smooth_feat` 原型太粗

当前 appearance 表示主要依赖 EMA 原型。

问题是:

- 不同姿态混合
- 遮挡样本污染
- 局部模糊样本污染

对固定监控下可重复姿态的人来说，单原型不够。

## 6. 后续整改的总原则

为了尽量高概率达标，后续整改必须遵循以下原则。

### 6.1 先修 correctness，再做结构优化

任何建立在错误状态上的实验，都会把你带偏。

### 6.2 不要再继续堆平行机制

当前最大的问题之一就是:

- adaptive zone
- spatial prior
- birth confirm
- zombie rescue

都在发挥作用，但没有统一仲裁逻辑。

后续改法必须朝着:

- 更少的分支
- 更清晰的职责
- 更稳定的优先级

收敛。

### 6.3 按时间尺度拆恢复策略

不要把所有恢复都放在同一种机制里。

建议至少拆三档:

1. `tracked` 短时连续: 继续以运动 + IoU 为主
2. `late-lost` 中短时恢复: 保守地引入 ReID
3. `zombie` 长时恢复: 更强的 ReID + 场景门控

### 6.4 在线 learned prior 只能做偏置，不能过早做硬规则

在 tracker 自己产生训练信号的情况下，任何 learned prior 都容易自增强错误。

因此:

- 初期只能做 soft bias
- 不能在成熟很早时就直接接管强决策

### 6.5 每一步都必须可观测

后续任何阶段都不能再靠“感觉好像变好了”。

必须增加统计，至少回答:

1. 哪类样本进了哪个阶段
2. 失败主要死在哪层 gate
3. hard sequence 里到底是 recall 问题还是 identity 问题

## 7. 最优先的重构方向

下面不是“能做什么”的清单，而是“应该按照什么顺序做”的路线。

## 7.1 阶段 A: 修状态与数据一致性

这是第一个必须完成的阶段。

### A1. 统一 `STrack` 的几何状态表达

目标:

- 所有依赖当前框的逻辑都读取同一套真实当前状态

建议实现:

1. 把 `tlwh` 改成动态属性
2. 统一从 `mean` 推导 `xywh/tlwh/xyxy`
3. 检查所有 `track.tlwh` 使用点

必须检查的逻辑:

- exit-zone
- birth suppression
- zombie matching
- pending birth 匹配

### A2. 统一 reset 生命周期残留状态

在以下函数中统一 reset:

- `activate()`
- `update()`
- `re_activate()`

建议重置:

- `frozen_mean = None`
- `lost_frame_id = 0`
- `exit_pending = False`

并明确每个字段在“再次进入 lost/zombie”时由谁重新设置。

### A3. 新增测试

必须新增:

1. `stale_tlwh_does_not_affect_exit_zone_decision`
2. `reactivated_track_clears_frozen_state`
3. `reactivated_track_uses_current_geometry_for_matching`

## 7.2 阶段 B: 建立完整观测统计

这是第二个必须完成的阶段。

### B1. 对 Step4 增加 gate 级统计

至少记录:

- `step4_unmatched_high_total`
- `step4_outside_bypass_birth`
- `step4_entry_birth`
- `step4_center_candidates`
- `step4_zombie_pairs_total`
- `step4_dist_gate_pass`
- `step4_shape_gate_pass`
- `step4_reid_gate_pass`
- `step4_zombie_rescue_success`
- `step4_center_fallback_birth`

### B2. 对恢复 gap 做分桶

恢复事件按 gap 分成:

- `<5`
- `5-10`
- `10-20`
- `20-30`
- `>30`

目的:

- 判断 IDSW 主要集中在哪个时间尺度

### B3. 对序列输出简化摘要

每个序列输出:

- FN
- FP
- IDSW
- rescue 成功数
- outside bypass 数
- entry birth 数
- fallback birth 数

优先盯:

- `SOMPT22-08`
- `SOMPT22-10`
- `SOMPT22-07`
- `SOMPT22-13`

## 7.3 阶段 C: 引入 late-lost ReID rescue

这是最关键的结构性改动，也是最有希望明显降低 `IDSW` 的部分。

### C1. 目标

解决当前只有:

- 短时纯 IoU
- 长时 zombie ReID

两档恢复的问题。

### C2. 新增第三层恢复

建议在 Step1 和 Step4 之间插入一层:

- `late-lost rescue`

输入:

- Step1/Step2 后未匹配的高分检测
- `lost_stracks` 中 gap 处于中短区间的轨迹

例如:

- `5 <= frames_lost < zombie_transition_frames`

### C3. 这一层的原则

- 不替代主干 IoU
- 只服务于 medium-gap 的困难样本
- gate 比 zombie 更严格
- 但比纯 IoU 更允许 appearance 发挥作用

### C4. 推荐代价结构

推荐:

- footpoint distance
- age penalty
- ReID cost
- 可选轻量 shape cost

不建议:

- 继续单纯用 bbox center + hard freeze

### C5. 实现落点

建议新增函数:

- `_match_late_lost_tracks(...)`
- `_build_late_lost_match_cost(...)`

并在 Step4 之前插入:

1. 先做 late-lost rescue
2. 再处理剩余 unmatched high 的 new birth / zombie rescue

## 7.4 阶段 D: 重做区域控制流

当前区域控制流需要从“并行叠加”改成“统一仲裁”。

### D1. 明确谁才是 entry 语义的最终来源

推荐方案:

- 概率场成熟前: 允许 adaptive zone 提供弱入口偏置
- 概率场成熟后: spatial prior 接管 entry/core 语义
- 此时 adaptive zone 不再参与 new-birth 决策，只保留可选的可见区域统计功能

也就是说:

- `outside-before-expand` 不应继续拥有绕过 spatial prior 的强放行能力

### D2. 重新定义 Step4 决策顺序

建议统一成:

1. 判断是否属于“新进入画面”的强证据
2. 若不是，先尝试 old-ID rescue
3. rescue 失败后再考虑新生
4. learned prior 只提供 bias，不单独制造捷径分支

### D3. entry-zone 不再直接跳过 old-ID rescue

当前 entry-zone 直接新生太激进。

建议改成:

- entry-zone 先查一个较严格的小候选旧轨迹池
- 只有 old-ID rescue 明确失败，才允许新生

## 7.5 阶段 E: 把 zombie gate 从“几何硬挡板”改成“轨迹化门控”

当前 zombie gate 对 fixed-scene pedestrian MOT 不够合适。

### E1. 用脚点替代框中心

行人目标在固定监控下:

- 脚点比框中心稳定
- 框中心更受尺度变化、截断、姿态影响

因此:

- 匹配距离优先用 footpoint

### E2. gate 半径按 gap 自适应扩大

当前固定距离门限不够合理。

建议:

- gap 越大，门限越宽
- 但不是无上限增长
- 可以和场景区域密度共同限制

### E3. 从单点冻结改为轨迹走廊

当前 `frozen_mean` 更像单点冻结。

建议:

- 保存丢失前最后若干帧脚点
- 基于速度方向和场景约束构建 corridor
- corridor 内 pair 放宽门控

## 7.6 阶段 F: 升级 ReID 表示

### F1. 从单原型升级到小型 gallery

建议:

- 每条轨迹维护 `K=5~10` 个高质量 embedding
- 同时保留一个平滑原型

匹配时使用:

- `min distance`
- 或 `top-k mean`

而不是只看单个 `smooth_feat`

### F2. 加入质量筛选

只有满足以下条件的 observation 才更新 gallery:

- 框面积足够
- 置信度较高
- 非严重截断
- 形状稳定

### F3. 低分续命后的 appearance 处理

需要决定:

- 是否允许 low-score detection 更新 appearance
- 若允许，应如何做质量保护

建议方案:

- 不直接让所有 low-score 更新 prototype
- 但允许中等置信度、面积足够的低分框进入 gallery 候选

## 7.7 阶段 G: 重新处理召回问题

要到 `MOTA 67`，必须显式处理召回。

### G1. 重新评估 `new_track_thresh`

当前默认:

- `track_thresh = 0.5`
- `new_track_thresh = 0.65`

在 hard sequence 中可能过高。

建议:

- 让 `new_track_thresh` 不再固定只高于 `track_thresh`
- 考虑按区域和阶段自适应

例如:

- entry 区域可以较低
- core 区域若经过 rescue 失败，再允许较保守的新生

### G2. 允许 medium-confidence reappearance 进入恢复逻辑

当前 Step4 只看 unmatched high。

这对 hard sequence 召回不够。

可考虑:

- 把 `0.4~0.5` 之间的一部分 detection 也纳入 late-lost rescue 候选

前提是:

- gate 足够严格
- 统计充分

### G3. 重新审视 birth confirm 对 hard sequence 的副作用

`birth_confirm_frames=2` 在一般情况下合理，但在低帧率、遮挡重的场景里可能压制真正的新生和重新可见。

建议:

- 不是全局关闭
- 而是按区域、按置信度、按 prior 区域动态调整

## 8. 推荐的最终系统结构

如果要实现较高概率达标，建议最终形成如下分层结构。

### 层 1: 主干跟踪

- `tracked` 与高分检测做 IoU 主关联
- 保持原始 ByteTrack 的稳定优势

### 层 2: late-lost rescue

- 对 medium-gap lost 轨迹启用保守 appearance rescue
- 专治中短时遮挡和交叉混淆

### 层 3: zombie rescue

- 对 long-gap lost 启用更强 appearance + 场景化门控
- 专治长时断链

### 层 4: birth decision

对于所有 rescue 失败的 unmatched detection:

1. 判断是否具有强新生证据
2. 结合 spatial prior 与区域规则决定是否允许新生
3. 再应用 duplicate suppression / confirmation

### 层 5: online prior update

注意:

- prior 更新只基于足够稳定的轨迹
- 初期作为 soft bias
- 成熟后逐步接管区域语义
- 不允许单次错误轻易改变 entry/core 结构

## 9. 具体实现顺序

下面是建议的实际开发顺序。

### 第 1 步: 修状态正确性

修改文件:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `tests/unit/test_trackers.py`

完成项:

- 统一 `tlwh` 当前状态
- 统一 reset zombie/lost 残留状态
- 补单测

### 第 2 步: 加统计与调试输出

修改文件:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/engine/evaluator.py`

完成项:

- gate 级计数器
- gap 分桶
- 序列级摘要输出

### 第 3 步: 加 late-lost rescue

修改文件:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/configs/trackers/bytetrack_improved.yaml`
- `tests/unit/test_trackers.py`

完成项:

- 新 cost build
- 新匹配阶段
- 新参数
- 新测试

### 第 4 步: 整理 adaptive zone 与 spatial prior 的职责

修改文件:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `project_notes/` 对应说明文档
- `tests/unit/test_trackers.py`

完成项:

- 移除冲突逻辑
- 统一 entry 决策顺序
- 降低 outside-before-expand 的强捷径作用

### 第 5 步: 重做 zombie gate

修改文件:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `tests/unit/test_trackers.py`

完成项:

- footpoint gate
- age-aware gate
- corridor 或轨迹化门控

### 第 6 步: 升级 ReID memory

修改文件:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `tests/unit/test_trackers.py`

完成项:

- gallery memory
- quality filter
- 新匹配规则

### 第 7 步: 重新调召回相关参数

修改文件:

- `boxmot/configs/trackers/bytetrack_improved.yaml`
- 可选调参脚本或实验脚本

完成项:

- `new_track_thresh`
- late-lost 候选阈值
- birth confirm 规则
- medium-confidence reappearance 的纳入策略

## 10. 试验设计要求

每做完一阶段，都必须按下面方式验证。

### 10.1 单测

至少运行:

```bash
uv run pytest tests/unit/test_trackers.py
```

对重大新逻辑，应新增针对性测试，不允许只靠全量 eval 感觉判断。

### 10.2 小规模回归

建议先在高风险序列上做定向验证:

- `SOMPT22-08`
- `SOMPT22-10`
- `SOMPT22-07`
- `SOMPT22-13`

目标:

- 快速观察是身份变好，还是召回变差

### 10.3 全量 fresh eval

必须使用全新 `--project`，避免旧 embedding cache 干扰。

### 10.4 每轮记录必须包含

- summary 指标
- 每序列 `FN/FP/IDSW`
- rescue 统计
- birth 统计
- gate fail 分布

## 11. 验收标准

### 阶段 A-B 验收

- 状态正确
- 统计可用
- 不再盲调

### 阶段 C-D 验收

- `IDSW` 明显下降
- `IDF1` 提升
- `HOTA` 提升
- 不出现明显 `MOTA` 崩溃

### 阶段 E-G 验收

- `FN` 显著减少
- `FP` 可控
- hard sequence 有明确改善
- 逐步逼近总目标

## 12. 最终判断

若要以最高概率达成目标，正确路线不是继续围绕“最终 zombie ReID”做局部调参，而是按以下顺序系统整改:

1. 修状态正确性
2. 建立统计观测
3. 在 `late-lost` 阶段前移 ReID
4. 统一区域控制流
5. 重做 zombie 几何门控
6. 升级 ReID memory
7. 再解决 hard sequence 的召回问题

一句话总结:

当前算法的核心问题不是“某个模块不够强”，而是“整个系统还没有形成统一的时间尺度恢复机制和统一的区域仲裁逻辑”。只有先把系统收敛成一个一致、可解释、可观测的控制流，后续冲 `HOTA 54 / MOTA 67 / IDF1 67 / IDSW < 600` 才有较高概率成功。
