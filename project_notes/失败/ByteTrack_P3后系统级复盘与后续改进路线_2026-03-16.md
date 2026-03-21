# ByteTrack P3 后系统级复盘与后续改进路线

更新时间: 2026-03-16

适用工程:

- `/root/autodl-tmp/boxmot/boxmot_demo2`

适用代码基线:

- `boxmot/trackers/bytetrack/bytetrack.py`
- `boxmot/trackers/bytetrack/basetrack.py`
- `boxmot/trackers/bytetrack/spatial_prior.py`
- `boxmot/configs/trackers/bytetrack_improved.yaml`
- `tests/unit/test_trackers.py`

适用结果基线:

- 旧 fresh baseline: `runs_fullrerun_20260315_094907/`
- P0-P3 后当前结果: `runs_reusecache_eval_20260316_043347/`

本文档不是对前两份诊断文档的重复转述，而是在以下三部分基础上形成的后续推进主文档:

1. 对 `project_notes/ByteTrack_最终版全面诊断与整改方案_2026-03-16.md` 的核对与继承
2. 对 `project_notes/ByteTrack_SOMPT22_新旧Baseline全量详细对比_2026-03-16.md` 的逐序列复盘
3. 对当前真实代码实现、配置、相关单元测试的重新核查

本文档的目标只有一个:

- 回答 `P1-P3 已经完成之后，为什么系统仍未达到预期目标`
- 明确 `下一轮真正应该投入的改造方向是什么`
- 给出 `按什么顺序改、每一步具体改什么、预期解决什么问题`


## 1. 当前阶段的真实判断

### 1.1 已经确认的事实

当前系统相比旧 fresh baseline 已经有明确进步:

- `HOTA: 53.023 -> 53.549`
- `MOTA: 65.710 -> 66.013`
- `IDF1: 66.179 -> 67.014`
- `IDSW: 815 -> 747`
- `Association FPS: 18.9 -> 28.3`

这说明:

1. `P0-P3` 并不是无效改造
2. fixed-scene 生命周期特化方向没有走偏
3. recent-lost 恢复前移、birth 路由收敛、prior 降权都已经带来了真实收益

但是，这组结果同时也说明了另外一件更重要的事:

- 系统已经进入“方向验证通过，但结构上限暴露出来”的阶段

也就是说，后续已经不是“再补一两个 bug、再调几组阈值”就能自然达标的阶段，而是必须重新处理控制流和决策位置的阶段。


### 1.2 现在离目标还差在哪里

目标仍然是尽量逼近并稳定达到:

- `HOTA >= 54`
- `MOTA >= 67`
- `IDF1 >= 67`
- `IDSW < 500`
- `Association FPS > 100`

当前离目标的距离分成两块:

#### 身份连续性线

虽然 `IDF1` 已过 67，但:

- `IDSW = 747` 仍然明显偏高
- hardest sequence 里的 `Frag` 依然很重
- `08/10` 两个序列仍是系统级身份问题核心来源

#### 召回与误检线

虽然总 `FN` 已下降，但:

- `MOTA = 66.013` 仍明显低于 67
- 系统仍然存在一类“把更多框接回来了，但方式不够干净”的现象
- 某些序列的 `FP` 和 `Frag` 甚至在上涨

因此，后续不能只盯 `IDSW`，也不能只盯 `FN`。

必须同时处理:

1. `不该断的轨迹不要先断`
2. `已经断掉的轨迹要在更早阶段、更合理地接回`
3. `新生和旧 ID 回归必须在同一套逻辑里统一决策`


## 2. 对 P1-P3 成效的真实评价

### 2.1 P1 的价值

P1 的核心价值在于:

- recent-lost 恢复从 zombie 阶段前移
- 中短时遮挡有了 appearance-assisted 的恢复通道
- 未直接破坏原始 ByteTrack 主干

这一步是对的，而且是必要的。

如果没有 P1，当前的 `IDF1 >= 67` 基本不成立。

但 P1 的问题在于:

- 恢复虽然前移了，但仍然放在“主干纯 IoU 已经做完之后”
- 因此它更多是在补救主干之后的剩余错误
- 而不是直接改变“主干在 ambiguous case 如何作决策”

这决定了它的收益一定有限。


### 2.2 P2 的价值

P2 的核心价值在于:

- unmatched-high 的路由变得显式
- birth / recover / block 的控制关系比之前更清楚
- `center unmatched fallback` 不再像早期版本那样混乱

这一步的收益体现在:

- `IDs` 在多数序列里下降
- 一部分重复新生被抑制
- birth 行为相对更可解释

但 P2 的问题是:

- 当前仍然是“硬路由”
- 不是“统一候选空间中的联合打分”

也就是说，系统现在是:

1. 先问是不是 entry
2. 再问能不能 birth
3. 再问是不是 recover
4. 最后做 fallback

而不是:

- 同时比较“新生解释”和“旧 ID 回归解释”哪个更可信

这是当前控制流上限的关键来源之一。


### 2.3 P3 的价值

P3 的核心价值在于:

- spatial prior 不再继续无限放权
- 对 birth source 做了更保守的过滤
- 对 recent recovery 后的 support 回写做了 cooldown

这一步非常重要，因为它切断了最危险的错误自强化闭环。

但 P3 也只完成了第一阶段:

- prior 已经从“强裁判”退回“偏置项”
- 但它还没有成为一个真正稳定、可退化、可审计的场景先验系统

换句话说:

- P3 解决的是“先别让 prior 继续伤系统”
- 还没有解决“如何让 prior 真正成为可靠长期资产”


## 3. 从结果看，系统现在到底卡在什么地方

### 3.1 不能只看 headline

当前 headline 指标整体在涨，容易造成一种错觉:

- 好像系统已经进入了“继续微调参数就能自然达标”的阶段

但分序列和分项指标并不支持这个判断。

真正更有信息量的是这些现象:

- `SOMPT22-08: IDSW 245 -> 228`，但 `Frag 610 -> 633`
- `SOMPT22-10: IDSW 187 -> 188`，但 `FN -1249` 同时 `FP +848`、`Frag +34`
- `SOMPT22-07: IDF1 +0.0175`，但 `FP +1193`、`Frag +23`

这些组合型现象说明:

- 系统确实找回了更多目标
- 但并没有把“身份连续地找回”这件事真正做好
- 很多收益来自“追回来”而不是“持续跟住”


### 3.2 最危险的信号不是 IDSW 没降够，而是 Frag 还在涨

`Frag` 的上涨意味着:

- 轨迹更容易被切碎
- 切碎后即使后面重新接回，也已经付出了身份连续性代价

当前 hardest sequence 的问题并不是简单的“完全找不回”，而更像:

1. 轨迹先被切断
2. 后续又通过 lost 或 zombie 路径被找回
3. headline 指标有部分改善
4. 但身份时序质量并不干净

这和当前系统结构完全一致:

- 前面仍偏纯 IoU 主导
- 后面叠加 recent-lost / zombie rescue 做补救

因此，`Frag` 在这个阶段比单独看 `IDSW` 更能揭示系统上限。


### 3.3 当前失败模式可以继续分成两大类

#### 类型 A: 身份连续性问题主导

代表序列:

- `SOMPT22-08`
- `SOMPT22-10`

特点:

- `IDSW` 很高
- `Frag` 很高
- `IDs` 偏高
- 目标经常经历交叉、局部遮挡、近邻漂移

这类问题的核心不是长时消失后完全找不回，而是:

- 中短时阶段没有在正确时机保住旧 ID


#### 类型 B: 召回/生命周期问题主导

代表序列:

- `SOMPT22-07`
- `SOMPT22-13`

特点:

- `MOTA` 偏低
- `FN` 占比高
- `IDSW` 不是唯一主导项

这类问题更像:

- 该新生时不够及时
- 该保活时不够稳
- 低分回归没有利用好
- 生命周期控制仍偏硬


## 4. 当前代码层已经实现了什么

这一部分只记录已经被代码证实的事实。

### 4.1 状态机 P0 修复已落地

以下行为已在代码中落实，并且有单元测试保护:

1. `exit zone` 已改为基于当前 KF 位置读取，而不是旧 `tlwh`
2. `re_activate()` / `update()` 会清理:
   - `lost_frame_id`
   - `frozen_mean`
   - `exit_pending`
3. frozen 状态按每次 lost 周期独立重建

相关实现位于:

- `boxmot/trackers/bytetrack/bytetrack.py`

相关测试位于:

- `tests/unit/test_trackers.py`


### 4.2 recent-lost 恢复前移已落地

当前代码中，Step4 会先对 unmatched center-zone high detections 尝试 recent-lost recovery:

- `_match_recent_lost_tracks()`

然后才进入 zombie rescue:

- `_match_zombie_tracks()`

这是 P1 的核心落地点。


### 4.3 unmatched-high 路由显式化已落地

当前 unmatched high detection 会先经过:

- `_classify_unmatched_high_detection()`

并显式落到:

- `birth`
- `recover_then_block`
- `recover_then_birth`

这比早期版本的局部 fallback 结构清楚得多。


### 4.4 spatial prior 已降级为偏置项

当前 prior 只从保守 birth source 学习:

- `geometric_entry`
- `outside_expand`

同时 recent recovery 后的轨迹不会立刻给 prior 回写 support。

这说明 P3 并非停留在文档层，而是已经进入真实实现。


## 5. 但当前实现仍然存在的核心结构缺陷

下面这些不是“还能进一步优化”的泛泛建议，而是我认为当前系统达不到预期区间的主要根因。

### 5.1 缺陷一: recent-lost 恢复虽然前移了，但仍然插得太晚

这是当前最重要的结构问题。

当前控制流是:

1. `tracked + lost` 先一起进入 Step2
2. Step2 对它们统一做纯 IoU + score 匹配
3. 真正剩下的 unmatched high 才进入 recent-lost ReID 恢复

这意味着什么:

- 很多本来应该依靠 appearance 才能保住的 case
- 在 Step2 里已经被纯 IoU 的错误匹配或错误未匹配决定了命运
- 到 Step4 时，appearance 已经只能补救剩余 case

换句话说:

- recent-lost 并没有真正进入“核心分歧决策点”
- 它只是变成了“主干之后的一轮补救”

这正是为什么 `IDSW` 虽降，但 hardest sequence 的 `Frag` 还在涨。


### 5.2 缺陷二: 当前存在一个明显的中等时长失联真空带

当前关键参数组合是:

- `lost_reid_max_frames = 15`
- `zombie_transition_frames = 30`
- `zombie_max_predict_frames = 5`

而 `get_tlwh_for_matching()` 的 frozen 逻辑同时被 recent-lost 和 zombie 复用。

这带来两个问题:

#### 问题 A

对 recent-lost 而言，丢失超过 `5` 帧后，就可能开始基于 frozen 位置匹配。

这对 `6-15` 帧的中短时遮挡并不理想，因为:

- 这类 case 仍处于可恢复范围
- 但冻结位置会迅速降低 motion gate 的真实性

#### 问题 B

对 `16-29` 帧这段区间而言:

- recent-lost 已经失效
- zombie 还没接管
- 系统只剩前面那轮纯 IoU 主干在兜底

这就是一个明确的生命周期真空带。

这一段正好对应固定监控场景中非常常见的一类 case:

- 行人被车/电动车/柱体/人群遮挡十几帧
- 重现时位置已有一定位移
- 但外观还完全值得用

当前系统对这类 case 的处理是不完整的。


### 5.3 缺陷三: birth / recover 仍是硬路由，不是统一打分

当前 unmatched high 的核心逻辑本质上仍然是:

1. 如果是 entry，就直接走 birth
2. 如果在中心，就先 recover
3. recover 失败后，再看 block 或 birth

这类结构的缺点是:

- 它强依赖先验判断是否正确
- 不能直接比较“这是新生”与“这是旧 ID 回归”哪种解释更可信

而在复杂固定监控场景里，很多 case 本来就不是硬二选一:

- 边缘附近再现，可能是真新生，也可能是旧轨回归
- 中心区高分框，可能是真旧 ID 回归，也可能是漏进场后的迟到新生

如果不在同一候选空间里比较两种解释的相对可信度，系统就会长期卡在:

- 某些 case 过早 birth
- 某些 case 过度保守
- 最终不断依赖 fallback


### 5.4 缺陷四: 低分检测路径没有承担 lost 回归职责

当前 Step3 只允许剩余 `Tracked` 去跟 low-score detections 匹配。

这意味着:

- `Lost` 轨迹无法借助低分检测完成回归

这会直接损伤两类场景:

1. 遮挡刚结束、目标初次露头时置信度偏低
2. 光照、边缘裁切、尺度变化导致短时 detector 分数不稳

在固定监控场景里，这类 case 非常常见。

因此当前系统会出现一种低效模式:

1. 低分回归没接
2. 下一帧高分框重新出现
3. 再通过 recent-lost 或 birth 路径处理
4. 指标上表现为 `FN`、`Frag`、局部 `FP` 同时被放大


### 5.5 缺陷五: spatial prior 现在是“低风险偏置器”，还不是“可靠场景先验”

当前 prior 最大的问题已经不再是“权力过大”，而是:

- 它的成熟与退化机制还不够完整

现在的 `spatial_prior_stage` 更像:

- 达标前 `learn_only`
- 达标后 `entry_only`

但这个状态转移仍然偏单向。

同时:

- maturity 基于累计样本计数
- map 本身会 decay
- 但 stage 的形成不随有效样本衰减自动回退

这意味着:

- prior 一旦成熟，就更像“默认长期有效”
- 而不是“始终根据近期可靠度动态保持可信”

因此，当前 prior 最合理的定位仍然是:

- 风险修正项
- birth 偏置项
- 辅助 entry region 建模器

还不应该进一步提升为主决策源。


### 5.6 缺陷六: 当前系统改善了“找回来”，但没有改善“先别断”

这是对当前所有结果最浓缩的总结。

很多序列的典型形态是:

- `IDSW` 有所下降
- `IDs` 有所下降
- 但 `Frag` 上升

说明系统在做的事情更接近:

- 让更多被切碎的轨迹有机会重新被找回来

而不是:

- 在关键分歧点尽量不把轨迹切碎

这就是为什么当前再继续围绕 zombie rescue 微调，收益空间会越来越小。


## 6. 对当前各模块的重新定位

### 6.1 ByteTrack 主干

应该继续保留。

但“保留主干”的准确含义是:

- 保留 `Tracked` 主通道的两段式 IoU 稳定性
- 不等于让 `Lost` 永远完全附着在这条纯 IoU 主通道上

真正应该避免的是:

- 让 appearance 粗暴接管全局主匹配

而不是:

- 在高风险歧义阶段给 `Lost` 单独的 appearance-assisted 恢复层


### 6.2 recent-lost recovery

方向对，但当前版本还只是第一版。

下一步不该做的是:

- 单纯继续调 `lost_reid_thresh`
- 单纯继续调 `lost_match_cost_thresh`

下一步真正该做的是:

- 重构它在整体控制流中的位置
- 让它进入核心分歧点，而不是只接剩余 case


### 6.3 zombie rescue

应继续保留，但从现在开始不应再作为主研发杠杆。

原因很简单:

- hardest case 的大部分损失发生在 zombie 之前
- zombie 本身已经是后段补救
- 后段补救再增强，也很难替代“前段少切碎”


### 6.4 birth control

方向正确，但需要从“规则堆叠”走向“统一决策”。

当前 birth control 的价值依然存在:

- 降低重复新生
- 避免中心区域无脑 birth
- 利用 fixed-scene 入口约束

但必须从现在这种:

- entry gate
- strict gate
- pending confirm
- duplicate suppress
- center fallback

的串联结构，改造成统一评分/统一比较结构。


### 6.5 spatial prior

短期内应继续保持弱裁判定位。

它最适合做的事:

- 提供 entry hotspot 偏置
- 辅助判断某个 unmatched high 更像“合理新生”还是“异常中心 birth”
- 为 future offline prior 铺路

它现在还不适合做的事:

- 直接决定中心区绝不 birth
- 直接决定某类检测一定是新生
- 进一步扩大主控制权


## 7. 下一阶段最应该做什么

这里给出我认为真正合理的改造主线。

### 7.1 第一主线: 把恢复真正前移到主干决策点

这是最重要的一条。

目标不是“让 ReID 进入全局主干”，而是:

- 让 `Lost` 不再完全依附于 `Tracked` 的纯 IoU 匹配命运

更具体地说，下一版控制流应该改成:

1. `Tracked` 高分匹配仍保留 ByteTrack 主干
2. `Lost` 不和 `Tracked` 混在同一纯 IoU 池里统一决策
3. 对 `unmatched high` 和 `recent/mid lost` 单独做受限恢复匹配
4. 只在歧义 case 上启用 appearance-assisted 决策

这样做的收益是:

- 不破坏主干稳定性
- 又能把 appearance 放到真正该放的位置


### 7.2 第二主线: 明确拆出三段式记忆，而不是 recent-lost / zombie 二段式

我建议后续明确引入三层生命周期:

#### short-lost

- 丢失 `1-6` 帧
- 以运动预测为主
- appearance 只做 tie-break

#### mid-lost

- 丢失 `7-20` 帧左右
- 允许更高权重 appearance
- 不应过早 frozen
- 是固定监控场景最有价值的一段恢复区间

#### long-lost / zombie

- 丢失 `20` 或 `30` 帧以上
- 进入更保守、更低频的 zombie rescue
- 允许 frozen position

当前最需要拆出来的不是更多模块，而是:

- `lost_freeze_frames`
- `mid_lost_max_frames`
- `mid_lost_reid_weight` 类参数语义

避免继续让:

- lost 的冻结策略
- zombie 的冻结策略

混用同一组参数。


### 7.3 第三主线: 新生与旧 ID 回归必须在同一决策面上比较

这一条是 P2 的真正下一阶段。

建议后续对每个 unmatched high 构建三类候选解释:

1. `best_recover_candidate`
2. `best_birth_candidate`
3. `defer / keep_pending`

然后比较它们的相对分数，而不是靠硬路由。

其中:

- `recover_score` 由 motion、shape、appearance、time_since_lost、entry consistency 共同组成
- `birth_score` 由 entry geometry、effective zone、prior hotspot、duplicate suppression risk 共同组成
- `defer_score` 由 detection stability、短时确认需求决定

这样回答的才是系统真正需要回答的问题:

- 这个未匹配高分框，究竟更像谁

而不是:

- 它先落在哪条规则支路上


### 7.4 第四主线: 给低分回归补一条窄而稳的恢复分支

当前不建议把 low-score detections 全面开放给复杂 appearance 匹配。

建议的是:

- 只对 `recent/mid lost`
- 只对距离和形状都合理的候选
- 只做有限恢复

这一分支的目标不是大规模追召回，而是:

- 避免“旧 ID 初次露头时因为检测分数不高被彻底漏掉”

这会直接帮助:

- `SOMPT22-07`
- `SOMPT22-13`

这类 recall 主导的序列。


### 7.5 第五主线: 把 spatial prior 从在线偏置器升级为“可退化、可冻结、可离线化”的先验系统

下一阶段 prior 真正值得做的方向不是继续提权，而是补齐工程可信度:

1. stage 不再只升不降
2. maturity 计数改成 decayed effective counts
3. 支持将 prior 从训练/预跑阶段离线导出
4. eval 阶段支持只读加载，不再完全依赖在线学习

固定监控场景天然适合离线先验。

如果每次都让 tracker 一边跑一边学，再拿学到的结果反过来约束自己，长期上限一定会受限。


## 8. 推荐的详细整改步骤

下面给出更具体、可执行的下一轮实施顺序。

### R1: 拆分 `Tracked` 与 `Lost` 的主匹配职责

目标:

- 让 `Tracked` 继续走稳定 ByteTrack 主干
- 让 `Lost` 从“附着在纯 IoU 主干上的被动角色”变成“独立受控恢复对象”

建议做法:

1. Step2 先只用 `Tracked` 匹配高分框
2. 形成 `unmatched_high`
3. 对 `recent/mid lost` 单独做一轮恢复匹配
4. 必要时再把剩余一小部分 ambiguous case 送入 appearance tie-break

预期解决:

- `08/10` 里大量本该保住但在 Step2 被定型的 IDSW / Frag


### R2: 引入 mid-lost 层，并拆开 freeze 语义

目标:

- 消除当前 `6-29` 帧附近的生命周期真空带

建议新增或重构参数:

- `short_lost_max_frames`
- `mid_lost_max_frames`
- `lost_freeze_frames`
- `mid_lost_match_max_dist`
- `mid_lost_reid_weight`
- `mid_lost_match_cost_thresh`

预期解决:

- 交叉遮挡后十几帧再现的目标接不回来
- frozen 过早导致位置门控偏死的问题


### R3: 重构 unmatched-high 的统一打分

目标:

- 用统一决策替代现有硬路由

建议做法:

1. 为每个 unmatched high 计算:
   - best recover score
   - best birth score
   - pending/defer score
2. 加入 margin 规则:
   - 若 recover 明显优于 birth，则认旧
   - 若 birth 明显优于 recover，则新生
   - 若差距不足，则进入 pending 或延后确认

特别建议:

- 不再让几何 entry 默认 `skip_confirmation=True`
- 只有“完全在有效区外、且与任何现存/记忆轨都不近”的 case 才允许 fast birth

预期解决:

- 边缘反复出生
- 中心回归和中心新生边界不清
- `strict_entry_gate=false` 带来的模糊 fallback


### R4: 给 low-score 加一条窄恢复支路

目标:

- 降低 recall 主导序列里的无谓 FN

建议做法:

1. 在 Step3 后增加一轮:
   - low-score unmatched detections
   - recent/mid lost tracks
2. 使用更严格的 motion/shape 门控
3. appearance 只作为辅助，不允许大范围误召

预期解决:

- 初次露头低分框被完全错过
- 旧轨迹重现晚一帧才能接上的问题


### R5: spatial prior 工程化

目标:

- 让 prior 从“尽量别害人”变成“逐渐成为稳定资产”

建议做法:

1. `spatial_prior_stage` 引入可退化逻辑
2. stage 判断改成有效样本量而不是简单累计样本量
3. 支持 prior dump / load
4. 训练阶段或 warmup 阶段单独构建 prior
5. 正式 eval 阶段优先只读使用

预期解决:

- online self-reinforcement 的长期隐患
- 不同运行中 prior 行为不稳定的问题


## 9. 不建议走的方向

### 9.1 不建议直接把 ReID 全面塞进 Step1/Step2 主干

原因:

- 回归风险太大
- 容易破坏 ByteTrack 原本最稳的部分
- 当前系统的真正问题不是“appearance 不够多”，而是“appearance 出场太晚”


### 9.2 不建议继续把主要资源押在 zombie rescue 调参上

原因:

- zombie 已经是后段补救
- hardest case 的主要损失发生在 zombie 之前
- 后段补救增强无法替代前段少断轨


### 9.3 不建议继续堆更多局部 gate 规则

原因:

- 当前系统已经不是缺规则，而是规则关系不对
- 继续加 if/else 只会让控制流更难解释、更难稳定


### 9.4 不建议在 prior 还未完全可信前继续扩大其决策权

原因:

- prior 仍然来源于 tracker 自身行为
- tracker 仍有 birth / recovery 错误
- 过早放权会重新引入自强化闭环


## 10. 建议的实验组织方式

后续实验必须明确拆成两条线，不能再把所有收益混在一起解释。

### 10.1 实验线 A: 身份连续性线

重点序列:

- `SOMPT22-08`
- `SOMPT22-10`

重点指标:

- `IDSW`
- `Frag`
- `IDs`
- `IDF1`
- `HOTA`

重点改法:

- `Tracked/Lost` 主干拆分
- `mid-lost` 层
- unified recover vs birth scoring


### 10.2 实验线 B: 召回与误检线

重点序列:

- `SOMPT22-07`
- `SOMPT22-13`

重点指标:

- `CLR_FN`
- `CLR_FP`
- `MOTA`
- `MT`
- `ML`

重点改法:

- low-score recovery 支路
- birth 统一打分
- lifecycle softening


### 10.3 评估顺序建议

建议每轮只改一个主要结构因素，然后按以下顺序验证:

1. 单元测试
2. 重点序列子集 eval
3. 全序列 eval
4. 对比分序列详细指标

否则很容易出现:

- 全局 headline 小涨
- 但看不清楚到底是哪一类错误在改善，哪一类在恶化


## 11. 最值得优先落地的一版具体方案

如果下一步只做一件事，我建议优先做:

- `Tracked` / `Lost` 主干拆分 + mid-lost 独立恢复带

原因:

1. 这是最贴近当前真实瓶颈的一刀
2. 它不要求大规模推翻现有主干
3. 它最有机会继续打掉 `08/10` 的 IDSW 和 Frag
4. 它也为后续 birth 统一打分提供更干净的输入

一个推荐的第一版控制流原型如下:

1. `Tracked` 高分匹配:
   - 保留原始 ByteTrack 风格 IoU 主干
2. `unmatched high` vs `recent/mid lost`:
   - motion + shape + appearance 受限匹配
3. `remaining unmatched high`:
   - 统一比较 recover_score 与 birth_score
4. `long-lost / zombie`:
   - 保留当前 zombie rescue 作为最后补救
5. `low-score`:
   - 仅为 recent/mid lost 提供窄恢复支路

这个结构相较当前版本的根本改进点在于:

- 不再把“旧 ID 是否应该被认回”这件事主要留到主干之后处理


## 12. 最终结论

最终只保留最核心的判断。

### 12.1 当前系统没有走错方向

fixed-scene 生命周期特化、recent-lost 恢复前移、zombie rescue、spatial prior 这些方向本身都成立。


### 12.2 当前系统的主要上限不是功能不够，而是功能插入位置不对

现在的问题不再是:

- 还缺一个模块

而是:

- 模块已经不少
- 但关键分歧点上的决策权分配仍然不合理


### 12.3 当前最关键的矛盾是“先切碎，再补救”

系统现在更擅长:

- 在后段把一部分切碎的轨迹再找回来

而不够擅长:

- 在前段就尽量不要把轨迹切碎

这就是当前 hardest sequence 里 `Frag` 居高不下甚至继续上升的根因。


### 12.4 下一轮真正该做的不是继续微调 zombie，而是重构恢复时机和统一决策

最值得推进的顺序应当是:

1. 拆分 `Tracked` / `Lost` 主干职责
2. 引入 `mid-lost` 恢复层并拆开 freeze 语义
3. 把 birth / recover 做成统一打分逻辑
4. 给 low-score 增加窄恢复支路
5. 再做 prior 工程化和后续速度优化

如果不先做前 3 步，而继续围绕:

- zombie rescue 微调
- 更多局部 gate
- prior 放权

则系统很可能继续停留在:

- headline 指标缓慢上升
- 但 hardest sequence 一直压不干净


## 13. 本文档的用途

后续如果继续推进这条线，建议把本文作为:

- `P3` 完成后的阶段总结文档
- 下一轮结构改造的对齐文档
- 后续代码改造和实验设计的依据

尤其在进入下一轮实现前，应优先围绕本文档确认两件事:

1. 是否接受“当前上限的核心问题在恢复时机和统一决策，而不是 zombie 强度不足”
2. 是否接受“下一轮要先动控制流结构，再做参数调优”

如果这两点不先对齐，后续实验很容易再次回到:

- 阈值反复微调
- headline 小幅波动
- 但 hardest sequence 迟迟压不下来的循环里


## 14. 附: 本次核查中额外确认的测试

为了确保本文判断建立在当前真实代码状态上，而不是文档滞后状态上，本次额外核对并跑通了以下相关单元测试子集:

```bash
uv run pytest tests/unit/test_trackers.py -k 'zombie_reid_global_assignment_prefers_appearance or zombie_reid_gate_blocks_wrong_appearance_rescue or recent_lost_reid_recovers_center_unmatched_high_before_zombie or recent_lost_reid_blocks_wrong_appearance or track_recovery_clears_lost_lifecycle_state or frozen_state_is_rebuilt_per_lost_cycle'
```

结果:

- `7 passed`

因此，本文的分析结论建立在:

- 当前代码
- 当前配置
- 当前测试
- 当前结果文件

四者一致的基础上，而不是建立在旧草稿或旧理解之上。
