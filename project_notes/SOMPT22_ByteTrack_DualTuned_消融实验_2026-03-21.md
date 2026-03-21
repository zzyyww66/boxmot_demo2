# SOMPT22 上 ByteTrack-DualTuned 消融实验（2026-03-21）

## 1. 目的

这次消融实验的目标，不是再去比较不同 tracker，而是回答下面两个更具体的问题：

1. `boxmot/configs/trackers/bytetrack_dual_tuned.yaml` 这套当前跨 SOMPT22 / MOT20 的统一最优参数，在 **SOMPT22** 上到底是靠哪些结构模块在贡献收益？
2. 如果后续希望做“保精度裁剪”或“保速度裁剪”，应该优先保留哪些模块，优先裁掉哪些模块？

---

## 2. 阅读与实现依据

这次实验前，我先对齐了仓库里的设计文档和代码实现，重点参考了下面这些内容：

- 设计总文档：
  - `project_notes/SPL-ByteTrack_最终版算法设计与实现_2026-03-20.md`
  - `project_notes/SPL-ByteTrack_答辩讲解版_2026-03-20.md`
- 当前 dual tuned 参数说明：
  - `project_notes/ByteTrack_双数据集联合最优参数与SOMPT22_MOT20对比_2026-03-20.md`
- 关键实现位置：
  - `boxmot/trackers/bytetrack/bytetrack.py`
  - `boxmot/engine/evaluator.py`
  - `boxmot/trackers/tracker_zoo.py`

从文档和代码对齐后，可以把当前 SPL-ByteTrack 的核心增强概括为：

- `birth control`
- `recent-lost recovery`
- `zombie rescue`
- `spatial prior`
- `exit-zone lifecycle`

同时，代码层面也验证了一个重要事实：

- `ByteTrack` 主干 Step1 / Step2 并没有被推翻；
- 主要增量都发生在 unmatched-high 路由、生命周期拆分、短时恢复、长时恢复和场景先验这几层。

这决定了这次消融应该围绕“结构模块关掉后对 SOMPT22 的影响”来做，而不是去做大量无解释力的单参数扫表。

---

## 3. 实验设置

### 3.1 固定项

- 数据集：`/root/autodl-tmp/boxmot_demo2/train`
- 序列数：9
- 总帧数：14100
- Detector：`yolov8m_pretrain_crowdhuman`
- ReID：`osnet_x0_25_msmt17`
- 类别过滤：`--classes 0`，即仅 `person`
- 基线配置：`boxmot/configs/trackers/bytetrack_dual_tuned.yaml`

### 3.2 缓存复用

本次实验严格复用了已经存在的 SOMPT22 person-only dets/embs 缓存：

- 缓存根目录：
  - `/root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17`
- 实际 dets / embs：
  - `/root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17/dets_n_embs/yolov8m_pretrain_crowdhuman/dets`
  - `/root/autodl-tmp/boxmot_demo2/runs_sompt22_person_cache_yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17/dets_n_embs/yolov8m_pretrain_crowdhuman/embs/osnet_x0_25_msmt17`

### 3.3 实际执行方式

这次没有直接走整条 `boxmot.engine.cli eval` 命令链，而是走了更贴近“纯算法消融”的执行路径：

1. 读取 `bytetrack_dual_tuned.yaml` 的 `default` 作为基础参数字典。
2. 针对每个消融实验，在这个基础字典上做少量 override。
3. 通过 `boxmot.engine.evaluator.process_sequence()` 直接消费已有 dets/embs 缓存，逐序列生成 MOT result。
4. 再通过 `boxmot.engine.evaluator.run_trackeval()` 做 TrackEval 评测。

这样做的原因是：

- CLI 即使缓存命中，仍会初始化 detector / ReID；
- 但本次目标是看 **tracker 关联与生命周期模块** 的消融，不是重复跑检测器；
- `process_sequence()` 和 `run_trackeval()` 走的仍然是仓库现有标准实现，因此 **指标口径不变**。

---

## 4. 消融设计

### 4.1 为什么这样拆

本次没有做“所有开关全排列”，而是做了 6 组 **最小但足够有解释力** 的实验。

设计原则是：

- 单模块消融优先对应设计文档里真正强调的结构模块；
- 避免做过度耦合的开关，导致一个实验同时破坏太多决策路径；
- 最后再补一组 `Mainline Only`，看整套生命周期增强相对主干究竟带来了多少收益。

### 4.2 6 组实验

| 实验名 | 改动 | 设计意图 |
| --- | --- | --- |
| `Full DualTuned` | 无 | 当前完整 dual tuned 系统，作为基准 |
| `No Recent-Lost` | `lost_reid_enabled=false` | 去掉 Step4 中的 recent-lost recovery，看短时恢复的边际收益 |
| `No Zombie` | `zombie_max_history=0`，`zombie_reid_enabled=false` | 去掉 zombie 生命周期与长期复活，看 long-gap memory 的贡献 |
| `No Spatial Prior` | `spatial_prior_enabled=false`，`spatial_prior_region_enabled=false` | 去掉被动场景先验，观察它是否真在当前配置中带来可见收益 |
| `No Birth QC` | `birth_confirm_frames=1`，`birth_suppress_iou=0`，`birth_suppress_center_dist=0` | 保留 recovery ordering，但拿掉“新生确认 + 近邻抑制”这层质量控制 |
| `Mainline Only` | 关闭 `recent-lost`、`zombie`、`spatial prior`、`adaptive_zone`、`exit_zone`，并把 `entry_margin=0`、`birth_confirm_frames=1`、`birth_suppress_*=0` | 尽量退化回不依赖 scene-aware birth / prior / recovery 的主干 ByteTrack 形态 |

### 4.3 一个有意识的取舍

本次没有单独做 `No Exit-Zone`。

原因不是忘了，而是刻意没有单拆：

- `exit_zone` 主要作用在边缘离场生命周期管理；
- 它和 `birth / recovery / zombie` 的耦合比 `recent-lost`、`zombie`、`birth QC` 更强；
- 单独拆它的信息量，低于先把 `zombie`、`birth QC`、`recent-lost` 这几个主贡献项拆清楚。

因此，这一轮里：

- 单模块实验保持 `exit_zone` 打开；
- 只在 `Mainline Only` 里和其他生命周期增强一起关闭。

如果后续想把边缘离场策略单独展开，可以再补一组 `No Exit-Zone`。

---

## 5. 运行产物

- 本轮实验根目录：
  - `/root/autodl-tmp/boxmot_demo2/runs_sompt22_dual_ablation_20260321_033545`
- 汇总 JSON：
  - `/root/autodl-tmp/boxmot_demo2/runs_sompt22_dual_ablation_20260321_033545/summary.json`

每组实验都各自落在：

- `.../full_dual/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack`
- `.../no_recent_lost/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack`
- `.../no_zombie/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack`
- `.../no_spatial_prior/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack`
- `.../no_birth_qc/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack`
- `.../mainline_only/mot/yolov8m_pretrain_crowdhuman_osnet_x0_25_msmt17_bytetrack`

---

## 6. 总结果

### 6.1 主表

`Δ` 一列均相对 `Full DualTuned`。

说明：

- `Full DualTuned` 的 `HOTA / MOTA / IDF1 / IDSW` 与仓库里先前记录的 dual tuned SOMPT22 结果完全对齐，说明这次缓存复用评测口径是正确的。
- 本文的 `Assoc FPS` 来自本轮 `summary.json` 中累计 `track_time_ms / frames` 的重新计算，因此和历史文档中的速度值可能有小幅差异；这不影响本轮消融的相对结论。

| 实验 | HOTA | ΔHOTA | MOTA | ΔMOTA | IDF1 | ΔIDF1 | IDSW | ΔIDSW | IDs | ΔIDs | Assoc FPS | ΔFPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full DualTuned | 54.243 | 0 | 66.515 | 0 | 68.150 | 0 | 812 | 0 | 840 | 0 | 198.3 | 0 |
| No Recent-Lost | 54.023 | -0.220 | 66.543 | +0.028 | 67.707 | -0.443 | 842 | +30 | 862 | +22 | 199.4 | +1.0 |
| No Zombie | 51.991 | -2.252 | 65.133 | -1.382 | 63.104 | -5.046 | 1143 | +331 | 1578 | +738 | 206.2 | +7.8 |
| No Spatial Prior | 54.246 | +0.003 | 66.524 | +0.009 | 68.155 | +0.005 | 809 | -3 | 837 | -3 | 209.1 | +10.8 |
| No Birth QC | 53.951 | -0.292 | 65.930 | -0.585 | 67.143 | -1.007 | 922 | +110 | 1017 | +177 | 197.0 | -1.3 |
| Mainline Only | 51.594 | -2.649 | 65.640 | -0.875 | 61.501 | -6.649 | 1384 | +572 | 1942 | +1102 | 214.4 | +16.1 |

### 6.2 单模块重要性排序

如果只看单模块关闭后的退化幅度，不把 `Mainline Only` 这个“总退化项”算进去，则在 SOMPT22 上的影响排序很清楚：

1. `zombie rescue / zombie lifecycle`
   - `HOTA -2.252`
   - `IDF1 -5.046`
   - `IDSW +331`
   - `IDs +738`
2. `birth QC`
   - `HOTA -0.292`
   - `IDF1 -1.007`
   - `IDSW +110`
   - `IDs +177`
3. `recent-lost recovery`
   - `HOTA -0.220`
   - `IDF1 -0.443`
   - `IDSW +30`
   - `IDs +22`
4. `spatial prior`
   - 指标基本不变，可视为近似 0 影响

---

## 7. 结果解读

### 7.1 `zombie` 是这套改进里最关键的主收益项

`No Zombie` 的退化最明显：

- `HOTA 54.243 -> 51.991`
- `MOTA 66.515 -> 65.133`
- `IDF1 68.150 -> 63.104`
- `IDSW 812 -> 1143`
- `IDs 840 -> 1578`

这和设计文档里的主张是对齐的：

- `recent-lost` 解决的是短时断链；
- `zombie rescue` 解决的是超出 lost window 之后的 long-gap recovery；
- 在 SOMPT22 上，后者是更大的主收益来源。

换句话说，**如果要保住这套 SPL-ByteTrack 的核心身份恢复能力，`zombie` 不能关。**

### 7.2 `birth QC` 的价值也是真实存在的，而且不只是“锦上添花”

关闭 `birth_confirm_frames` 和 `birth_suppress_*` 之后：

- `HOTA -0.292`
- `MOTA -0.585`
- `IDF1 -1.007`
- `IDSW +110`
- `IDs +177`

这里很值得注意的一点是：

- 它不只是拉低了 `IDF1`；
- 连 `MOTA` 也一起掉了；
- 而且 `Assoc FPS` 还略微下降了 `1.3 FPS`。

这说明 `birth QC` 并不是纯粹“多做一道保守门”而已，而是确实减少了：

- 中心区误新生
- 重复出生
- 后续由错误新轨迹带来的额外关联负担

也就是说，**把 birth control 做成“可控新生”而不是“看见 unmatched high 就生新 ID”，在 SOMPT22 上是有效的。**

### 7.3 `recent-lost` 的收益相对温和，但方向很稳定

关闭 `recent-lost` 后：

- `HOTA -0.220`
- `IDF1 -0.443`
- `AssRe -0.511`
- `IDSW +30`

但 `MOTA` 反而有一个非常小的上升：

- `66.515 -> 66.543`

我对这个现象的解释是：

- `recent-lost` 主要提升的是短时身份连续性；
- 它更直接作用在 `IDF1 / AssRe / IDSW` 上；
- 对 `MOTA` 这种更偏检测匹配总量的指标，收益没有 `zombie` 或 `birth QC` 那么直接；
- 而且少做一次短时恢复，有时会轻微减少某些错误拉回，因此 `MOTA` 会出现非常小的波动。

所以这组结果更适合解读为：

> `recent-lost` 是“有用但不是主战场”的增强项，它主要负责补足 short-gap identity continuity。

### 7.4 `spatial prior` 在当前 SOMPT22 + dual tuned 设定下几乎是弱偏置

这是这次实验里最值得注意的一个“负结论”：

- `No Spatial Prior` 的主指标基本不变：
  - `HOTA +0.003`
  - `MOTA +0.009`
  - `IDF1 +0.005`
- 但 `Assoc FPS` 提升了 `10.8 FPS`

这意味着在 **当前这套 dual tuned 参数、当前本地 SOMPT22 split、当前实现版本** 下：

- `spatial prior` 不是主收益来源；
- 它更像一个保守的弱偏置项；
- 从精度角度看，几乎可以视为“开和不开都差不多”。

这里我做一个明确说明：

- 这是 **本轮实验结果**；
- 不是说 `spatial prior` 在所有固定机位场景都没用；
- 更合理的结论是：

> 在当前 SOMPT22 + dual tuned 设置下，`spatial prior` 的收益远小于 `zombie`、`birth QC` 和 `recent-lost`，更适合作为可选模块，而不是必须保留的主收益模块。

### 7.5 `Mainline Only` 给出了整套生命周期增强的总贡献下界

把 `recent-lost`、`zombie`、`spatial prior`、`adaptive zone`、`exit_zone` 和 birth QC 一起拿掉之后：

- `HOTA 54.243 -> 51.594`
- `MOTA 66.515 -> 65.640`
- `IDF1 68.150 -> 61.501`
- `IDSW 812 -> 1384`
- `IDs 840 -> 1942`
- `Assoc FPS 198.3 -> 214.4`

这说明完整 dual tuned 相比 stripped mainline 的综合收益大约是：

- `HOTA +2.649`
- `MOTA +0.875`
- `IDF1 +6.649`
- `IDSW -572`
- `IDs -1102`

代价则是：

- `Association FPS` 下降约 `16.1 FPS`

这组结果非常适合用来回答“这套改进值不值”这个问题。

我的判断是：**值。**

因为它换来的不是一点点局部修饰，而是：

- 更低的 IDSW
- 显著更少的 ID 数量膨胀
- 明显更好的 IDF1

对固定监控类 MOT 任务来说，这个 trade-off 是成立的。

---

## 8. 序列层面的补充观察

从 `summary.json` 的 per-sequence 指标看，下面两个序列对“去掉恢复链”最敏感：

- `SOMPT22-02`
- `SOMPT22-07`

具体表现为：

- `No Zombie` 下，`SOMPT22-02` 的 `HOTA` 下降约 `8.0`，`IDF1` 下降约 `14.7`
- `Mainline Only` 下，`SOMPT22-02` 的 `HOTA` 下降约 `7.4`，`IDF1` 下降约 `15.2`
- `No Zombie` / `Mainline Only` 下，`SOMPT22-07` 也都是退化最明显的序列之一

这说明一个很实际的现象：

- SOMPT22 并不是所有序列都一样依赖 long-gap recovery；
- 但至少有一部分序列，确实高度依赖 `zombie` 这条长时身份恢复链。

这是一个**基于结果的推断**：

> 我推测这些序列里存在更多“长遮挡后再出现”或“中部区域断链后重新被检测到”的 case，因此对 `zombie rescue` 和整体 lifecycle layering 更敏感。

---

## 9. 最终结论

如果只给一句结论，我会这样总结：

> 在 SOMPT22 上，`bytetrack_dual_tuned.yaml` 的主要收益不是来自 `spatial prior`，而是来自“`zombie` 长时恢复 + `birth QC` 新生控制 + `recent-lost` 短时恢复”这条完整的生命周期增强链。

更具体一点：

1. **必须保留的核心模块**
   - `zombie rescue / zombie lifecycle`
   - `birth QC`
   - `recent-lost recovery`

2. **当前设定下可以优先裁剪的模块**
   - `spatial prior`

3. **如果要做速度优先的简化版**
   - 第一优先级：先尝试关闭 `spatial prior`
   - 不建议优先关闭：`zombie`
   - 也不建议直接把 birth QC 全关

4. **如果只想保留 ByteTrack 主干**
   - 可以得到更高一点的关联速度
   - 但会付出很明显的身份一致性代价，尤其是 `IDF1` 和 `IDSW`

---

## 10. 复现建议

如果后续要继续扩展这组消融，我建议按下面顺序补实验：

1. `No Exit-Zone`
   - 单独验证边缘离场管理的贡献
2. `No Adaptive-Zone`
   - 和 `No Spatial Prior` 区分开，看几何 scene mask 自身的作用
3. `No Birth Routing, Keep Recovery`
   - 做更细粒度的 unmatched-high 路由层实验
4. `SOMPT22-02 / SOMPT22-07` 的 case study
   - 直接抽出最敏感序列做可视化比对

如果只是想快速复现本轮结果，建议直接沿用本轮做法：

- 仍然使用 shared dets/embs cache
- 仍然以 `bytetrack_dual_tuned.yaml` 为 base config
- 仍然通过 `process_sequence()` + `run_trackeval()` 做结构开关消融

因为这种方式：

- 指标口径和仓库当前 evaluator 保持一致
- 不会重复浪费 detector / ReID 计算
- 更适合做纯 tracker 算法消融
