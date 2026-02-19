# ByteTrack IDSW 测试全流程复盘（2026-02-19）

## 1. 文档目的

这份文档汇总“到目前为止”的关键测试过程，重点回答：

1. 开启 `strict_entry_gate` 时，`IDSW=20` 的现象如何理解；
2. 关闭 `strict_entry_gate` 后为什么 `IDSW` 一度显著上升；
3. 后续如何在关闭 `strict_entry_gate` 的同时把 `IDSW` 压回去；
4. MOT20 外部消融结果说明了什么，后续怎么改。

---

## 2. 数据来源

本复盘来自以下记录与结果目录：

1. `project_notes/MOT17-04_ByteTrack对比测试指南.md`
2. `project_notes/MOT17-04_参数调优实验记录_2026-02-18.md`
3. `project_notes/MOT17-04_最终测试结果_2026-02-19.md`
4. `runs/ablation/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_*`（历史 run 指标）
5. 代码现状：`boxmot/trackers/bytetrack/bytetrack.py`

---

## 3. 关键时间线与测试结果

### 3.1 阶段 A：Baseline vs Improved（早期主对比）

口径：MOT17-04，固定检测输入，只做关联+评估。

- Baseline（改进能力关闭）：
  - `HOTA 77.861`
  - `MOTA 90.305`
  - `IDF1 88.638`
  - `IDSW 32`
  - `IDs 86`
- Improved（当时核心改进开启，`strict_entry_gate=True`）：
  - `HOTA 78.375`
  - `MOTA 90.855`
  - `IDF1 89.816`
  - `IDSW 20`
  - `IDs 76`

结论：

1. `strict_entry_gate + zombie + exit 延迟删除` 的组合，能明显降低切换与重生；
2. 当时 improved 相比 baseline 的 `IDSW` 降低 12（32 -> 20）。

---

### 3.2 阶段 B：围绕 `IDSW=20` 的参数扫参（开启 strict）

执行：`run_mot17_04_param_sweep.py`，共 37 组（A/B/复跑）。

结果：

1. 30 组候选中绝大多数指标完全一致；
2. `IDSW` 基本固定在 `20`，未继续下降；
3. 将 `exit_zone_remove_grace` 拉大到 36/42 会让 `MOTA` 下降，但 `IDSW` 不降。

结论：

1. 在“仅调改进模块参数”的局部空间里，已进入平台区；
2. 这就是“开启 strict 时检测到 20 个 IDSW 但怎么调都不太变”的直接原因。

---

### 3.3 阶段 C：关闭 `strict_entry_gate` 后的第一次测试（未加 soft birth gating）

对应结果目录：`runs/ablation/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_103`

- `HOTA 78.303`
- `MOTA 90.305`
- `IDF1 89.623`
- `IDSW 29`
- `IDs 81`

与“开启 strict 的参考 improved（IDSW=20, IDs=76）”相比：

1. `IDSW +9`（20 -> 29）
2. `IDs +5`（76 -> 81）

当时结论：

1. 关闭 strict 后，中心区域未匹配检测更容易直接新建 ID；
2. 这些新生 ID 与已有轨迹并行/交替出现，切换概率明显上升；
3. 所以你观察到“刚开始关闭 strict 时 IDSW 大幅增加”是符合机制预期的。

---

### 3.4 阶段 D：zombie 相关测试与逻辑修订

做过的事项：

1. `zombie_max_predict_frames` 做过调整试验（5 -> 10，后又恢复 5）；
2. 评估后未观察到稳定收益，且为控制漂移风险恢复到 5；
3. 删除了 `zombie_iou_thresh` 相关配置与逻辑，避免误导；
4. 当前 zombie 匹配逻辑为“中心距离阈值匹配”，并在超预测帧后使用冻结位置。

当前代码要点：

1. 匹配函数：`_try_match_zombie`
2. 取框函数：`get_tlwh_for_matching`
3. 阈值：`min(zombie_match_max_dist, zombie_dist_thresh)`

---

### 3.5 阶段 E：为“关闭 strict 但压低 IDSW”实施 3 项改法

最终落地的 3 项核心改法：

1. 独立新生阈值：`new_track_thresh`
2. 新生重复抑制：`birth_suppress_iou` + `birth_suppress_center_dist`
3. 新生二次确认：`birth_confirm_frames`

实现位置：

1. `boxmot/trackers/bytetrack/bytetrack.py`
2. `boxmot/configs/trackers/bytetrack.yaml`
3. 测试覆盖：`tests/unit/test_trackers.py` 新增 4 个 ByteTrack 用例

---

### 3.6 阶段 F：实施 3 项改法后的最终 MOT17-04 结果（关闭 strict）

对应结果目录：`runs/ablation/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_104`

配置核心：

1. `strict_entry_gate=False`
2. `new_track_thresh=0.65`
3. `birth_confirm_frames=2`
4. `birth_suppress_iou=0.7`
5. `birth_suppress_center_dist=35`

结果：

1. `HOTA 78.305`
2. `MOTA 89.904`
3. `IDF1 90.485`
4. `IDSW 19`
5. `IDs 73`

与“关闭 strict 但未加 soft gating（run 103）”相比：

1. `IDSW -10`（29 -> 19）
2. `IDs -8`（81 -> 73）
3. `IDF1 +0.862`（89.623 -> 90.485）

与“开启 strict 的参考 improved（IDSW=20, IDs=76）”相比：

1. `IDSW -1`（20 -> 19）
2. `IDs -3`（76 -> 73）
3. `IDF1 +0.669`（89.816 -> 90.485）
4. `MOTA -0.951`（90.855 -> 89.904）

结论：

1. 关闭 strict 并不必然导致 IDSW 高，只要把“新建 ID 入口”做软约束；
2. 当前版本已经把“关闭 strict 后的 IDSW 爆增问题”从 29 拉回到 19。

---

## 4. “20 个 IDSW 到底发生在哪里”的结论

当前结论是“机制级结论”，不是逐事件计数表。

1. 不是全部发生在 zombie；
2. 一部分来自第一轮/第二轮关联误配；
3. 一部分来自 lost 后重现时的误关联；
4. 一部分来自中心区域误新生（strict 关闭时更明显）。

说明：

1. 现有评估输出给的是总量（如 `IDSW=20`），没有直接按“步骤来源”分解；
2. 若要逐个 IDSW 事件定位，需要额外导出事件级日志或写专门分析脚本。

---

## 5. MOT20 外部消融结果（你在另一台机器测得）

你给出的结果（改进算法 vs ByteTrack 原版）：

1. `HOTA`: `72.79 -> 69.35`（-3.45）
2. `MOTA`: `89.81 -> 82.90`（-6.91）
3. `IDF1`: `88.12 -> 83.59`（-4.53）
4. `IDSW`: `665 -> 529`（改善）
5. `MT`: `1137 -> 1023`（下降）
6. `ML`: `89 -> 165`（变差）
7. `Fragments`: `2036 -> 1608`（改善）

解释：

1. 算法变得“更保守”：切换和碎片少了；
2. 但连续覆盖能力下降：更多轨迹未被长期跟住（MT 降、ML 升）；
3. 所以出现“IDSW 改善但 HOTA/MOTA/IDF1 大幅下降”的典型权衡。

---

## 6. 后续改进方法（已讨论并建议）

针对 MOT17 与 MOT20 目标差异，建议分两条线：

1. MOT17 线（当前优先）：
   - 保持 `strict_entry_gate=False`
   - 保持 soft birth gating 三件套
   - 以 `IDSW/IDs` 和 `IDF1` 为主目标微调

2. MOT20 线（人群更密、更易遮挡）：
   - 减少过度保守：
   - `new_track_thresh` 可尝试 `0.60~0.64`
   - `birth_confirm_frames` 可尝试 `1~2`
   - `birth_suppress_iou` 可尝试 `0.55~0.70`
   - `birth_suppress_center_dist` 可尝试 `20~35`
   - 单独建立 MOT20 preset，不与 MOT17 共用同一套阈值

3. 事件级诊断线（建议补工具）：
   - 新增“IDSW 逐事件导出”脚本
   - 给每次切换打标签：第一轮误配 / 第二轮误配 / zombie 复活 / 新生重叠
   - 再做有针对性的规则微调，效率比盲扫参更高

---

## 7. 当前总论

1. “开启 strict 时 IDSW=20 难再降”是已验证的平台区现象；
2. “关闭 strict 后 IDSW +9”是因为中心区误新生放开；
3. 引入 soft birth gating 后，已把关闭 strict 的 IDSW 拉回并优于 strict 版本（19 vs 20）；
4. MOT20 的大幅掉点说明当前参数更偏“防切换”，需要做数据集分治与事件级调优。
