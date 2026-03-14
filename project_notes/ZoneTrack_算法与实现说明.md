# ZoneTrack: 面向固定监控中等密度行人场景的 Zone-Aware ByteTrack 改进

**文档版本**: 1.0  
**最后更新**: 2026-02-27  
**代码基线**: `00ec55125a138cd14641bcacfdd29732cbc0eaf2`  

本说明文档面向“写论文”和“复现实验”两类读者，目标是把 ZoneTrack 的**算法思想、假设前提、每个子模块的动机与实现细节**讲清楚，并做到论文术语与代码实现一一对应。

Repo 内对应实现仍在 `ByteTrack` 追踪器里完成（通过开关参数启用/禁用特性），为了论文叙述清晰，本文将该改进版统一称为 **ZoneTrack**。

---

## 1. 适用场景、目标与核心先验

### 1.1 适用场景

- 固定监控摄像头（静止视角，基本无相机运动）。
- 城市场景、中等密度行人多目标追踪（遮挡存在但不“满屏交叉”）。
- 主要面向后续数据分析任务：人流统计、停留/异常检测、轨迹行为分析等。

### 1.2 优化目标

ZoneTrack 的设计目标优先级是：

1. **降低 `IDSW` 与 `IDs`**（减少 ID 交换、减少身份碎片化产生的新 ID 数）。
2. 尽量不显著降低 `HOTA / MOTA / IDF1`（在保持总体精度的前提下做 ID 稳定性优化）。
3. 在极端/违背先验的场景下，避免指标“崩盘”（尤其是拥挤遮挡极重时）。

### 1.3 核心先验：中心非新生（Central Non-Emergence Prior, CNEP）

在固定监控的中等密度行人场景中：

- 真正“新目标进入画面”的位置几乎总发生在**画面边缘**（或活动区域边缘）。
- 画面**中心区域**出现一个未匹配检测，更多情况下不是“凭空新生”，而是：
  - 旧轨迹短时遮挡/断检后的重新出现；
  - 旧轨迹被过早删除后的回归；
  - 低分检测未被两阶段匹配吸收导致的暂时断链。

而在手持摄像机、行车记录仪等移动视角里，上述先验往往不成立：人可能随着视角变化“在中心出现”。因此 ZoneTrack 的改动本质上是一种**场景先验驱动的轨迹生命周期策略**，并非通用 MOT 的“无条件改进”。

---

## 2. ByteTrack Baseline 回顾（对照 ZoneTrack 改动点）

ByteTrack（原始思想）可以概括为：

- 以检测为输入，使用 KF 预测轨迹位置；
- **高分检测**与（tracked + lost）轨迹池做 IoU 关联；
- **次高分检测**只与未匹配的 tracked 再做一次关联，用于补召回；
- 对未匹配的高分检测，直接作为新轨迹出生（birth），从而保持召回；
- 对连续未匹配的轨迹，超过 buffer 后移除。

在固定监控的实际部署里，ByteTrack 的典型痛点不是“完全追不到人”，而是：

- 由于断检/遮挡，轨迹会在中心区域频繁断开又新生，导致 **IDs 膨胀**；
- 一个人被分裂成多个短轨迹，导致 **IDSW/碎片化** 增多；
- 特别是在城市场景中，中心区域误新生对下游统计会造成系统性偏差。

ZoneTrack 的总体策略是：**保守新生、积极寻回、区域化解释**。

---

## 3. ZoneTrack 总览：三类区域 + 三个子模块

ZoneTrack 的改造可以看成 3 个维度叠加：

1. **区域化（Zone-aware）**：把画面划分为 entry / center / exit 语义区，用不同策略解释“未匹配高分检测”与“轨迹丢失”。
2. **软新生门控（STBG）**：把“新 ID 创建”改造成一个更严格的、可确认的流程，降低误新生与重复新生。
3. **僵尸记忆（PZM）**：把超出 buffer 的 lost 轨迹转入一个有限记忆池，在 center 区优先尝试“复活旧 ID”，降低 IDs/IDSW。

此外，为了让区域划分可泛化到不同监控画面，ZoneTrack 引入：

- **自适应有效区（AEZ）**：从检测分布估计“人群活动有效区”，并单调扩展，避免 ROI 手工标定。
- **离开区延迟删除（EDR）**：对于在边缘离开区丢失的轨迹，延迟删除并避免其进入僵尸池，降低误复活概率。

在 MOT17-04 的消融实验命名里，你已使用：

- **STBG**: Soft Track Birth Gating（软新生门控）
- **PZM**: Persistent Zombie Memory（僵尸记忆）
- **Zone**: Entry/Exit/Adaptive Effective Zone（区域化生命周期管理）

---

## 4. 术语、坐标与实现约定（非常重要）

### 4.1 检测与轨迹的数据格式

- 检测输入 `dets` 是 `N x 6`：`[x1, y1, x2, y2, conf, cls]`。
- `ByteTrack.update()` 内部会拼接 `det_ind`：`dets = hstack([dets, arange(N)])`，变成 `N x 7`。
- 输出 `outputs` 是 `M x 8`：
  - `[x1, y1, x2, y2, track_id, conf, cls, det_ind]`
  - 单测里默认 `track_id` 在第 5 列（索引 4）。

### 4.2 tlwh / xyxy

在本仓库 `bytetrack.py` 内：

- `xyxy` 表示 `[x1, y1, x2, y2]`。
- `tlwh` 表示 `[left(x), top(y), width, height]`。
- `STrack.xyxy` 是属性（基于 KF `mean` 动态计算）。
- `STrack.tlwh` 在 detection track 上来自输入检测（用于 birth/pending 比较）。
- 对于历史轨迹参与距离/抑制等计算时，统一通过：
  - `STrack.get_tlwh_for_matching(frame_id, max_predict_frames)`
  - 它会优先使用 KF `mean` 或冻结态 `frozen_mean` 来生成 tlwh。

### 4.3 “新轨迹激活”与输出时延（很多人会误读）

`STrack.activate()` 里有一个原始 ByteTrack 语义：

- **只有在 `frame_id == 1` 时，新轨迹才会立刻 `is_activated=True` 并出现在输出里**。
- 对于后续帧的新轨迹，即使已经 `activate()`，也需要在下一帧被 `update()` 后才会 `is_activated=True`，从而进入输出。

因此，ZoneTrack 的 `birth_confirm_frames=2` 并不意味着“第 2 帧就输出轨迹”，而是：

- 第一次命中：进入 pending
- 第二次命中：`activate()`（但可能仍不输出）
- 第三次命中：`update()` 后开始稳定输出

这个设计在论文写法上要解释清楚，否则读者会困惑“确认帧数为何与输出延迟不一致”。

---

## 5. AEZ: Adaptive Effective Zone（自适应有效区）

### 5.1 动机

如果把 entry zone 简单定义为“画面边缘 margin”，在固定监控里通常有效，但存在两个现实问题：

1. 不同摄像头视角下，人群活动区域未必覆盖整幅图像，边缘可能是天空/建筑/无效区域。
2. 摄像头可能有裁剪、数字变焦、遮挡区域，导致“固定边缘带”不稳定。

因此 ZoneTrack 用检测分布估计一个**有效区** `E_t`，并围绕 `E_t` 定义 entry band 与 center 区域。

### 5.2 实现位置

- `boxmot/trackers/bytetrack/bytetrack.py`
  - `_update_effective_zone(detections, phase)`
  - `_compute_effective_zone()`
  - `_mark_outside_zone_det_inds(detections)`
  - `_select_expand_candidates(detections, phase)`

### 5.3 关键设计：单调扩展（always_expand）

在 `adaptive_zone_update_mode="always_expand"` 时：

- 每帧先基于**上一帧的有效区**标记“本帧哪些检测在扩展前位于有效区外”：
  - `_outside_zone_det_inds`（存 det_ind）
  - 这一步发生在 `_mark_outside_zone_det_inds()`
- 然后再选择候选检测用于扩展 `E_t`，并且只做单调扩展，不收缩：
  - `E_t = expand(E_{t-1}, bbox(candidates))`

这个“先标记 outside-before-expand，再扩展”的顺序是 ZoneTrack 非常关键的工程细节，因为后续 Step4 会用它解决一个悖论：

- 如果某个检测位于旧有效区外，它很可能代表“新进入/新可见区域”，应允许新建 ID；
- 但如果先扩展有效区，把它包进来了，它就会被判定为“中心区”，从而触发“中心非新生”策略，反而把它误当成旧目标回归，导致漏新生。

因此 ZoneTrack 在 Step4 中对 `_outside_zone_det_inds` 做了一个**优先放行**规则（见第 9 节）。

### 5.4 扩展触发策略

`adaptive_zone_expand_trigger` 支持：

- `all_high`: 使用所有高分检测更新/扩展有效区（最稳健，默认）。
- `outside_high`: 只用位于有效区外的高分检测触发扩展。
- `unmatched_high`: 主要在 Step4 对未匹配高分检测触发扩展，避免匹配阶段把有效区“拉大”。

---

## 6. Entry/Center/Exit 三类区域的定义与语义

### 6.1 Entry zone（进入区）

实现函数：`_is_in_entry_zone(tlwh, img_shape, margin=None)`

行为分两种模式：

1. **固定边缘带模式**（当 `adaptive_zone_enabled=False` 或 `_effective_zone is None`）  
   只要 box 任一边触碰图像边缘带（宽度 `entry_margin`）就认为在 entry zone。
2. **有效区模式**（当 `adaptive_zone_enabled=True` 且 `_effective_zone` 已存在）  
   - box 在有效区外：entry
   - box 在有效区内但落在有效区边缘带（宽度 `adaptive_zone_margin`）：entry
   - 否则：center

一个容易忽略但对复现实验很重要的实现约定：

- 当 `entry_margin <= 0` 时，函数直接返回 `True`（等价于关闭门控，允许新生）。

### 6.2 Center zone（中心区）

中心区不是单独计算出来的，而是：

- “不在 entry zone”的剩余区域。

ZoneTrack 的核心先验（CNEP）就是对 center 区的未匹配检测采取不同解释策略：

1. 优先尝试复活旧 ID（PZM）。
2. 如果复活失败，再决定是否允许新生（可由 strict_entry_gate 控制硬禁止）。

### 6.3 Exit zone（离开区）

实现函数：`_is_in_exit_zone(tlwh, img_shape)`

Exit zone 语义用于轨迹丢失时的生命周期控制：

- 当轨迹在 exit zone 丢失时，标记 `exit_pending=True`；
- 经过 `exit_zone_remove_grace` 帧后，直接移除（而不是进入 zombie 池）。

直觉解释：

- 固定监控里，边缘丢失更可能代表“离开画面”，继续保留为 zombie 容易误复活到其他行人身上；
- 但又不希望一丢失就马上删（可能只是边缘断检/框抖动），所以给 grace。

实现注意：

- Exit zone 的判定**不依赖有效区**，只基于图像边缘 `exit_zone_margin`。
- `exit_zone_margin <= 0` 会禁用 exit zone。

---

## 7. STBG: Soft Track Birth Gating（软新生门控）

STBG 解决的是“误新生/重复新生”导致的 IDs 膨胀与 IDSW 增多，尤其在固定监控中心区断检回归时非常常见。

### 7.1 组成 1：独立新生阈值 `new_track_thresh`

问题：ByteTrack 的新生阈值与匹配阈值常常耦合；在固定监控中，你希望“匹配尽量宽松救回旧轨迹”，但“新生尽量保守”。

实现：Step4 对每个未匹配高分检测先判断：

- `if det_track.conf < new_track_thresh: continue`

这样可以独立收紧 birth 入口，而不影响 Step1/2 的关联策略。

### 7.2 组成 2：多帧确认 `birth_confirm_frames`（pending birth）

问题：单帧误检或抖动在 ByteTrack 中会立刻产生新 ID。

实现位置：

- `pending_births: list[dict]`（元素结构见下）
- `_try_activate_new_track(det_track, activated_starcks, reference_tracks)`
- `_prune_pending_births()`
- `_find_pending_birth_idx(det_track)`

pending 元素结构：

```text
{
  "track": det_track,        # 当前帧候选 detection track
  "hits": 1..K,              # 连续命中次数
  "last_frame": frame_count, # 上一次命中帧
}
```

确认逻辑（实现精确对应代码）：

- 若 `birth_confirm_frames <= 1`：直接走 duplicate suppression，然后 `activate()`
- 若 `birth_confirm_frames > 1`：
  1. 先找当前 det 是否能与某个 pending 对应（`_find_pending_birth_idx`）。
  2. 若无对应：新增 pending（hits=1）。
  3. 若有对应：hits += 1。
  4. hits 达到阈值：对当前 det `activate()`，并把它加入 `activated_starcks`。

**连续性约束（非常关键）**：

- `_prune_pending_births` 会删除 “`frame_count - last_frame > 1`” 的 pending。
- 这意味着确认命中必须近似连续（最多允许 miss 1 帧）。

匹配 pending 的相似性约束：

- `_birth_confirm_iou = 0.3`
- 同时结合中心距 `max_dist`（优先使用 `birth_suppress_center_dist`，否则默认 40px）
- 满足 “IoU 足够” 或 “距离足够近” 才视为同一个 pending。

### 7.3 组成 3：重复新生抑制（duplicate birth suppression）

问题：同一人可能被检测成多个高度重叠框，或者同一目标在同一帧/邻近帧触发重复 birth，导致 double-id。

实现位置：

- `_collect_birth_reference_tracks(...)`: 把多个轨迹池去重合并成 reference set。
- `_is_birth_suppressed(det_track, reference_tracks)`:
  - IoU 抑制：`birth_suppress_iou`
  - 中心距抑制：`birth_suppress_center_dist`

reference_tracks 在 Step4 中通常会包含：

- `active_tracks` / `lost_stracks` / `zombie_stracks`
- 当帧刚激活的 `activated_starcks`
- 当帧刚 re-find 的 `refind_stracks`
- 当帧新 lost 的 `lost_stracks`

这样做的目的：避免“同一帧内部”和“跨池重复”产生的双 ID。

---

## 8. PZM: Persistent Zombie Memory（僵尸记忆与寻回）

### 8.1 动机

原始 ByteTrack 会在轨迹丢失超过 buffer 后删除轨迹。固定监控里，这会导致：

- 人被遮挡较久或被大物体挡住后再出现时，旧 ID 已被删，只能新生，IDs 增加。

PZM 的策略是：

- 对“丢失很久但未明确离开”的轨迹，保留一个有限的历史池（zombie pool）。
- 在中心区出现未匹配高分检测时，先尝试与 zombie pool 进行复活匹配，从而复用旧 ID。

### 8.2 轨迹进入 zombie pool 的条件

实现位置：`ByteTrack.update()` 的 Step6。

- 当 `zombie_enabled == True` 时：
  - 对每条 lost track 计算 `frames_lost = frame_count - lost_frame_id`
  - 若 `frames_lost >= zombie_transition_frames`：
    - 轨迹不再按 baseline 逻辑删除，而是加入 `zombie_stracks`
  - 否则仍留在 `lost_stracks`，可被 Step1 IoU 直接 re_activate（短期遮挡恢复）

`zombie_enabled` 的开关条件是：

```text
zombie_max_history > 0
and zombie_transition_frames > 0
and zombie_match_max_dist > 0
```

### 8.3 Zombie 寻回匹配（center-distance NN）

实现函数：`_try_match_zombie(detection, zombie_stracks, max_dist=None)`

算法：

1. 对每个 zombie，用 `get_tlwh_for_matching()` 得到用于匹配的框位置；
2. 计算 detection 与 zombie 的中心距；
3. 取最小距离的 zombie，若距离 < `max_dist` 则 `re_activate(new_id=False)` 复活旧 ID。

阈值融合：

- `max_dist = zombie_match_max_dist`
- 若 `zombie_dist_thresh > 0`，则 `max_dist = min(max_dist, zombie_dist_thresh)`

### 8.4 冻结策略：限制 KF 漂移导致误匹配

长时间丢失时，KF 预测会漂移，导致中心距匹配不可靠。

ZoneTrack 的实现策略：

- 对 lost track，当 `frames_lost >= zombie_max_predict_frames` 时，把 `track.mean` 拷贝到 `track.frozen_mean`。
- 在 `get_tlwh_for_matching()` 中：
  - 若 `frozen_mean` 存在且已经超过 `max_predict_frames`，就用冻结态计算 tlwh；
  - 否则用当前 KF mean。

直觉：超过一定帧数后，不再信任运动外推，宁愿用“最后可信位置”。

### 8.5 Zombie 池容量控制

实现位置：Step8。

- `zombie_stracks` 超过 `zombie_max_history` 时按 `lost_frame_id` 排序，移除更旧的。
- 被成功复活的 zombie，会在 Step7 从 zombie pool 中删除，避免同一 ID 被重复复活。

---

## 9. Zone-Aware Step4：未匹配高分检测的“区域化解释”核心

ZoneTrack 相比原始 ByteTrack 的最大结构性改动在 Step4。

实现位置：`ByteTrack.update()` 中 Step4 逻辑（见 `bytetrack.py` 860 行附近）。

### 9.1 Step4 总规则（逐条对应实现）

对每个未匹配的高分 detection：

1. **Birth 置信度门控**：`conf >= new_track_thresh` 才继续。
2. **AEZ 扩展（可选）**：always_expand 且触发策略为 `unmatched_high/outside_high` 时，Step4 会用该 detection 触发一次 `_update_effective_zone(..., phase='step4')`。
3. **outside-before-expand 放行**（关键工程补丁）：
   - 若该 detection 在本帧扩展前位于有效区外（`det_ind in _outside_zone_det_inds`），直接允许走 STBG 新生流程；
   - 这用于保护“新进入/新活动区域”的合理新生，不被中心先验误杀。
4. **Entry vs Center 判定**：
   - `is_entry = _is_in_entry_zone(det_tlwh, img_shape)`
5. `is_entry == True`：
   - 允许新生：走 `_try_activate_new_track`（STBG）
6. `is_entry == False`（center 区）：
   - 先尝试 `_try_match_zombie(det_track, zombie_stracks)`：
     - 成功：复活旧 ID，加入 `refind_stracks`
     - 失败：进入 “center 区是否允许新生” 分支
7. **严格中心禁生（可选硬先验）**：
   - 若 `strict_entry_gate == True` 且 `entry_margin > 0`：
     - center 区复活失败后直接 `continue`（禁止新生）
   - 否则：center 区也允许走 STBG 新生流程（默认配置就是如此）

### 9.2 为何把 zombie 匹配放在 Step4，而不是 Step1？

实现上，Step1 的 `strack_pool` 仍是 (tracked + lost)，保持 ByteTrack 原结构：

- 短期遮挡恢复：交给 IoU/KF 的“标准 re_activate”
- 长期丢失恢复：交给 PZM 的“center-distance re_activate”

这样做的好处：

- 不影响 ByteTrack 原始关联稳定性；
- zombie 匹配只在“真正未匹配且高分”的检测上发生，控制误匹配与计算量。

---

## 10. Exit-aware Delayed Removal（离开区延迟删除）

实现位置：

- Step3：轨迹从 tracked -> lost 时，若 `exit_zone_enabled`，则判定是否在 exit zone，设置 `track.exit_pending`。
- Step6：对 `exit_pending` 的 lost track，若 `frames_lost >= exit_zone_remove_grace` 则移除。

设计意图：

- exit zone 丢失更像离开，尽早移除可以减少 zombie 误复活；
- 但立刻移除会在边缘断检时制造碎片化，所以设置 grace。

实现注意（与论文写法相关）：

- `exit_pending` 是动态属性（通过 `_set_exit_pending(track, bool)` 注入）。
- 在当前实现里，exit zone 的判定调用了 `track.tlwh`。如需确保其表示“最后一次观测位置”，应保证 track 的 tlwh 在 `update/re_activate` 时同步更新；否则建议改为使用 `track.get_tlwh_for_matching(...)` 参与 exit zone 判定。

---

## 11. ZoneTrack 完整算法伪代码（论文级）

下面伪代码刻意与 `ByteTrack.update()` 的阶段对齐，便于你在论文中写成 Algorithm 1 并与实现对应。

```text
Algorithm 1: ZoneTrack Update (per frame t)
Input:
  D_t = {(b_i, s_i, c_i)} detections in xyxy with confidence and class
  I_t image (optional, for size)
State:
  A active_tracks, L lost_stracks, Z zombie_stracks
  P pending_births
  E effective_zone, O outside_zone_det_inds

1:  Append det_ind to each detection; split D_t into:
      D_high = {d | s(d) > track_thresh}
      D_low  = {d | min_conf < s(d) < track_thresh}
2:  Prune pending births P by consecutive-miss rule
3:  Update effective zone:
      O <- mark detections outside E (before expansion)
      E <- update/expand E using D_high (always_expand / warmup_once)

4:  Step1 Association (high):
      pool <- A(tracked) ∪ L
      KF predict(pool)
      match by IoU+score between pool and D_high
      matched: update/re_activate, clear exit_pending
      unmatched tracks -> proceed

5:  Step2 Association (low):
      match remaining tracked with D_low by IoU
      matched: update, clear exit_pending
      unmatched tracked:
         exit_pending <- in_exit_zone(track)
         mark_lost; set lost_frame_id=t; push to lost_new

6:  Unconfirmed handling:
      match unconfirmed with remaining D_high; unmatched unconfirmed removed

7:  Step4 Unmatched-high processing:
      For each unmatched detection d in D_high:
         if s(d) < new_track_thresh: continue
         optionally expand E in step4 (depending on expand_trigger)
         if det_ind(d) in O: Birth(d) via STBG; continue
         if InEntryZone(d, E): Birth(d) via STBG; continue
         # Center zone
         if ZombieMatch(d, Z): re_activate; continue
         if strict_entry_gate and entry_margin>0: continue
         Birth(d) via STBG

8:  Step5/6 Lifecycle maintenance:
      L <- L ∪ lost_new
      if zombie_enabled:
         freeze lost tracks after zombie_max_predict_frames
      for each track in L:
         if exit_pending and frames_lost>=grace: remove
         else if zombie_enabled and frames_lost>=zombie_transition_frames: move to Z
         else if not zombie_enabled and frames_lost>max_time_lost: remove
      limit Z by zombie_max_history; delete rescued ids from Z

9:  Output:
      return {track in A | track.is_activated==True} as xyxy + id + conf + cls + det_ind

Birth(d) via STBG:
  if suppressed(d, refs): reject
  if birth_confirm_frames<=1: activate(d)
  else: pending-match(d) and activate only after consecutive hits
```

---

## 12. 参数与实现对照（论文写作与复现实验用）

### 12.1 配置文件

本仓库的 tracker YAML 采用“可调参空间 + default”的格式：评测时实际生效的是 `default` 值。

- Baseline（原版对齐）: `boxmot/configs/trackers/bytetrack_original.yaml`
- Improved（ZoneTrack 固定版）: `boxmot/configs/trackers/bytetrack_improved.yaml`

### 12.2 Baseline vs ZoneTrack 默认参数（关键项）

| Category | Param | Baseline default | ZoneTrack default | Code Field |
|---|---|---:|---:|---|
| Core | `track_thresh` | 0.5 | 0.5 | `self.track_thresh` |
| Core | `new_track_thresh` | 0.5 | 0.65 | `self.new_track_thresh` |
| Core | `track_buffer` | 30 | 30 | `self.track_buffer` / `self.buffer_size` |
| Assoc | `match_thresh` | 0.8 | 0.8 | `self.match_thresh` |
| Zone | `entry_margin` | 0 | 50 | `self.entry_margin` |
| Zone | `strict_entry_gate` | false | false | `self.strict_entry_gate` |
| STBG | `birth_confirm_frames` | 1 | 2 | `self.birth_confirm_frames` |
| STBG | `birth_suppress_iou` | 0.0 | 0.7 | `self.birth_suppress_iou` |
| STBG | `birth_suppress_center_dist` | 0 | 35 | `self.birth_suppress_center_dist` |
| PZM | `zombie_max_history` | 0 | 100 | `self.zombie_max_history` |
| PZM | `zombie_transition_frames` | 30 | 30 | `self.zombie_transition_frames` |
| PZM | `zombie_match_max_dist` | 200 (disabled by max_history=0) | 200 | `self.zombie_match_max_dist` |
| PZM | `zombie_dist_thresh` | 999999 (disabled) | 150 | `self.zombie_dist_thresh` |
| PZM | `zombie_max_predict_frames` | 0 | 5 | `self.zombie_max_predict_frames` |
| Exit | `exit_zone_enabled` | false | true | `self.exit_zone_enabled` |
| Exit | `exit_zone_margin` | 50 (unused) | 50 | `self.exit_zone_margin` |
| Exit | `exit_zone_remove_grace` | 30 | 30 | `self.exit_zone_remove_grace` |
| AEZ | `adaptive_zone_enabled` | false | true | `self.adaptive_zone_enabled` |
| AEZ | `adaptive_zone_update_mode` | warmup_once | always_expand | `self.adaptive_zone_update_mode` |
| AEZ | `adaptive_zone_expand_trigger` | all_high | all_high | `self.adaptive_zone_expand_trigger` |
| AEZ | `adaptive_zone_warmup` | 10 | 10 | `self.adaptive_zone_warmup` |
| AEZ | `adaptive_zone_margin` | 50 | 50 | `self.adaptive_zone_margin` |
| AEZ | `adaptive_zone_padding` | 1.2 | 1.2 | `self.adaptive_zone_padding` |

STBG 的两个内部常量（当前写死在代码中）：

- `_birth_confirm_iou = 0.3`
- `_birth_pending_max_miss = 1`

如果你在论文里强调可调性，可以考虑把这两个也提升为 YAML 参数；但现阶段写在实现细节里更符合“工程固定版”。

---

## 13. 单元测试：关键语义的可执行证据

`tests/unit/test_trackers.py` 已覆盖以下 ZoneTrack 关键语义（建议论文复现时附录引用）：

- AEZ 单调扩展：`test_bytetrack_adaptive_zone_always_expand_grows_monotonically`
- outside-before-expand 放行：`test_bytetrack_outside_before_expand_keeps_new_id_creation`
- exit zone 禁用与 grace：`test_bytetrack_exit_zone_margin_zero_is_disabled` / `test_bytetrack_exit_zone_remove_grace_delays_removal`
- STBG 三件套：
  - `new_track_thresh` 阻断低置信 birth：`test_bytetrack_new_track_thresh_blocks_low_conf_birth`
  - `birth_confirm_frames=2` 需要两次命中：`test_bytetrack_birth_confirm_requires_two_hits`
  - IoU 重复抑制防双 ID：`test_bytetrack_birth_duplicate_suppression_blocks_double_id`
  - pending 必须连续：`test_bytetrack_pending_birth_expires_without_consecutive_confirmation`

---

## 14. 失败模式与工程建议（论文 Discussion 可直接复用）

### 14.1 高密度拥挤（如 MOT20）下的风险

当画面几乎“满屏行人”时：

- center-distance 的 zombie 近邻匹配更容易误配（最近的不一定是同一人）。
- 过强的 birth 抑制可能降低真实新生与恢复速度，导致 AssA/HOTA 下降。

这解释了为什么 ZoneTrack 在 MOT20 上常呈现 trade-off：IDs/IDSW 显著下降，但 HOTA 可能下滑。

### 14.2 移动摄像头/大视角变化不适用

若相机移动或画面中心经常出现“真实新生”，CNEP 不成立，此时建议：

- 关闭 entry gate/AEZ：`entry_margin=0`, `adaptive_zone_enabled=false`
- 关闭 strict_entry_gate（或保持 false）
- 视情况关闭 PZM（防止长时误复活）

