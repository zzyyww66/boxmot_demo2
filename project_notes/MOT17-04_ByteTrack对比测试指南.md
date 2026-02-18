# MOT17-04 ByteTrack 对比测试指南（修订版）

**文档版本**: 2.2  
**最后更新**: 2026-02-18

---

## 1. 测试目标

对 `MOT17-04` 单序列进行 baseline 与 improved 的可复现对比，重点观察：

- HOTA / MOTA / IDF1
- IDSW / IDs

---

## 2. 关键修订点

旧流程存在两个问题：

1. 脚本名是 `MOT17-04`，但可能实际跑整个 `MOT17-ablation`。
2. 指标通过 stdout 文本提取，容易受日志格式影响。

修订后：

- `run_mot17_04_comparison.py` 会构造仅含 `MOT17-04` 的子数据源；
- 指标从结果目录的 `pedestrian_detailed.csv` 读取 `seq=MOT17-04` 行。

---

## 3. 环境准备

```bash
cd /home/zyw/code/boxmot_demo2
uv sync --all-extras --all-groups
uv run python -m boxmot.engine.cli --help
```

---

## 4. 对比配置

### 4.1 Baseline（关闭改进能力）

```python
BASELINE_CONFIG = {
    "entry_margin": 0,
    "strict_entry_gate": False,
    "zombie_iou_thresh": 0.3,
    "zombie_max_history": 0,
    "zombie_dist_thresh": 999999,
    "zombie_max_predict_frames": 0,
    "zombie_transition_frames": 30,
    "zombie_match_max_dist": 200,
    "lost_max_history": 0,
    "exit_zone_enabled": False,
    "exit_zone_margin": 50,
    "exit_zone_remove_grace": 30,
    "adaptive_zone_enabled": False,
    "adaptive_zone_update_mode": "warmup_once",
    "adaptive_zone_expand_trigger": "all_high",
    "adaptive_zone_min_box_area": 0,
}
```

### 4.2 Improved（启用改进能力）

```python
IMPROVED_CONFIG = {
    "entry_margin": 50,
    "strict_entry_gate": True,
    "zombie_iou_thresh": 0.3,
    "zombie_max_history": 100,
    "zombie_dist_thresh": 150,
    "zombie_max_predict_frames": 5,
    "zombie_transition_frames": 30,
    "zombie_match_max_dist": 200,
    "lost_max_history": 0,
    "exit_zone_enabled": True,
    "exit_zone_margin": 50,
    "exit_zone_remove_grace": 30,
    "adaptive_zone_enabled": True,
    "adaptive_zone_update_mode": "always_expand",
    "adaptive_zone_expand_trigger": "all_high",
    "adaptive_zone_min_box_area": 0,
}
```

---

## 5. 运行方式

### 5.1 完整对比

```bash
uv run python run_mot17_04_comparison.py
```

### 5.2 仅运行改进版

```bash
uv run python run_improved_only.py
```

### 5.3 推荐：只做关联+评估（复现实验口径）

```bash
uv run python run_mot17_04_assoc_eval_only.py
```

---

## 6. 结果查看

脚本会输出每次 run 的真实目录（`runs/ablation/mot/...`）。

重点文件：

- `pedestrian_detailed.csv`（按序列）
- `pedestrian_summary.txt`（汇总）
- `MOT17-04.txt`（轨迹输出）

### 6.1 当前代码的期望指标（`exit_zone_remove_grace=30`）

- Baseline: `HOTA 77.861 / MOTA 90.305 / IDF1 88.638 / IDSW 32 / IDs 86`
- Improved: `HOTA 78.375 / MOTA 90.855 / IDF1 89.816 / IDSW 20 / IDs 76`

---

## 7. 故障排查

### 7.1 数据缺失

```bash
ls -la boxmot/engine/trackeval/MOT17-ablation/train/MOT17-04/
```

### 7.2 依赖问题

```bash
uv sync --all-extras --all-groups
```

### 7.3 GPU 不可用

```bash
export CUDA_VISIBLE_DEVICES=""
```
