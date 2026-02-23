# SOMPT22-10 ByteTrack 原版 vs 改进版对比（2026-02-23）

## 1. 测试目标

- 数据集：`SOMPT22/train/SOMPT22-10`
- 检测器：`yolov8m_pretrain_crowdhuman.pt`
- 跟踪器：`bytetrack`（原版配置 vs 改进配置）
- 约束：不使用 ReID 特征进行关联（motion-only）


## 2. 固定配置

### 2.1 数据集配置

- 文件：`boxmot/configs/datasets/SOMPT22-full-10.yaml`
- 核心设置：
  - `source: /home/zyw/code/boxmot_demo2/.venv/lib/python3.11/site-packages/trackeval/SOMPT22`
  - `split: train`
  - `sequence_filter: [SOMPT22-10]`
  - `eval_classes: {1: pedestrian}`

### 2.2 跟踪器配置

- 原版：`boxmot/configs/trackers/bytetrack_original.yaml`
- 改进版：`boxmot/configs/trackers/bytetrack_improved.yaml`
  - 关键固定参数：`strict_entry_gate=false`、`entry_margin=50`、soft birth gating + zombie + exit zone + adaptive zone


## 3. 执行命令

### 3.1 原版（首次运行，生成 dets 缓存）

```bash
uv run python -m boxmot.engine.cli eval \
  --source SOMPT22-full-10 \
  --tracking-method bytetrack \
  --tracker-config boxmot/configs/trackers/bytetrack_original.yaml \
  --yolo-model yolov8m_pretrain_crowdhuman.pt \
  --reid-model lmbn_n_duke.pt \
  --project runs/sompt22_compare_v8m_s10 \
  --name baseline_eval \
  --device 0 \
  --batch-size 4 \
  --ci
```

### 3.2 改进版（复用同一份 dets 缓存）

```bash
uv run python -m boxmot.engine.cli eval \
  --source SOMPT22-full-10 \
  --tracking-method bytetrack \
  --tracker-config boxmot/configs/trackers/bytetrack_improved.yaml \
  --yolo-model yolov8m_pretrain_crowdhuman.pt \
  --reid-model lmbn_n_duke.pt \
  --project runs/sompt22_compare_v8m_s10 \
  --name improved_eval \
  --device 0 \
  --batch-size 4 \
  --ci
```

日志确认改进版命中缓存：
- `Skipping SOMPT22-10 (cached complete; 1800/1800 frames).`

### 3.3 原版（复跑一次，仅为补齐同口径 summary）

```bash
uv run python -m boxmot.engine.cli eval \
  --source SOMPT22-full-10 \
  --tracking-method bytetrack \
  --tracker-config boxmot/configs/trackers/bytetrack_original.yaml \
  --yolo-model yolov8m_pretrain_crowdhuman.pt \
  --reid-model lmbn_n_duke.pt \
  --project runs/sompt22_compare_v8m_s10 \
  --name baseline_eval2 \
  --device 0 \
  --batch-size 4 \
  --ci
```


## 4. 输入缓存与结果路径

- 检测缓存（两组共用）：
  - `runs/sompt22_compare_v8m_s10/dets_n_embs/yolov8m_pretrain_crowdhuman/dets/SOMPT22-10.txt`
- 轨迹输出：
  - 原版：`runs/sompt22_compare_v8m_s10/mot/yolov8m_pretrain_crowdhuman_lmbn_n_duke_bytetrack_3/SOMPT22-10.txt`
  - 改进：`runs/sompt22_compare_v8m_s10/mot/yolov8m_pretrain_crowdhuman_lmbn_n_duke_bytetrack_2/SOMPT22-10.txt`


## 5. 结果对比（SOMPT22-10）

| 指标 | Baseline | Improved | Delta (Improved-Baseline) | 变化率 |
|---|---:|---:|---:|---:|
| HOTA | 56.14 | 52.80 | -3.34 | -5.95% |
| MOTA | 69.04 | 69.27 | +0.23 | +0.33% |
| IDF1 | 65.24 | 60.88 | -4.36 | -6.68% |
| AssA | 54.03 | 48.09 | -5.94 | -10.99% |
| AssRe | 59.85 | 61.09 | +1.24 | +2.07% |
| IDSW | 278 | 258 | -20 | -7.19% |
| IDs | 332 | 133 | -199 | -59.94% |


## 6. 结论

- 在 SOMPT22-10 上，改进版相对原版：
  - `IDSW` 有下降（278 -> 258）
  - `IDs` 显著下降（332 -> 133）
  - `MOTA` 小幅上升
  - 但 `HOTA/IDF1/AssA` 下降，说明该参数组在此数据上存在明显权衡
