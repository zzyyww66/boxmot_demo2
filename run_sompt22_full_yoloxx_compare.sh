#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zyw/code/boxmot_demo2"
PROJECT="runs/sompt22_full_yoloxx_compare"
LOG_DIR="$PROJECT/logs"
PREFIX="$PROJECT/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack"

mkdir -p "$ROOT/$LOG_DIR"
cd "$ROOT"

run_log="$LOG_DIR/orchestrator.log"
baseline_log="$LOG_DIR/baseline_eval.log"
improved_log="$LOG_DIR/improved_eval.log"
report_md="$PROJECT/FINAL_COMPARISON.md"

list_dirs() {
  ls -1d ${PREFIX}* 2>/dev/null || true
}

latest_dir() {
  ls -1dt ${PREFIX}* 2>/dev/null | head -n 1 || true
}

echo "[$(date '+%F %T')] Start SOMPT22 full compare (YOLOX_X)" | tee -a "$run_log"

list_dirs | sort > "$LOG_DIR/mot_dirs_before_baseline.txt"
uv run python -m boxmot.engine.cli eval \
  --source SOMPT22-full \
  --tracking-method bytetrack \
  --tracker-config boxmot/configs/trackers/bytetrack_original.yaml \
  --yolo-model yolox_x_MOT17_ablation.pt \
  --reid-model lmbn_n_duke.pt \
  --project "$PROJECT" \
  --name baseline_eval \
  --device 0 \
  --batch-size 4 \
  --no-auto-batch \
  --resume \
  --ci | tee "$baseline_log"

list_dirs | sort > "$LOG_DIR/mot_dirs_after_baseline.txt"
baseline_dir="$(comm -13 "$LOG_DIR/mot_dirs_before_baseline.txt" "$LOG_DIR/mot_dirs_after_baseline.txt" | tail -n 1 || true)"
if [[ -z "${baseline_dir}" ]]; then
  baseline_dir="$(latest_dir)"
fi
echo "$baseline_dir" > "$LOG_DIR/baseline_mot_dir.txt"
echo "[$(date '+%F %T')] Baseline dir: $baseline_dir" | tee -a "$run_log"

list_dirs | sort > "$LOG_DIR/mot_dirs_before_improved.txt"
uv run python -m boxmot.engine.cli eval \
  --source SOMPT22-full \
  --tracking-method bytetrack \
  --tracker-config boxmot/configs/trackers/bytetrack_improved.yaml \
  --yolo-model yolox_x_MOT17_ablation.pt \
  --reid-model lmbn_n_duke.pt \
  --project "$PROJECT" \
  --name improved_eval \
  --device 0 \
  --batch-size 4 \
  --no-auto-batch \
  --resume \
  --ci | tee "$improved_log"

list_dirs | sort > "$LOG_DIR/mot_dirs_after_improved.txt"
improved_dir="$(comm -13 "$LOG_DIR/mot_dirs_before_improved.txt" "$LOG_DIR/mot_dirs_after_improved.txt" | tail -n 1 || true)"
if [[ -z "${improved_dir}" ]]; then
  improved_dir="$(latest_dir)"
fi
echo "$improved_dir" > "$LOG_DIR/improved_mot_dir.txt"
echo "[$(date '+%F %T')] Improved dir: $improved_dir" | tee -a "$run_log"

python - "$baseline_dir" "$improved_dir" "$report_md" <<'PY'
import csv
import pathlib
import sys

baseline_dir = pathlib.Path(sys.argv[1])
improved_dir = pathlib.Path(sys.argv[2])
report_path = pathlib.Path(sys.argv[3])

metrics = ["HOTA___AUC", "MOTA", "IDF1", "IDSW", "CLR_FP", "CLR_FN", "IDTP", "IDFN", "IDFP"]

def read_combined_row(mot_dir: pathlib.Path) -> dict:
    candidates = [mot_dir / "pedestrian_detailed.csv", mot_dir / "person_detailed.csv"]
    detail = next((p for p in candidates if p.exists()), None)
    if detail is None:
        raise FileNotFoundError(f"No detailed csv found in {mot_dir}")
    with detail.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty csv: {detail}")
    row = next((r for r in rows if r.get("seq") == "COMBINED"), rows[-1])
    out = {}
    for k in metrics:
        if k in row and row[k] != "":
            out[k] = float(row[k])
    return out

base = read_combined_row(baseline_dir)
imp = read_combined_row(improved_dir)

def fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}"

lines = []
lines.append("# SOMPT22 Full Results (YOLOX_X, no ReID)")
lines.append("")
lines.append(f"- Baseline dir: `{baseline_dir}`")
lines.append(f"- Improved dir: `{improved_dir}`")
lines.append("")
lines.append("| Metric | Baseline | Improved | Delta (Imp-Base) |")
lines.append("|---|---:|---:|---:|")
for m in metrics:
    b = base.get(m, float("nan"))
    i = imp.get(m, float("nan"))
    d = i - b
    lines.append(f"| {m} | {fmt(b)} | {fmt(i)} | {fmt(d)} |")

if "IDSW" in base and "IDSW" in imp and base["IDSW"] != 0:
    rel = (base["IDSW"] - imp["IDSW"]) / base["IDSW"] * 100.0
    lines.append("")
    lines.append(f"- IDSW reduction: `{rel:.2f}%`")

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote report to {report_path}")
PY

echo "[$(date '+%F %T')] Completed. Report: $report_md" | tee -a "$run_log"
