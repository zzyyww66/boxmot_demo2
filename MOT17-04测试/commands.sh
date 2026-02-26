#!/usr/bin/env bash
set -euo pipefail

run_eval () {
  local stage_name="$1"
  local cfg="$2"
  local log="MOT17-04测试/logs/${stage_name}.log"

  echo "[RUN] ${stage_name}"
  uv run python -m boxmot.engine.cli eval \
    --source MOT17-full-04 \
    --tracking-method bytetrack \
    --tracker-config "${cfg}" \
    --yolo-model yolox_x_MOT17_ablation.pt \
    --reid-model lmbn_n_duke.pt \
    --project "MOT17-04测试/runs" \
    --name "${stage_name}" \
    --device 0 \
    --batch-size 4 \
    --ci 2>&1 | tee "${log}"
}

run_eval stage0_baseline "MOT17-04测试/configs/stage0_baseline.yaml"
run_eval stage1_stbg "MOT17-04测试/configs/stage1_stbg.yaml"
run_eval stage2_stbg_pzm "MOT17-04测试/configs/stage2_stbg_pzm.yaml"
run_eval stage3_stbg_pzm_zone "MOT17-04测试/configs/stage3_stbg_pzm_zone.yaml"
