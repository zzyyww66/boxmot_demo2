from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path("MOT17-04测试")

STAGES = [
    {
        "stage": "S0",
        "name": "original_baseline",
        "display": "Stage 0 - Original ByteTrack (Baseline)",
        "config": ROOT / "configs/stage0_baseline.yaml",
        "log": ROOT / "logs/stage0_baseline.log",
        "output": ROOT / "runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack",
        "features": {"STBG": False, "PZM": False, "Zone": False},
    },
    {
        "stage": "S1",
        "name": "stbg",
        "display": "Stage 1 - Baseline + STBG",
        "config": ROOT / "configs/stage1_stbg.yaml",
        "log": ROOT / "logs/stage1_stbg.log",
        "output": ROOT / "runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_2",
        "features": {"STBG": True, "PZM": False, "Zone": False},
    },
    {
        "stage": "S2",
        "name": "stbg_pzm",
        "display": "Stage 2 - Baseline + STBG + PZM",
        "config": ROOT / "configs/stage2_stbg_pzm.yaml",
        "log": ROOT / "logs/stage2_stbg_pzm.log",
        "output": ROOT / "runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_3",
        "features": {"STBG": True, "PZM": True, "Zone": False},
    },
    {
        "stage": "S3",
        "name": "stbg_pzm_zone",
        "display": "Stage 3 - Baseline + STBG + PZM + Zone",
        "config": ROOT / "configs/stage3_stbg_pzm_zone.yaml",
        "log": ROOT / "logs/stage3_stbg_pzm_zone.log",
        "output": ROOT / "runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_4",
        "features": {"STBG": True, "PZM": True, "Zone": True},
    },
]

TRACKER_KEYS = [
    "min_conf",
    "track_thresh",
    "new_track_thresh",
    "match_thresh",
    "entry_margin",
    "strict_entry_gate",
    "birth_confirm_frames",
    "birth_suppress_iou",
    "birth_suppress_center_dist",
    "zombie_max_history",
    "zombie_dist_thresh",
    "zombie_max_predict_frames",
    "exit_zone_enabled",
    "adaptive_zone_enabled",
    "adaptive_zone_update_mode",
]

CORE_METRICS = [
    "HOTA",
    "MOTA",
    "IDF1",
    "DetA",
    "AssA",
    "IDSW",
    "Frag",
    "CLR_FP",
    "CLR_FN",
    "IDs",
    "MTR",
    "MLR",
]


def parse_number(s: str) -> int | float:
    if any(c in s for c in [".", "e", "E"]):
        return float(s)
    return int(s)


def read_summary(summary_path: Path) -> dict[str, int | float]:
    text = summary_path.read_text(encoding="utf-8").strip().splitlines()
    if len(text) < 2:
        raise ValueError(f"Invalid summary file: {summary_path}")
    header = text[0].split()
    values = text[1].split()
    if len(header) != len(values):
        raise ValueError(f"Header/value length mismatch in {summary_path}")
    return {k: parse_number(v) for k, v in zip(header, values)}


def read_config_defaults(cfg_path: Path) -> dict[str, Any]:
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for k in TRACKER_KEYS:
        v = data.get(k)
        if isinstance(v, dict) and "default" in v:
            out[k] = v["default"]
        else:
            out[k] = v
    return out


def delta_metrics(current: dict[str, int | float], base: dict[str, int | float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in current.items():
        bv = base.get(k)
        if isinstance(v, (int, float)) and isinstance(bv, (int, float)):
            out[k] = round(float(v) - float(bv), 6)
    return out


def fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if abs(v - round(v)) < 1e-12:
            return str(int(round(v)))
        return f"{v:.3f}"
    return str(v)


def generate() -> None:
    stages_out: list[dict[str, Any]] = []

    for s in STAGES:
        summary = read_summary(s["output"] / "pedestrian_summary.txt")
        cfg_defaults = read_config_defaults(s["config"])
        stages_out.append(
            {
                "stage": s["stage"],
                "name": s["name"],
                "display": s["display"],
                "features": s["features"],
                "config_path": str(s["config"]),
                "log_path": str(s["log"]),
                "output_dir": str(s["output"]),
                "tracker_params": cfg_defaults,
                "metrics": summary,
            }
        )

    baseline = stages_out[0]["metrics"]
    prev = None
    for st in stages_out:
        st["delta_vs_baseline"] = delta_metrics(st["metrics"], baseline)
        st["delta_vs_previous"] = (
            delta_metrics(st["metrics"], prev["metrics"]) if prev is not None else {}
        )
        prev = st

    best_hota = max(stages_out, key=lambda x: float(x["metrics"]["HOTA"]))
    best_mota = max(stages_out, key=lambda x: float(x["metrics"]["MOTA"]))
    best_idf1 = max(stages_out, key=lambda x: float(x["metrics"]["IDF1"]))

    result = {
        "experiment": {
            "name": "MOT17-04 ByteTrack Ablation",
            "dataset": "MOT17-04",
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_dir": str(ROOT),
            "command_script": str(ROOT / "commands.sh"),
            "stages": [s["stage"] for s in STAGES],
        },
        "stages": stages_out,
        "best": {
            "HOTA": {
                "stage": best_hota["stage"],
                "value": best_hota["metrics"]["HOTA"],
            },
            "MOTA": {
                "stage": best_mota["stage"],
                "value": best_mota["metrics"]["MOTA"],
            },
            "IDF1": {
                "stage": best_idf1["stage"],
                "value": best_idf1["metrics"]["IDF1"],
            },
        },
    }

    json_path = ROOT / "ablation_results.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    metric_names = list(stages_out[0]["metrics"].keys())
    report_lines: list[str] = []
    report_lines.append("# MOT17-04 消融实验报告（ByteTrack）")
    report_lines.append("")
    report_lines.append(f"- 生成时间: {result['experiment']['date']}")
    report_lines.append("- 数据集: MOT17-04")
    report_lines.append("- 评测阶段: S0 -> S1(STBG) -> S2(+PZM) -> S3(+Zone)")
    report_lines.append(f"- 命令脚本: `{ROOT / 'commands.sh'}`")
    report_lines.append("")

    report_lines.append("## 1. 阶段配置与功能开关")
    report_lines.append("")
    report_lines.append(
        "| Stage | STBG | PZM | Zone | min_conf | track_thresh | new_track_thresh | match_thresh | birth_confirm_frames | birth_suppress_iou | birth_suppress_center_dist | zombie_max_history | zombie_dist_thresh | zombie_max_predict_frames | entry_margin | exit_zone_enabled | adaptive_zone_enabled | adaptive_zone_update_mode |"
    )
    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"
    )
    for st in stages_out:
        p = st["tracker_params"]
        f = st["features"]
        report_lines.append(
            "| {stage} | {stbg} | {pzm} | {zone} | {min_conf} | {track_thresh} | {new_track_thresh} | {match_thresh} | {birth_confirm_frames} | {birth_suppress_iou} | {birth_suppress_center_dist} | {zombie_max_history} | {zombie_dist_thresh} | {zombie_max_predict_frames} | {entry_margin} | {exit_zone_enabled} | {adaptive_zone_enabled} | {adaptive_zone_update_mode} |".format(
                stage=st["stage"],
                stbg="on" if f["STBG"] else "off",
                pzm="on" if f["PZM"] else "off",
                zone="on" if f["Zone"] else "off",
                min_conf=fmt(p["min_conf"]),
                track_thresh=fmt(p["track_thresh"]),
                new_track_thresh=fmt(p["new_track_thresh"]),
                match_thresh=fmt(p["match_thresh"]),
                birth_confirm_frames=fmt(p["birth_confirm_frames"]),
                birth_suppress_iou=fmt(p["birth_suppress_iou"]),
                birth_suppress_center_dist=fmt(p["birth_suppress_center_dist"]),
                zombie_max_history=fmt(p["zombie_max_history"]),
                zombie_dist_thresh=fmt(p["zombie_dist_thresh"]),
                zombie_max_predict_frames=fmt(p["zombie_max_predict_frames"]),
                entry_margin=fmt(p["entry_margin"]),
                exit_zone_enabled=fmt(p["exit_zone_enabled"]),
                adaptive_zone_enabled=fmt(p["adaptive_zone_enabled"]),
                adaptive_zone_update_mode=fmt(p["adaptive_zone_update_mode"]),
            )
        )
    report_lines.append("")

    report_lines.append("## 2. 核心指标对比")
    report_lines.append("")
    report_lines.append("| Stage | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag | CLR_FP | CLR_FN | IDs | MTR | MLR |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for st in stages_out:
        m = st["metrics"]
        report_lines.append(
            "| {stage} | {HOTA:.3f} | {MOTA:.3f} | {IDF1:.3f} | {DetA:.3f} | {AssA:.3f} | {IDSW} | {Frag} | {CLR_FP} | {CLR_FN} | {IDs} | {MTR:.4f} | {MLR:.4f} |".format(
                stage=st["stage"],
                HOTA=float(m["HOTA"]),
                MOTA=float(m["MOTA"]),
                IDF1=float(m["IDF1"]),
                DetA=float(m["DetA"]),
                AssA=float(m["AssA"]),
                IDSW=int(m["IDSW"]),
                Frag=int(m["Frag"]),
                CLR_FP=int(m["CLR_FP"]),
                CLR_FN=int(m["CLR_FN"]),
                IDs=int(m["IDs"]),
                MTR=float(m["MTR"]),
                MLR=float(m["MLR"]),
            )
        )
    report_lines.append("")

    report_lines.append("## 3. 相对增益（相对 S0 Baseline）")
    report_lines.append("")
    report_lines.append("| Stage | ΔHOTA | ΔMOTA | ΔIDF1 | ΔDetA | ΔAssA | ΔIDSW | ΔFrag | ΔCLR_FP | ΔCLR_FN | ΔIDs |")
    report_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for st in stages_out:
        d = st["delta_vs_baseline"]
        report_lines.append(
            "| {stage} | {HOTA:+.3f} | {MOTA:+.3f} | {IDF1:+.3f} | {DetA:+.3f} | {AssA:+.3f} | {IDSW:+.0f} | {Frag:+.0f} | {CLR_FP:+.0f} | {CLR_FN:+.0f} | {IDs:+.0f} |".format(
                stage=st["stage"],
                HOTA=float(d["HOTA"]),
                MOTA=float(d["MOTA"]),
                IDF1=float(d["IDF1"]),
                DetA=float(d["DetA"]),
                AssA=float(d["AssA"]),
                IDSW=float(d["IDSW"]),
                Frag=float(d["Frag"]),
                CLR_FP=float(d["CLR_FP"]),
                CLR_FN=float(d["CLR_FN"]),
                IDs=float(d["IDs"]),
            )
        )
    report_lines.append("")

    report_lines.append("## 4. 全量指标矩阵")
    report_lines.append("")
    report_lines.append("以下包含 `pedestrian_summary.txt` 的全部指标列。")
    report_lines.append("")
    report_lines.append("```csv")
    report_lines.append("Stage," + ",".join(metric_names))
    for st in stages_out:
        row = [st["stage"]] + [str(st["metrics"][k]) for k in metric_names]
        report_lines.append(",".join(row))
    report_lines.append("```")
    report_lines.append("")

    report_lines.append("## 5. 最优结果")
    report_lines.append("")
    report_lines.append(
        f"- HOTA 最优: {result['best']['HOTA']['stage']} ({float(result['best']['HOTA']['value']):.3f})"
    )
    report_lines.append(
        f"- MOTA 最优: {result['best']['MOTA']['stage']} ({float(result['best']['MOTA']['value']):.3f})"
    )
    report_lines.append(
        f"- IDF1 最优: {result['best']['IDF1']['stage']} ({float(result['best']['IDF1']['value']):.3f})"
    )
    report_lines.append("")

    report_lines.append("## 6. 文件索引")
    report_lines.append("")
    for st in stages_out:
        report_lines.append(f"- {st['stage']} 配置: `{st['config_path']}`")
        report_lines.append(f"- {st['stage']} 日志: `{st['log_path']}`")
        report_lines.append(f"- {st['stage']} 输出目录: `{st['output_dir']}`")
    report_lines.append(f"- 汇总 JSON: `{json_path}`")

    md_path = ROOT / "Ablation_Report_MOT17-04.md"
    md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate()
