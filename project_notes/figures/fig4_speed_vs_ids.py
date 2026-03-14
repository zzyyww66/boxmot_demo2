#!/usr/bin/env python3
"""Plot IDSW vs FPS for SOMPT22."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter


def main() -> None:
    # ==========================================
    # 1. 数据准备 (SOMPT22)
    # ==========================================
    # 类别: 0=Appearance(带ReID,慢), 1=Motion(纯运动,快), 2=Ours
    trackers = ["StrongSORT", "BotSORT", "OC-SORT", "ByteTrack", "ZLM-Track (Ours)"]
    fps = [14.0, 68.0, 196.0, 223.0, 220.0]
    idsw = [1518, 910, 1055, 1250, 917]
    hota = [53.69, 53.79, 51.13, 53.31, 51.84]
    categories = [0, 0, 1, 1, 2]

    # 顶会高级配色方案
    colors = {
        0: "#6A4C93",  # 优雅紫 (Appearance-based)
        1: "#457B9D",  # 灰蓝色 (Motion-based)
        2: "#E63946",  # 砖红色 (Ours - 醒目但不刺眼)
    }

    # ==========================================
    # 2. 全局样式设置 (学术极简风)
    # ==========================================
    plt.rcParams["font.family"] = "serif"
    # 优先使用 Times New Roman，如果没有会自动回退到系统 serif
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["axes.linewidth"] = 1.2

    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=300)

    # 背景与网格：仅保留水平虚线网格，留白更显高级
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, axis="y", color="#E0E0E0", linestyle="--", linewidth=0.8, zorder=0)
    ax.grid(False, axis="x")

    # 去除顶部和右侧的边框 (Spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    # ==========================================
    # 3. 绘制散点数据
    # ==========================================
    for i in range(len(trackers)):
        cat = categories[i]
        # ZLM-Track 使用更大的五角星，对比算法使用圆点
        marker = "*" if cat == 2 else "o"
        size = 400 if cat == 2 else 180
        zorder = 4 if cat == 2 else 3

        ax.scatter(
            fps[i],
            idsw[i],
            s=size,
            c=colors[cat],
            edgecolors="white",
            linewidths=1.5,
            alpha=0.9,
            marker=marker,
            zorder=zorder,
        )

    # ==========================================
    # 4. 视觉引导线 (核心叙事逻辑)
    # ==========================================
    # 画一个从 ByteTrack 原版 指向 ZLM-Track 的弧线箭头
    idx_orig = trackers.index("ByteTrack")
    idx_ours = trackers.index("ZLM-Track (Ours)")

    ax.annotate(
        "",
        xy=(fps[idx_ours], idsw[idx_ours] + 25),
        xytext=(fps[idx_orig], idsw[idx_orig] - 25),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#888888",
            lw=1.5,
            shrinkA=5,
            shrinkB=5,
            connectionstyle="arc3,rad=-0.15",
        ),
        zorder=2,
    )

    # 在箭头旁边写上你的核心贡献
    ax.text(
        275,
        1095,
        "-26.6% IDSW\nMaintaining FPS",
        color="#555555",
        fontsize=10,
        fontstyle="italic",
        ha="left",
        va="center",
        zorder=2,
    )

    # 右下角仅保留说明文字，不使用箭头
    ax.text(
        520,
        730,
        "Ideal Operational Zone\n(Fast & Stable)",
        color="#A0A0A0",
        fontsize=9.5,
        fontstyle="italic",
        ha="right",
        va="center",
    )

    # ==========================================
    # 5. 优雅的文本标签标注 (无背景框，位置精调)
    # ==========================================
    # Use fixed label anchor positions in data coordinates to avoid overlaps.
    label_positions = {
        "StrongSORT": (4.1, 1565),
        "BotSORT": (70, 835),
        "OC-SORT": (180, 1140),
        "ByteTrack": (235, 1330),
        "ZLM-Track (Ours)": (245, 815),
    }
    alignments = {
        "StrongSORT": ("left", "top"),
        "BotSORT": ("center", "top"),
        "OC-SORT": ("right", "center"),
        "ByteTrack": ("center", "center"),
        "ZLM-Track (Ours)": ("center", "center"),
    }

    for i, txt in enumerate(trackers):
        label = f"{txt}\n[HOTA: {hota[i]:.1f}]"
        label_x, label_y = label_positions[txt]
        ha, va = alignments[txt]
        font_weight = "bold" if "Ours" in txt else "normal"
        color = "#B22222" if "Ours" in txt else "#222222"

        ax.annotate(
            label,
            (fps[i], idsw[i]),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=10,
            fontweight=font_weight,
            color=color,
            ha=ha,
            va=va,
            zorder=5,
        )

    # ==========================================
    # 6. 坐标轴格式化
    # ==========================================
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    # 设置优雅的 X 轴刻度
    ax.set_xticks([3, 10, 30, 100, 200, 400])

    ax.set_xlabel("Tracking Speed (FPS) $\\rightarrow$", fontweight="bold", labelpad=10)
    ax.set_ylabel("Identity Switches (IDSW, Lower is Better) $\\downarrow$", fontweight="bold", labelpad=10)

    ax.set_ylim(700, 1650)
    ax.set_xlim(2, 600)

    # ==========================================
    # 7. 添加极简图例 (Legend)
    # ==========================================
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Appearance-based (ReID)",
            markerfacecolor=colors[0],
            markersize=10,
        ),
        Line2D([0], [0], marker="o", color="w", label="Motion-only", markerfacecolor=colors[1], markersize=10),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            label="Ours (Prior-guided)",
            markerfacecolor=colors[2],
            markersize=15,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )

    # ==========================================
    # 8. 导出高清 PDF
    # ==========================================
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_dir = Path(__file__).resolve().parent
    output_pdf = out_dir / "fig4_idsw_vs_fps.pdf"
    output_png = out_dir / "fig4_idsw_vs_fps.png"
    plt.savefig(output_pdf, dpi=300, format="pdf", bbox_inches="tight")
    plt.savefig(output_png, dpi=300, format="png", bbox_inches="tight")
    print(f"Elegant plot saved successfully as: {output_pdf}")
    print(f"Elegant plot saved successfully as: {output_png}")
    plt.show()


if __name__ == "__main__":
    main()
