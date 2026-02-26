# MOT17-04 消融实验报告（ByteTrack）

- 生成时间: 2026-02-26 15:16:31
- 数据集: MOT17-04
- 评测阶段: S0 -> S1(STBG) -> S2(+PZM) -> S3(+Zone)
- 命令脚本: `MOT17-04测试/commands.sh`

## 1. 阶段配置与功能开关

| Stage | STBG | PZM | Zone | min_conf | track_thresh | new_track_thresh | match_thresh | birth_confirm_frames | birth_suppress_iou | birth_suppress_center_dist | zombie_max_history | zombie_dist_thresh | zombie_max_predict_frames | entry_margin | exit_zone_enabled | adaptive_zone_enabled | adaptive_zone_update_mode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| S0 | off | off | off | 0.100 | 0.500 | 0.500 | 0.800 | 1 | 0 | 0 | 0 | 999999 | 0 | 0 | false | false | warmup_once |
| S1 | on | off | off | 0.100 | 0.500 | 0.650 | 0.800 | 2 | 0.700 | 35 | 0 | 999999 | 0 | 0 | false | false | warmup_once |
| S2 | on | on | off | 0.100 | 0.500 | 0.650 | 0.800 | 2 | 0.700 | 35 | 100 | 150 | 5 | 0 | false | false | warmup_once |
| S3 | on | on | on | 0.100 | 0.500 | 0.650 | 0.800 | 2 | 0.700 | 35 | 100 | 150 | 5 | 50 | true | true | always_expand |

## 2. 核心指标对比

| Stage | HOTA | MOTA | IDF1 | DetA | AssA | IDSW | Frag | CLR_FP | CLR_FN | IDs | MTR | MLR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 81.102 | 93.744 | 90.469 | 82.301 | 80.046 | 47 | 154 | 1478 | 1450 | 117 | 93.9760 | 1.2048 |
| S1 | 79.425 | 92.722 | 88.594 | 81.461 | 77.558 | 25 | 106 | 1014 | 2422 | 95 | 90.3610 | 1.2048 |
| S2 | 79.755 | 91.682 | 89.466 | 80.563 | 79.048 | 19 | 95 | 486 | 3451 | 84 | 81.9280 | 7.2289 |
| S3 | 81.814 | 94.466 | 91.813 | 82.941 | 80.814 | 18 | 145 | 691 | 1923 | 88 | 87.9520 | 3.6145 |

## 3. 相对增益（相对 S0 Baseline）

| Stage | ΔHOTA | ΔMOTA | ΔIDF1 | ΔDetA | ΔAssA | ΔIDSW | ΔFrag | ΔCLR_FP | ΔCLR_FN | ΔIDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | +0 | +0 | +0 | +0 | +0 |
| S1 | -1.677 | -1.022 | -1.875 | -0.840 | -2.488 | -22 | -48 | -464 | +972 | -22 |
| S2 | -1.347 | -2.062 | -1.003 | -1.738 | -0.998 | -28 | -59 | -992 | +2001 | -33 |
| S3 | +0.712 | +0.722 | +1.344 | +0.640 | +0.768 | -29 | -9 | -787 | +473 | -29 |

## 4. 全量指标矩阵

以下包含 `pedestrian_summary.txt` 的全部指标列。

```csv
Stage,HOTA,DetA,AssA,DetRe,DetPr,AssRe,AssPr,LocA,OWTA,HOTA(0),LocA(0),HOTALocA(0),MOTA,MOTP,MODA,CLR_Re,CLR_Pr,MTR,PTR,MLR,CLR_TP,CLR_FN,CLR_FP,IDSW,MT,PT,ML,Frag,sMOTA,IDF1,IDR,IDP,IDTP,IDFN,IDFP,Dets,GT_Dets,IDs,GT_IDs
S0,81.102,82.301,80.046,88.017,87.965,84.582,88.131,89.994,83.936,91.37,88.502,80.864,93.744,89.235,93.843,96.951,96.894,93.976,4.8193,1.2048,46107,1450,1478,47,78,4,1,154,83.307,90.469,90.496,90.442,43037,4520,4548,47585,47557,117,83
S1,79.425,81.461,77.558,86.23,88.861,83.027,85.562,90.024,81.778,89.563,88.444,79.214,92.722,89.297,92.775,94.907,97.803,90.361,8.4337,1.2048,45135,2422,1014,25,75,7,1,106,82.565,88.594,87.283,89.946,41509,6048,4640,46149,47557,95,83
S2,79.755,80.563,79.048,84.331,89.939,83.829,86.778,90.161,81.644,89.769,88.786,79.702,91.682,89.437,91.722,92.743,98.91,81.928,10.843,7.2289,44106,3451,486,19,68,9,6,95,81.885,89.466,86.677,92.44,41221,6336,3371,44592,47557,84,83
S3,81.814,82.941,80.814,87.231,89.551,85.77,87.694,90.08,83.959,92.088,88.694,81.677,94.466,89.274,94.503,95.956,98.508,87.952,8.4337,3.6145,45634,1923,691,18,73,7,3,145,84.173,91.813,90.624,93.034,43098,4459,3227,46325,47557,88,83
```

## 5. 最优结果

- HOTA 最优: S3 (81.814)
- MOTA 最优: S3 (94.466)
- IDF1 最优: S3 (91.813)

## 6. 文件索引

- S0 配置: `MOT17-04测试/configs/stage0_baseline.yaml`
- S0 日志: `MOT17-04测试/logs/stage0_baseline.log`
- S0 输出目录: `MOT17-04测试/runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack`
- S1 配置: `MOT17-04测试/configs/stage1_stbg.yaml`
- S1 日志: `MOT17-04测试/logs/stage1_stbg.log`
- S1 输出目录: `MOT17-04测试/runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_2`
- S2 配置: `MOT17-04测试/configs/stage2_stbg_pzm.yaml`
- S2 日志: `MOT17-04测试/logs/stage2_stbg_pzm.log`
- S2 输出目录: `MOT17-04测试/runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_3`
- S3 配置: `MOT17-04测试/configs/stage3_stbg_pzm_zone.yaml`
- S3 日志: `MOT17-04测试/logs/stage3_stbg_pzm_zone.log`
- S3 输出目录: `MOT17-04测试/runs/mot/yolox_x_MOT17_ablation_lmbn_n_duke_bytetrack_4`
- 汇总 JSON: `MOT17-04测试/ablation_results.json`
