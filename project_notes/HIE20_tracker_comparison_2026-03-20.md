# HIE20 Tracker Comparison (2026-03-20)

## Setup

- Dataset: `HIE20/train`
- Detector: `yolov8m_pretrain_crowdhuman.pt`
- ReID weights: `osnet_x0_25_msmt17.pt`
- Evaluated trackers/configs: `bytetrack` (default), `bytetrack_sompt22_tuned`, `bytetrack_dual_tuned`, `botsort`, `deepocsort`, `hybridsort`, `boosttrack`, `strongsort`, `ocsort`
- Shared det/emb cache reused from `/root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617/dets_n_embs`
- Existing tracker outputs reused from `/root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617/mot`; after fixing GT, only TrackEval was rerun
- Additional ByteTrack config rows were appended from `/root/autodl-tmp/boxmot_demo2/runs_hie20_bytetrack_cfgs_20260320_025718`, which reused the same det/emb cache
- HIE20 GT repair: split overlapping raw track IDs into valid MOT IDs and removed near-identical duplicate boxes
- Post-fix validation: `0` duplicate GT IDs remain in any frame across all 19 sequences

## Overall Results

| Rank | Tracker | ReID | HOTA | MOTA | IDF1 | AssA | AssRe | IDSW | IDs | Assoc FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ByteTrack-SOMPT22Tuned | yes | 44.564 | 58.367 | 54.786 | 42.574 | 48.934 | 2705 | 1836 | 325.8 |
| 2 | ByteTrack-DualTuned | yes | 44.284 | 58.497 | 54.243 | 42.068 | 48.849 | 2771 | 1684 | 338.4 |
| 3 | botsort | yes | 44.251 | 58.612 | 53.446 | 41.826 | 47.306 | 2979 | 2297 | 118.3 |
| 4 | boosttrack | yes | 42.417 | 50.862 | 52.245 | 43.247 | 46.459 | 2814 | 3182 | 108.2 |
| 5 | deepocsort | yes | 42.115 | 53.300 | 51.531 | 41.225 | 44.883 | 3330 | 3230 | 100.0 |
| 6 | strongsort | yes | 41.360 | 52.192 | 49.960 | 40.569 | 45.034 | 4926 | 2400 | 11.5 |
| 7 | hybridsort | yes | 41.336 | 56.804 | 48.258 | 35.572 | 47.538 | 9065 | 1605 | 59.2 |
| 8 | bytetrack (default) | yes | 40.353 | 58.262 | 46.453 | 34.940 | 38.127 | 4619 | 4075 | 358.3 |
| 9 | ocsort | no | 39.828 | 48.743 | 48.099 | 40.150 | 43.693 | 2717 | 2767 | 400.3 |

## Metric Leaders

- Best HOTA: `ByteTrack-SOMPT22Tuned` (44.564)
- Best MOTA: `botsort` (58.612)
- Best IDF1: `ByteTrack-SOMPT22Tuned` (54.786)
- Best AssA: `boosttrack` (43.247)
- Best AssRe: `ByteTrack-SOMPT22Tuned` (48.934)
- Lowest IDSW: `ByteTrack-SOMPT22Tuned` (2705)
- Fastest association: `ocsort` (400.3 FPS)
- Fastest ReID-based tracker: `bytetrack (default)` (358.3 FPS)

## Observations

- `ByteTrack-SOMPT22Tuned` is now the best overall choice on HIE20: it ranks first in HOTA (44.564) and IDF1 (54.786), and it also has the lowest IDSW (2705) in this table.
- `ByteTrack-DualTuned` is a very close second on HIE20: it trails `ByteTrack-SOMPT22Tuned` by only `0.280` HOTA while keeping slightly higher MOTA (58.497), so it remains a strong cross-dataset compromise.
- `boosttrack` is the strongest association-focused ReID tracker after `botsort`: it ranks second in HOTA and IDF1, and it has the best AssA (43.247) among all trackers.
- `ocsort` is the fastest tracker by a large margin (400.3 FPS) and also has the lowest IDSW (2717), but its overall HOTA/MOTA remain below the best ReID-based methods.
- `botsort` still holds the best MOTA (58.612), but after adding the two tuned ByteTrack configs it is no longer the HOTA/IDF1 leader on HIE20.
- `bytetrack (default)` is still the fastest ReID-based option (358.3 FPS), but its HOTA (40.353) and IDF1 (46.453) are now clearly behind the two tuned ByteTrack variants.
- `hybridsort` has the best AssRe (47.538), but it also shows the highest IDSW (9065) in this comparison, which hurts its final ranking.
- `strongsort` is the slowest association stage by far (11.5 FPS), so it is likely the least practical option for larger HIE20 sweeps.

## Repro Notes

- Original full pipeline command per tracker:

```bash
uv run python -m boxmot.engine.cli eval \
  yolov8m_pretrain_crowdhuman \
  osnet_x0_25_msmt17 \
  <tracker> \
  --source HIE20 \
  --classes 0 \
  --device 0 \
  --project /root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617 \
  --exist-ok \
  --verbose
```

- After the first run exposed duplicate GT IDs, the GT was regenerated with `uv run python scripts/prepare_hie20_mot.py`.
- Detection and embedding caches were not recomputed.
- Tracker result txt files were also reused; only TrackEval was rerun against the repaired GT.

## Artifacts

- Run root: `/root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617`
- ByteTrack config study root: `/root/autodl-tmp/boxmot_demo2/runs_hie20_bytetrack_cfgs_20260320_025718`
- Logs: `/root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617/logs`
- Dets/embs cache: `/root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617/dets_n_embs`
- Tracker outputs and TrackEval summaries: `/root/autodl-tmp/boxmot_demo2/runs_hie20_tracker_compare_20260320_014617/mot`
