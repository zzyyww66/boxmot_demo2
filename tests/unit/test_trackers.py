from pathlib import Path

import numpy as np
import pytest

from boxmot import (
    DeepOcSort,
    OcSort,
    create_tracker,
    get_tracker_config,
)
from boxmot.trackers.deepocsort.deepocsort import (
    KalmanBoxTracker as DeepOCSortKalmanBoxTracker,
)
from boxmot.trackers.bytetrack.bytetrack import ByteTrack
from boxmot.trackers.ocsort.ocsort import KalmanBoxTracker as OCSortKalmanBoxTracker
from boxmot.utils import WEIGHTS
from tests.test_config import (
    ALL_TRACKERS,
    MOTION_N_APPEARANCE_TRACKING_METHODS,
    MOTION_N_APPEARANCE_TRACKING_NAMES,
    MOTION_ONLY_TRACKING_METHODS,
    PER_CLASS_TRACKERS,
)

# --- existing tests ---


@pytest.mark.parametrize("Tracker", MOTION_N_APPEARANCE_TRACKING_METHODS)
def test_motion_n_appearance_trackers_instantiation(Tracker):
    Tracker(
        reid_weights=Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
        device="cpu",
        half=True,
    )


@pytest.mark.parametrize("Tracker", MOTION_ONLY_TRACKING_METHODS)
def test_motion_only_trackers_instantiation(Tracker):
    Tracker()


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_tracker_output_size(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([[144, 212, 400, 480, 0.82, 0], [425, 281, 576, 472, 0.72, 65]])

    output = tracker.update(det, rgb)
    assert output.shape == (2, 8)


def test_dynamic_max_obs_based_on_max_age():
    max_age = 400
    ocsort = OcSort(max_age=max_age)
    assert ocsort.max_obs == (max_age + 5)


def create_kalman_box_tracker_ocsort(bbox, cls, det_ind, tracker):
    return OCSortKalmanBoxTracker(
        bbox,
        cls,
        det_ind,
        Q_xy_scaling=tracker.Q_xy_scaling,
        Q_s_scaling=tracker.Q_s_scaling,
    )


def create_kalman_box_tracker_deepocsort(bbox, cls, det_ind, tracker):
    det = np.concatenate([bbox, [cls, det_ind]])
    return DeepOCSortKalmanBoxTracker(
        det, Q_xy_scaling=tracker.Q_xy_scaling, Q_s_scaling=tracker.Q_s_scaling
    )


TRACKER_CREATORS = {
    OcSort: create_kalman_box_tracker_ocsort,
    DeepOcSort: create_kalman_box_tracker_deepocsort,
}


@pytest.mark.parametrize(
    "Tracker, init_args",
    [
        (OcSort, {}),
        (
            DeepOcSort,
            {
                "reid_weights": Path(WEIGHTS / "osnet_x0_25_msmt17.pt"),
                "device": "cpu",
                "half": True,
            },
        ),
    ],
)
def test_Q_matrix_scaling(Tracker, init_args):
    bbox = np.array([0, 0, 100, 100, 0.9])
    cls = 1
    det_ind = 0
    Q_xy_scaling = 0.05
    Q_s_scaling = 0.0005

    tracker = Tracker(Q_xy_scaling=Q_xy_scaling, Q_s_scaling=Q_s_scaling, **init_args)

    create_kalman_box_tracker = TRACKER_CREATORS[Tracker]
    kalman_box_tracker = create_kalman_box_tracker(bbox, cls, det_ind, tracker)

    assert kalman_box_tracker.kf.Q[4, 4] == Q_xy_scaling, "Q_xy scaling incorrect for x' velocity"
    assert kalman_box_tracker.kf.Q[5, 5] == Q_xy_scaling, "Q_xy scaling incorrect for y' velocity"
    assert kalman_box_tracker.kf.Q[6, 6] == Q_s_scaling, "Q_s scaling incorrect for s' (scale) velocity"

@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_output_size(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=True,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([
        [100, 100, 300, 250, 0.95,   0],  # class 0
        [400, 300, 550, 450, 0.90,  65],  # class 65
    ])
    embs = np.random.random(size=(2, 512))

    _ = tracker.update(det, rgb, embs)
    output = tracker.update(det, rgb, embs)
    assert output.shape == (2, 8)


@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_tracker_active_tracks(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=True,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    det = np.array([
        [100, 100, 300, 250, 0.95,   0],  # class 0
        [400, 300, 550, 450, 0.90,  65],  # class 65
    ])
    embs = np.random.random(size=(2, 512))

    tracker.update(det, rgb, embs)
    assert tracker.per_class_active_tracks[0], "No active tracks for class 0"
    assert tracker.per_class_active_tracks[65], "No active tracks for class 65"


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
@pytest.mark.parametrize("dets", [None, np.array([])])
def test_tracker_with_no_detections(tracker_type, dets):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    rgb = np.random.randint(255, size=(640, 640, 3), dtype=np.uint8)
    embs = np.random.random(size=(0, 512))

    output = tracker.update(dets, rgb, embs)
    assert output.size == 0, "Output should be empty when no detections are provided"


@pytest.mark.parametrize("tracker_type", PER_CLASS_TRACKERS)
def test_per_class_isolation(tracker_type):
    tracker = create_tracker(
        tracker_type,
        get_tracker_config(tracker_type),
        WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=True,
    )
    det = np.array(
        [
            [100, 100, 150, 150, 0.9, 1],
            [102, 102, 152, 152, 0.9, 2],
        ]
    )
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    embs = np.random.rand(2, 512)
    out = tracker.update(det, rgb, embs)
    ids = set(out[:, 1].tolist())
    assert len(ids) == 2, "Each class should get a separate track even if overlapping"


@pytest.mark.parametrize("tracker_type", MOTION_N_APPEARANCE_TRACKING_NAMES)
def test_emb_trackers_requires_embeddings(tracker_type):
    tracker_conf = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=tracker_conf,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )
    det = np.array([[10, 10, 20, 20, 0.7, 0]])
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        tracker.update(det, rgb, np.random.rand(2, 512))


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_invalid_det_array_shape(tracker_type):
    tracker = create_tracker(
        tracker_type,
        get_tracker_config(tracker_type),
        WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    embs = np.random.rand(2, 512)
    bad_det = np.random.rand(2, 5)
    with pytest.raises(AssertionError):
        tracker.update(bad_det, img, embs)


# def test_get_tracker_config_invalid_name():
#     """Requesting config for an unknown tracker should raise a KeyError."""
#     with pytest.raises(KeyError):
#         get_tracker_config("not_a_tracker")


@pytest.mark.parametrize("tracker_type", ALL_TRACKERS)
def test_track_id_stable_over_frames(tracker_type):
    """
    If the same detection appears in successive frames,
    the tracker should assign the same track ID.
    """
    cfg = get_tracker_config(tracker_type)
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=cfg,
        reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
        device="cpu",
        half=False,
        per_class=False,
    )

    det = np.array([[50, 50, 100, 100, 0.95, 3]])
    rgb = np.zeros((640, 640, 3), dtype=np.uint8)

    # choose embedding only if needed
    if tracker_type in MOTION_N_APPEARANCE_TRACKING_NAMES:
        embs = np.random.rand(1, 512)
        out1 = tracker.update(det, rgb, embs)
        out2 = tracker.update(det, rgb, embs)
    else:
        out1 = tracker.update(det, rgb)
        out2 = tracker.update(det, rgb)

    assert out1.shape == out2.shape == (1, 8), "Unexpected output shape"
    # track ID is at column 1
    assert out1[0, 4] == out2[0, 4], "Track ID should remain the same across frames"


def test_create_tracker_invalid_tracker_name():
    """Creating a tracker with an unknown name should raise a ValueError."""
    with pytest.raises(ValueError, match="Unknown tracker type: 'nonexistent_tracker'"):
        create_tracker(
            tracker_type="nonexistent_tracker",
            tracker_config=get_tracker_config("botsort"),
            reid_weights=WEIGHTS / "mobilenetv2_x1_4_dukemtmcreid.pt",
            device="cpu",
            half=False,
            per_class=False,
        )


def test_bytetrack_adaptive_zone_always_expand_grows_monotonically():
    tracker = ByteTrack(
        entry_margin=50,
        strict_entry_gate=True,
        adaptive_zone_enabled=True,
        adaptive_zone_update_mode="always_expand",
        adaptive_zone_expand_trigger="all_high",
        adaptive_zone_padding=1.0,
        adaptive_zone_min_box_area=0,
        zombie_max_history=0,
    )
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    det1 = np.array([[450, 450, 500, 550, 0.95, 0]], dtype=np.float32)
    tracker.update(det1, img)
    assert tracker._effective_zone is not None
    z1 = tracker._effective_zone.copy()

    det2 = np.array([[700, 450, 750, 550, 0.95, 0]], dtype=np.float32)
    tracker.update(det2, img)
    z2 = tracker._effective_zone.copy()

    assert z2[0] <= z1[0]
    assert z2[1] <= z1[1]
    assert z2[2] >= z1[2]
    assert z2[3] >= z1[3]
    assert z2[2] >= 750


def test_bytetrack_outside_before_expand_keeps_new_id_creation():
    tracker = ByteTrack(
        entry_margin=50,
        strict_entry_gate=True,
        adaptive_zone_enabled=True,
        adaptive_zone_update_mode="always_expand",
        adaptive_zone_expand_trigger="all_high",
        adaptive_zone_padding=1.0,
        adaptive_zone_min_box_area=0,
        zombie_max_history=0,
    )
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Frame 1 initializes effective zone around center.
    out1 = tracker.update(np.array([[450, 450, 500, 550, 0.95, 0]], dtype=np.float32), img)
    assert out1.shape[0] == 1
    id1 = int(out1[0, 4])

    # Frame 2 high-conf detection was outside previous effective zone.
    # In always_expand+all_high mode the zone expands before step4; the new
    # outside-before-expand marker ensures this detection gets activated.
    out2 = tracker.update(np.array([[700, 450, 750, 550, 0.95, 0]], dtype=np.float32), img)
    assert out2.shape[0] == 0  # non-first-frame activation is unconfirmed until next frame
    assert tracker._outside_zone_det_inds

    # Frame 3 repeats the same detection to confirm the new track and expose ID in output.
    out3 = tracker.update(np.array([[700, 450, 750, 550, 0.95, 0]], dtype=np.float32), img)
    assert out3.shape[0] == 1
    id3 = int(out3[0, 4])
    assert id3 != id1


def test_bytetrack_entry_zone_birth_skips_pending_confirmation():
    tracker = ByteTrack(
        new_track_thresh=0.6,
        birth_confirm_frames=2,
        entry_margin=50,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 6), dtype=np.float32)
    det = np.array([[5, 120, 65, 260, 0.9, 0]], dtype=np.float32)

    tracker.update(empty, img)
    out2 = tracker.update(det, img)
    assert out2.shape[0] == 0
    assert len(tracker.pending_births) == 0
    assert len(tracker.active_tracks) == 1

    out3 = tracker.update(det, img)
    assert out3.shape[0] == 1


def test_bytetrack_exit_zone_margin_zero_is_disabled():
    tracker = ByteTrack(
        exit_zone_enabled=True,
        exit_zone_margin=0,
        zombie_max_history=0,
    )
    assert tracker._is_in_exit_zone(np.array([100, 100, 50, 80], dtype=np.float32), (1000, 1000)) is False


def test_bytetrack_exit_zone_remove_grace_delays_removal():
    tracker = ByteTrack(
        entry_margin=0,
        strict_entry_gate=False,
        exit_zone_enabled=True,
        exit_zone_margin=50,
        exit_zone_remove_grace=3,
        zombie_max_history=0,
    )
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    det = np.array([[5, 100, 55, 220, 0.95, 0]], dtype=np.float32)
    empty = np.empty((0, 6), dtype=np.float32)

    out1 = tracker.update(det, img)
    assert out1.shape[0] == 1

    tracker.update(empty, img)
    assert len(tracker.lost_stracks) == 1
    assert len(tracker.removed_stracks) == 0
    assert bool(getattr(tracker.lost_stracks[0], "exit_pending", False))

    tracker.update(empty, img)
    tracker.update(empty, img)
    tracker.update(empty, img)
    assert len(tracker.removed_stracks) >= 1


def test_bytetrack_new_track_thresh_blocks_low_conf_birth():
    tracker = ByteTrack(
        track_thresh=0.45,
        new_track_thresh=0.8,
        entry_margin=0,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    low_birth = np.array([[100, 100, 180, 260, 0.7, 0]], dtype=np.float32)
    high_birth = np.array([[100, 100, 180, 260, 0.9, 0]], dtype=np.float32)

    out1 = tracker.update(low_birth, img)
    assert out1.shape[0] == 0
    assert len(tracker.active_tracks) == 0

    out2 = tracker.update(high_birth, img)
    assert out2.shape[0] == 0

    out3 = tracker.update(high_birth, img)
    assert out3.shape[0] == 1


def test_bytetrack_birth_confirm_requires_two_hits():
    tracker = ByteTrack(
        new_track_thresh=0.6,
        birth_confirm_frames=2,
        entry_margin=50,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 6), dtype=np.float32)
    det = np.array([[200, 120, 260, 260, 0.9, 0]], dtype=np.float32)

    tracker.update(empty, img)
    out2 = tracker.update(det, img)
    assert out2.shape[0] == 0
    assert len(tracker.pending_births) == 1
    assert len(tracker.active_tracks) == 0

    out3 = tracker.update(det, img)
    assert out3.shape[0] == 0
    assert len(tracker.pending_births) == 0
    assert len(tracker.active_tracks) == 1

    out4 = tracker.update(det, img)
    assert out4.shape[0] == 1


def test_bytetrack_birth_duplicate_suppression_blocks_double_id():
    tracker = ByteTrack(
        entry_margin=0,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        birth_confirm_frames=1,
        birth_suppress_iou=0.7,
        birth_suppress_center_dist=0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    det1 = np.array([[100, 100, 160, 240, 0.95, 0]], dtype=np.float32)
    det_dup = np.array(
        [
            [100, 100, 160, 240, 0.95, 0],
            [102, 102, 162, 242, 0.93, 0],
        ],
        dtype=np.float32,
    )

    out1 = tracker.update(det1, img)
    assert out1.shape[0] == 1

    out2 = tracker.update(det_dup, img)
    assert out2.shape[0] == 1

    out3 = tracker.update(det_dup, img)
    assert out3.shape[0] == 1
    assert len(tracker.active_tracks) == 1


def test_bytetrack_pending_birth_expires_without_consecutive_confirmation():
    tracker = ByteTrack(
        new_track_thresh=0.6,
        birth_confirm_frames=2,
        entry_margin=0,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 6), dtype=np.float32)
    det = np.array([[260, 140, 320, 280, 0.92, 0]], dtype=np.float32)

    tracker.update(empty, img)
    tracker.update(det, img)
    assert len(tracker.pending_births) == 1

    tracker.update(empty, img)
    tracker.update(det, img)
    assert len(tracker.pending_births) == 1
    assert len(tracker.active_tracks) == 0


def test_bytetrack_spatial_prior_commits_stable_birth_and_support():
    tracker = ByteTrack(
        entry_margin=0,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_birth_commit_age=3,
        spatial_prior_birth_commit_hits=3,
        spatial_prior_support_min_age=2,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    det = np.array([[220, 120, 280, 280, 0.95, 0]], dtype=np.float32)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    tracker.update(det, img)
    tracker.update(det, img)
    tracker.update(det, img)

    stats = tracker.spatial_prior.stats()
    assert stats.support_total >= 1.9
    assert stats.birth_total >= 0.9
    assert tracker.active_tracks[0].spatial_birth_committed is True


def test_bytetrack_spatial_prior_skips_first_frame_births():
    tracker = ByteTrack(
        entry_margin=0,
        strict_entry_gate=False,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_birth_commit_age=2,
        spatial_prior_birth_commit_hits=2,
        spatial_prior_support_min_age=2,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    dets = np.array(
        [
            [120, 120, 180, 260, 0.95, 0],
            [260, 120, 320, 260, 0.95, 0],
        ],
        dtype=np.float32,
    )

    tracker.update(dets, img)
    tracker.update(dets, img)
    tracker.update(dets, img)

    stats = tracker.spatial_prior.stats()
    assert tracker.spatial_prior_birth_events == 0
    assert stats.birth_total == 0.0
    assert stats.support_total > 0.0


def test_bytetrack_spatial_entry_region_uses_walkable_outer_band():
    tracker = ByteTrack(
        entry_margin=50,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_grid_w=12,
        spatial_prior_grid_h=8,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.1,
        spatial_prior_region_birth=0.5,
        spatial_prior_entry_band_radius=1,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    tracker.spatial_prior.support_count.fill(0.0)
    tracker.spatial_prior.birth_count.fill(0.0)
    tracker.spatial_prior.support_count[1:7, 2:10] = 10.0
    tracker.spatial_prior.birth_count[4, 5] = 12.0
    tracker.spatial_prior.birth_count[1, 5] = 15.0
    tracker.spatial_prior_support_samples = 100
    tracker.spatial_prior_birth_events = 10

    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    def grid_point(ix: int, iy: int) -> np.ndarray:
        x = ix / float(tracker.spatial_prior.grid_w - 1) * float(img.shape[1] - 1)
        y = iy / float(tracker.spatial_prior.grid_h - 1) * float(img.shape[0] - 1)
        return np.array([x, y], dtype=np.float32)

    center_point = grid_point(5, 4)
    edge_point = grid_point(5, 1)
    assert tracker._get_spatial_region_label(center_point) == "core"
    assert tracker._get_spatial_region_label(edge_point) == "entry"


def test_bytetrack_spatial_entry_region_grows_along_connected_band():
    tracker = ByteTrack(
        entry_margin=50,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_grid_w=12,
        spatial_prior_grid_h=8,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.1,
        spatial_prior_region_birth=0.8,
        spatial_prior_region_birth_grow=0.3,
        spatial_prior_region_grow_max_steps=0,
        spatial_prior_region_component_mean_ratio=0.0,
        spatial_prior_entry_band_radius=1,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    tracker.spatial_prior.support_count.fill(0.0)
    tracker.spatial_prior.birth_count.fill(0.0)
    tracker.spatial_prior.support_count[1:7, 2:10] = 10.0
    tracker.spatial_prior.birth_count[1, 4] = 2.0
    tracker.spatial_prior.birth_count[1, 5] = 6.0
    tracker.spatial_prior.birth_count[1, 6] = 2.0
    tracker.spatial_prior.birth_count[6, 9] = 3.0
    tracker.spatial_prior_support_samples = 100
    tracker.spatial_prior_birth_events = 10

    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    def grid_point(ix: int, iy: int) -> np.ndarray:
        x = ix / float(tracker.spatial_prior.grid_w - 1) * float(img.shape[1] - 1)
        y = iy / float(tracker.spatial_prior.grid_h - 1) * float(img.shape[0] - 1)
        return np.array([x, y], dtype=np.float32)

    assert tracker._get_spatial_region_label(grid_point(4, 1)) == "entry"
    assert tracker._get_spatial_region_label(grid_point(5, 1)) == "entry"
    assert tracker._get_spatial_region_label(grid_point(6, 1)) == "entry"
    assert tracker._get_spatial_region_label(grid_point(9, 6)) == "core"


def test_bytetrack_spatial_entry_region_limits_long_tail_growth():
    tracker = ByteTrack(
        entry_margin=50,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_grid_w=12,
        spatial_prior_grid_h=8,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.1,
        spatial_prior_region_birth=0.8,
        spatial_prior_region_birth_grow=0.0,
        spatial_prior_region_grow_max_steps=1,
        spatial_prior_region_component_mean_ratio=0.0,
        spatial_prior_entry_band_radius=1,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    tracker.spatial_prior.support_count.fill(0.0)
    tracker.spatial_prior.birth_count.fill(0.0)
    tracker.spatial_prior.support_count[1:7, 2:10] = 10.0
    tracker.spatial_prior.birth_count[1, 4] = 3.0
    tracker.spatial_prior.birth_count[1, 5] = 7.0
    tracker.spatial_prior.birth_count[1, 6] = 3.0
    tracker.spatial_prior.birth_count[1, 7] = 1.2
    tracker.spatial_prior.birth_count[1, 8] = 1.0
    tracker.spatial_prior.birth_count[1, 9] = 1.0
    tracker.spatial_prior_support_samples = 100
    tracker.spatial_prior_birth_events = 10

    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    def grid_point(ix: int, iy: int) -> np.ndarray:
        x = ix / float(tracker.spatial_prior.grid_w - 1) * float(img.shape[1] - 1)
        y = iy / float(tracker.spatial_prior.grid_h - 1) * float(img.shape[0] - 1)
        return np.array([x, y], dtype=np.float32)

    assert tracker._get_spatial_region_label(grid_point(7, 1)) == "entry"
    assert tracker._get_spatial_region_label(grid_point(8, 1)) == "core"
    assert tracker._get_spatial_region_label(grid_point(9, 1)) == "core"


def test_bytetrack_spatial_entry_region_allows_center_birth_override():
    tracker = ByteTrack(
        entry_margin=50,
        strict_entry_gate=True,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.0,
        spatial_prior_region_birth=0.0,
        spatial_prior_entry_band_radius=0,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 6), dtype=np.float32)
    det = np.array([[220, 120, 280, 280, 0.95, 0]], dtype=np.float32)
    point = np.array([250.0, 280.0], dtype=np.float32)

    tracker.update(empty, img)
    for _ in range(10):
        tracker._spatial_prior_add_support(point)
        tracker._spatial_prior_add_birth(point)
    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    tracker.update(det, img)
    assert len(tracker.active_tracks) == 1


def test_bytetrack_spatial_entry_region_blocks_center_birth_outside_hotspot():
    tracker = ByteTrack(
        entry_margin=50,
        strict_entry_gate=True,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.0,
        spatial_prior_region_birth=0.0,
        spatial_prior_entry_band_radius=0,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    empty = np.empty((0, 6), dtype=np.float32)
    det = np.array([[220, 120, 280, 280, 0.95, 0]], dtype=np.float32)
    point = np.array([250.0, 280.0], dtype=np.float32)
    birth_point = np.array([20.0, 280.0], dtype=np.float32)

    tracker.update(empty, img)
    for _ in range(10):
        tracker._spatial_prior_add_support(point)
    for _ in range(10):
        tracker._spatial_prior_add_support(birth_point)
        tracker._spatial_prior_add_birth(birth_point)
    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    tracker.update(det, img)
    assert len(tracker.active_tracks) == 0


def test_bytetrack_spatial_region_masks_create_center_entry_zone():
    tracker = ByteTrack(
        entry_margin=50,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.0,
        spatial_prior_entry_band_radius=0,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    point = np.array([320.0, 320.0], dtype=np.float32)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    for _ in range(10):
        tracker._spatial_prior_add_support(point)
        tracker._spatial_prior_add_birth(point)
    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    assert tracker._get_spatial_region_label(point) == "entry"
    assert tracker._is_in_entry_zone(np.array([290, 240, 60, 80], dtype=np.float32), img.shape[:2]) is True


def test_bytetrack_spatial_region_core_is_not_entry():
    tracker = ByteTrack(
        entry_margin=50,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_decay=1.0,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.0,
        spatial_prior_entry_band_radius=0,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    point = np.array([320.0, 320.0], dtype=np.float32)
    other_point = np.array([80.0, 320.0], dtype=np.float32)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    for _ in range(10):
        tracker._spatial_prior_add_support(point)
    tracker._spatial_prior_add_support(other_point)
    tracker._spatial_prior_add_birth(other_point)
    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    assert tracker._get_spatial_region_label(point) == "core"
    assert tracker._is_in_entry_zone(np.array([290, 240, 60, 80], dtype=np.float32), img.shape[:2]) is False


def test_bytetrack_spatial_prior_stage_uses_entry_thresholds_only():
    tracker = ByteTrack(
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_entry_support_threshold=3,
        spatial_prior_entry_birth_threshold=2,
    )

    assert tracker.spatial_prior_stage == "learn_only"
    tracker.spatial_prior_support_samples = 3
    tracker.spatial_prior_birth_events = 1
    tracker._update_spatial_prior_stage()
    assert tracker.spatial_prior_stage == "learn_only"

    tracker.spatial_prior_birth_events = 2
    tracker._update_spatial_prior_stage()
    assert tracker.spatial_prior_stage == "entry_only"


def test_bytetrack_exit_zone_stays_geometric_when_entry_region_is_enabled():
    tracker = ByteTrack(
        entry_margin=50,
        exit_zone_enabled=True,
        exit_zone_margin=40,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_entry_support_threshold=1,
        spatial_prior_entry_birth_threshold=1,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.0,
        spatial_prior_region_birth=0.0,
        spatial_prior_entry_band_radius=0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    center_point = np.array([320.0, 320.0], dtype=np.float32)

    tracker.update(np.empty((0, 6), dtype=np.float32), img)
    for _ in range(10):
        tracker._spatial_prior_add_support(center_point)
        tracker._spatial_prior_add_birth(center_point)
    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    assert tracker._get_spatial_region_label(center_point) == "entry"
    assert tracker._is_in_exit_zone(np.array([290, 240, 60, 80], dtype=np.float32), img.shape[:2]) is False
    assert tracker._is_in_exit_zone(np.array([10, 240, 60, 80], dtype=np.float32), img.shape[:2]) is True


def test_bytetrack_spatial_regions_fallback_to_rectangle_before_entry_stage():
    tracker = ByteTrack(
        entry_margin=50,
        strict_entry_gate=True,
        adaptive_zone_enabled=False,
        zombie_max_history=0,
        spatial_prior_enabled=True,
        spatial_prior_region_enabled=True,
        spatial_prior_entry_support_threshold=2,
        spatial_prior_entry_birth_threshold=2,
        spatial_prior_region_conf=0.1,
        spatial_prior_region_walk=0.0,
    )
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    center_det = np.array([[290, 240, 350, 320, 0.95, 0]], dtype=np.float32)
    empty = np.empty((0, 6), dtype=np.float32)

    tracker.update(empty, img)
    tracker._spatial_prior_add_support(np.array([320.0, 320.0], dtype=np.float32))
    tracker._spatial_prior_add_birth(np.array([320.0, 320.0], dtype=np.float32))
    tracker._update_spatial_prior_stage()
    tracker._refresh_spatial_region_masks()

    tracker.update(center_det, img)
    assert len(tracker.active_tracks) == 0
    assert tracker._spatial_region_ready() is False
