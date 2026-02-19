# Mikel Broström 🔥 BoxMOT 🧾 AGPL-3.0 license

from collections import deque

import numpy as np

from boxmot.motion.kalman_filters.aabb.xyah_kf import KalmanFilterXYAH
from boxmot.trackers.basetracker import BaseTracker
from boxmot.trackers.bytetrack.basetrack import BaseTrack, TrackState
from boxmot.utils.matching import fuse_score, iou_distance, linear_assignment
from boxmot.utils.ops import tlwh2xyah, xywh2tlwh, xywh2xyxy, xyxy2xywh


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, det, max_obs):
        # wait activate
        self.xywh = xyxy2xywh(det[0:4])  # (x1, y1, x2, y2) --> (xc, yc, w, h)
        self.tlwh = xywh2tlwh(self.xywh)  # (xc, yc, w, h) --> (t, l, w, h)
        self.xyah = tlwh2xyah(self.tlwh)
        self.conf = det[4]
        self.cls = det[5]
        self.det_ind = det[6]
        self.max_obs = max_obs
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.tracklet_len = 0
        self.history_observations = deque([], maxlen=self.max_obs)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.id = self.next_id()
        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.history_observations.append(self.xyxy)

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.conf = new_track.conf
        self.cls = new_track.cls
        self.det_ind = new_track.det_ind

    @property
    def xyxy(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        if self.mean is None:
            ret = self.xywh.copy()  # (xc, yc, w, h)
        else:
            ret = self.mean[:4].copy()  # kf (xc, yc, a, h)
            ret[2] *= ret[3]  # (xc, yc, a, h)  -->  (xc, yc, w, h)
        ret = xywh2xyxy(ret)
        return ret

    def get_tlwh_for_matching(self, frame_id: int = None, max_predict_frames: int = 5):
        """Get tlwh for matching, considering frozen state for zombie tracks.

        For zombie tracks that have exceeded max_predict_frames, return the frozen position.

        Args:
            frame_id: Current frame ID (for calculating frames since lost)
            max_predict_frames: Maximum frames to predict before freezing

        Returns:
            np.ndarray: [left, top, width, height] in tlwh format
        """
        # If frozen_mean exists and we've exceeded max prediction frames, use frozen position
        if (self.frozen_mean is not None and
            frame_id is not None and
            self.lost_frame_id > 0 and
            frame_id - self.lost_frame_id >= max_predict_frames):
            # Use frozen mean to compute tlwh
            ret = self.frozen_mean[:4].copy()
            ret[2] *= ret[3]  # (xc, yc, a, h) -> (xc, yc, w, h)
            return xywh2tlwh(ret)

        # Otherwise use current state (normal prediction)
        if self.mean is not None:
            ret = self.mean[:4].copy()
            ret[2] *= ret[3]  # (xc, yc, a, h) -> (xc, yc, w, h)
            return xywh2tlwh(ret)
        return self.tlwh


class ByteTrack(BaseTracker):
    """
    Initialize the ByteTrack tracker with various parameters.

    Parameters:
    - det_thresh (float): Detection threshold for considering detections.
    - max_age (int): Maximum age (in frames) of a track before it is considered lost.
    - max_obs (int): Maximum number of historical observations stored for each track. Always greater than max_age by minimum 5.
    - min_hits (int): Minimum number of detection hits before a track is considered confirmed.
    - iou_threshold (float): IOU threshold for determining match between detection and tracks.
    - per_class (bool): Enables class-separated tracking.
    - nr_classes (int): Total number of object classes that the tracker will handle (for per_class=True).
    - asso_func (str): Algorithm name used for data association between detections and tracks.
    - is_obb (bool): Work with Oriented Bounding Boxes (OBB) instead of standard axis-aligned bounding boxes.

    ByteTrack-specific parameters:
    - min_conf (float): Threshold for detection confidence. Detections below this threshold are discarded.
    - track_thresh (float): Threshold for detection confidence. Detections above this threshold are considered for tracking in the first association round.
    - match_thresh (float): Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
    - new_track_thresh (float): Threshold for creating a new track from unmatched detections (default: track_thresh).
    - track_buffer (int): Number of frames to keep a track alive after it was last detected.
    - frame_rate (int): Frame rate of the video being processed. Used to scale the track buffer size.
    - entry_margin (int): Pixel width of entry zone at frame edges. New IDs are only created within this margin (birth control). Set to 0 to disable (default: 0).
    - strict_entry_gate (bool): If True, unmatched center-zone detections do NOT create new IDs when entry gating is enabled (default: False).
    - birth_confirm_frames (int): Number of consecutive candidate hits required before activating a new ID (default: 1, i.e., disabled).
    - birth_suppress_iou (float): Suppress new births if IoU with existing tracks exceeds this threshold (<=0 disables).
    - birth_suppress_center_dist (float): Suppress new births if center distance to existing tracks is below this threshold in pixels (<=0 disables).

    Exit zone parameters:
    - exit_zone_enabled (bool): Enable exit zone feature. When a track disappears in the exit zone (frame edges), it enters exit-pending and can be removed after grace frames (default: False).
    - exit_zone_margin (int): Pixel width of exit zone at frame edges. Defaults to entry_margin if not specified (default: entry_margin).
    - exit_zone_remove_grace (int): Grace frames before removing an exit-pending lost track (default: 30).

    Adaptive effective zone parameters:
    - adaptive_zone_enabled (bool): Enable adaptive effective zone computation based on detection distribution (default: True).
    - adaptive_zone_warmup (int): Number of frames to collect detections for computing the effective zone (default: 10).
    - adaptive_zone_margin (int): Margin width within the effective zone to consider as entry zone (default: 50).
    - adaptive_zone_padding (float): Padding factor applied to the effective zone bounding box (default: 1.2).
    - adaptive_zone_update_mode (str): Effective-zone update strategy, "warmup_once" or "always_expand" (default: "always_expand").
    - adaptive_zone_expand_trigger (str): Expansion trigger in always_expand mode, "all_high", "outside_high", or "unmatched_high" (default: "all_high").
    - adaptive_zone_min_box_area (float): Ignore tiny boxes below this area when updating effective zone (default: 0).

    Attributes:
    - frame_count (int): Counter for the frames processed.
    - active_tracks (list): List to hold active tracks.
    - lost_stracks (list[STrack]): List of lost tracks.
    - removed_stracks (list[STrack]): List of removed tracks.
    - buffer_size (int): Size of the track buffer based on frame rate.
    - max_time_lost (int): Maximum time a track can be lost.
    - kalman_filter (KalmanFilterXYAH): Kalman filter for motion prediction.
    - _effective_zone (np.ndarray): [x1, y1, x2, y2] effective zone computed during warmup.
    """

    def __init__(
        self,
        # ByteTrack-specific parameters
        min_conf: float = 0.1,
        track_thresh: float = 0.45,
        match_thresh: float = 0.8,
        track_buffer: int = 25,
        frame_rate: int = 30,
        **kwargs  # BaseTracker parameters
    ):
        # Capture all init params for logging
        init_args = {k: v for k, v in locals().items() if k not in ('self', 'kwargs')}
        super().__init__(**init_args, _tracker_name='ByteTrack', **kwargs)
        
        # Track lifecycle parameters
        self.frame_id = 0
        self.track_buffer = track_buffer
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size

        # Detection thresholds
        self.min_conf = min_conf
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh  # Same as track_thresh
        self.new_track_thresh = kwargs.get('new_track_thresh', self.track_thresh)

        # Motion model
        self.kalman_filter = KalmanFilterXYAH()

        # Lifecycle gating configuration (scene-mask based birth control and zombie rescue)
        self.entry_margin = kwargs.get('entry_margin', 0)  # Entry zone pixel width (0 = disabled)
        self.strict_entry_gate = kwargs.get('strict_entry_gate', False)  # Center unmatched detections cannot spawn IDs when gating is on
        # Zombie track management parameters
        self.zombie_max_history = kwargs.get('zombie_max_history', 100)  # Max number of zombie tracks to keep
        self.zombie_dist_thresh = kwargs.get('zombie_dist_thresh', 150)  # Distance threshold for zombie rescue (pixels)
        self.zombie_max_predict_frames = kwargs.get('zombie_max_predict_frames', 5)  # Max frames to predict zombie position

        # Zombie track separation parameters
        self.zombie_transition_frames = kwargs.get('zombie_transition_frames', self.buffer_size)  # 转为僵尸的帧数阈值
        self.zombie_match_max_dist = kwargs.get('zombie_match_max_dist', 200)  # 僵尸匹配距离上限（像素）
        self.lost_max_history = kwargs.get('lost_max_history', 0)  # 0 means unlimited

        # Soft birth gating (optional, disabled by default for backwards compatibility)
        self.birth_confirm_frames = max(1, int(kwargs.get('birth_confirm_frames', 1)))
        self.birth_suppress_iou = float(kwargs.get('birth_suppress_iou', 0.0))
        self.birth_suppress_center_dist = float(kwargs.get('birth_suppress_center_dist', 0.0))
        self._birth_confirm_iou = 0.3
        self._birth_pending_max_miss = 1

        # Exit zone configuration
        self.exit_zone_enabled = kwargs.get('exit_zone_enabled', False)
        self.exit_zone_margin = kwargs.get('exit_zone_margin', self.entry_margin)  # Default to entry_margin
        self.exit_zone_remove_grace = kwargs.get('exit_zone_remove_grace', 30)

        # Adaptive effective zone configuration
        self.adaptive_zone_enabled = kwargs.get('adaptive_zone_enabled', True)  # Whether to enable adaptive zone
        self.adaptive_zone_warmup = kwargs.get('adaptive_zone_warmup', 10)  # Number of frames to collect detections
        self.adaptive_zone_margin = kwargs.get('adaptive_zone_margin', 50)  # Margin within effective zone for entry
        self.adaptive_zone_padding = kwargs.get('adaptive_zone_padding', 1.2)  # Padding factor for effective zone
        self.adaptive_zone_update_mode = kwargs.get('adaptive_zone_update_mode', 'always_expand')
        self.adaptive_zone_expand_trigger = kwargs.get('adaptive_zone_expand_trigger', 'all_high')
        self.adaptive_zone_min_box_area = kwargs.get('adaptive_zone_min_box_area', 0)

        # Runtime state for adaptive zone
        self._warmup_detections = []  # Cache of detection boxes during warmup
        self._effective_zone = None  # Effective zone [x1, y1, x2, y2] in xyxy format
        self._outside_zone_det_inds = set()  # Detection indices that were outside effective zone before expansion
        self._effective_zone_last_frame = -1

        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.zombie_stracks = []  # type: list[STrack]
        self.pending_births = []  # type: list[dict]
        self.removed_stracks = []  # type: list[STrack]

    def _zombie_enabled(self) -> bool:
        """Return whether zombie rescue/history should be active."""
        return (
            self.zombie_max_history > 0
            and self.zombie_transition_frames > 0
            and self.zombie_match_max_dist > 0
        )

    def _is_in_entry_zone(self, tlwh, img_shape, margin=None):
        """
        Check if a bounding box is in the entry zone.

        Supports both fixed margin mode (original behavior) and adaptive effective zone mode.

        Args:
            tlwh: [left, top, width, height]
            img_shape: (height, width)
            margin: Edge zone pixel width, defaults to self.entry_margin

        Returns:
            bool: True if the box is in the entry zone
        """
        if margin is None:
            margin = self.entry_margin

        # If entry_margin is 0, consider everything as entry zone (disabled gating)
        if margin <= 0:
            return True

        img_h, img_w = img_shape[0], img_shape[1]
        # tlwh format in this codebase: [left(x), top(y), width, height]
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h

        # Fixed margin mode (original behavior)
        if not self.adaptive_zone_enabled or self._effective_zone is None:
            # Consider entry if any side touches the edge zone
            if x1 < margin or y1 < margin:
                return True
            if x2 > (img_w - margin) or y2 > (img_h - margin):
                return True
            return False

        # Adaptive effective zone mode
        ex1, ey1, ex2, ey2 = self._effective_zone

        # If box is outside effective zone, it's in entry zone
        if x2 < ex1 or x1 > ex2 or y2 < ey1 or y1 > ey2:
            return True

        # If box is inside effective zone, check margin within effective zone
        margin = self.adaptive_zone_margin
        if x1 < ex1 + margin or y1 < ey1 + margin:
            return True
        if x2 > ex2 - margin or y2 > ey2 - margin:
            return True

        return False

    def _is_in_exit_zone(self, tlwh, img_shape):
        """Check whether a box is in the image-border exit zone.

        Exit-zone semantics intentionally do not depend on adaptive effective zone.
        A non-positive margin disables exit-zone triggering.
        """
        margin = self.exit_zone_margin
        if margin <= 0:
            return False

        img_h, img_w = img_shape[0], img_shape[1]
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        return bool(
            x1 < margin
            or y1 < margin
            or x2 > (img_w - margin)
            or y2 > (img_h - margin)
        )

    @staticmethod
    def _set_exit_pending(track, enabled: bool):
        track.exit_pending = bool(enabled)

    def _compute_effective_zone(self):
        """Compute effective zone from warmup detections.

        Returns:
            np.ndarray: [x1, y1, x2, y2] effective zone in xyxy coordinate system,
                       or None if no warmup detections available.
        """
        if len(self._warmup_detections) == 0:
            return None

        # Convert tlwh[x, y, w, h] to xyxy
        boxes = np.array(self._warmup_detections)
        # boxes[:, 0] = left (x), boxes[:, 1] = top (y), boxes[:, 2] = w, boxes[:, 3] = h
        x1 = boxes[:, 0]  # left
        y1 = boxes[:, 1]  # top
        x2 = boxes[:, 0] + boxes[:, 2]  # left + width
        y2 = boxes[:, 1] + boxes[:, 3]  # top + height

        # Compute bounding box
        min_x, max_x = x1.min(), x2.max()
        min_y, max_y = y1.min(), y2.max()

        # Apply padding from center
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        w = (max_x - min_x) * self.adaptive_zone_padding
        h = (max_y - min_y) * self.adaptive_zone_padding

        min_x = max(0, cx - w / 2)
        max_x = min(self.img_w if hasattr(self, 'img_w') else max_x, cx + w / 2)
        min_y = max(0, cy - h / 2)
        max_y = min(self.img_h if hasattr(self, 'img_h') else max_y, cy + h / 2)

        return np.array([min_x, min_y, max_x, max_y])

    def _tlwh_to_xyxy(self, tlwh):
        """Convert tlwh[x, y, w, h] to xyxy[x1, y1, x2, y2]."""
        x1, y1, w, h = tlwh
        return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

    def _is_outside_effective_zone(self, tlwh):
        """Check whether tlwh box extends outside current effective zone."""
        if self._effective_zone is None:
            return True
        x1, y1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h
        ex1, ey1, ex2, ey2 = self._effective_zone
        return x1 < ex1 or y1 < ey1 or x2 > ex2 or y2 > ey2

    def _get_bbox_from_detections(self, detections):
        """Get xyxy bbox that encloses all given detections."""
        if not detections:
            return None
        boxes = np.array([self._tlwh_to_xyxy(det.tlwh) for det in detections], dtype=np.float32)
        return np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()], dtype=np.float32)

    def _clip_zone_to_image(self, zone_xyxy):
        """Clip effective zone to image bounds."""
        if zone_xyxy is None:
            return None
        if not hasattr(self, 'img_w') or not hasattr(self, 'img_h'):
            return zone_xyxy
        zone_xyxy[0] = max(0, zone_xyxy[0])
        zone_xyxy[1] = max(0, zone_xyxy[1])
        zone_xyxy[2] = min(self.img_w, zone_xyxy[2])
        zone_xyxy[3] = min(self.img_h, zone_xyxy[3])
        return zone_xyxy

    def _apply_padding_to_zone(self, zone_xyxy):
        """Apply adaptive padding around zone center (used for initial zone only)."""
        if zone_xyxy is None:
            return None
        x1, y1, x2, y2 = zone_xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w = (x2 - x1) * self.adaptive_zone_padding
        h = (y2 - y1) * self.adaptive_zone_padding
        padded = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
        return self._clip_zone_to_image(padded)

    def _mark_outside_zone_det_inds(self, detections):
        """Mark detection indices that were outside effective zone before current expansion."""
        for det in detections:
            if det.tlwh[2] * det.tlwh[3] < self.adaptive_zone_min_box_area:
                continue
            if self._is_outside_effective_zone(det.tlwh):
                self._outside_zone_det_inds.add(int(det.det_ind))

    def _select_expand_candidates(self, detections, phase):
        """Select detection candidates for effective-zone expansion in always_expand mode."""
        if not detections:
            return []

        trigger = self.adaptive_zone_expand_trigger
        if trigger not in ('all_high', 'outside_high', 'unmatched_high'):
            trigger = 'all_high'

        candidates = []
        for det in detections:
            if det.tlwh[2] * det.tlwh[3] < self.adaptive_zone_min_box_area:
                continue
            outside = self._is_outside_effective_zone(det.tlwh)
            if trigger == 'all_high':
                candidates.append(det)
            elif trigger == 'outside_high' and outside:
                candidates.append(det)
            elif trigger == 'unmatched_high':
                # unmatched_high expansion is only meaningful in step4 where detections are unmatched
                if phase == 'step4' or self._effective_zone is None:
                    candidates.append(det)
        return candidates

    def _update_effective_zone(self, detections, phase='pre'):
        """Update effective zone.

        Args:
            detections: List of STrack objects from current frame
            phase: "pre" (before association) or "step4" (unmatched-high stage)
        """
        if not self.adaptive_zone_enabled:
            return

        if len(detections) == 0:
            return

        mode = self.adaptive_zone_update_mode
        if mode not in ('warmup_once', 'always_expand'):
            mode = 'always_expand'

        if mode == 'warmup_once':
            if self.frame_count > self.adaptive_zone_warmup:
                return
            for det in detections:
                self._warmup_detections.append(det.tlwh.copy())
            if self.frame_count == self.adaptive_zone_warmup and len(self._warmup_detections) > 0:
                self._effective_zone = self._compute_effective_zone()
                self._effective_zone_last_frame = self.frame_count
            return

        # always_expand mode: mark outside detections using previous-frame zone
        self._mark_outside_zone_det_inds(detections)
        candidates = self._select_expand_candidates(detections, phase=phase)
        if not candidates:
            return

        candidate_bbox = self._get_bbox_from_detections(candidates)
        if candidate_bbox is None:
            return

        if self._effective_zone is None:
            # First valid frame: initialize effective zone from detections with optional padding.
            self._effective_zone = self._apply_padding_to_zone(candidate_bbox.astype(np.float32))
        else:
            # Monotonic expansion only: expand to include out-of-zone detections, never shrink.
            ex1, ey1, ex2, ey2 = self._effective_zone
            cx1, cy1, cx2, cy2 = candidate_bbox
            expanded = np.array([min(ex1, cx1), min(ey1, cy1), max(ex2, cx2), max(ey2, cy2)], dtype=np.float32)
            self._effective_zone = self._clip_zone_to_image(expanded)
        self._effective_zone_last_frame = self.frame_count

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.

        Args:
            box1: [left, top, width, height] (tlwh format)
            box2: [left, top, width, height] (tlwh format)

        Returns:
            float: IoU value [0, 1]
        """
        # tlwh format: [left(x), top(y), width, height]
        x1 = max(box1[0], box2[0])  # max(left)
        y1 = max(box1[1], box2[1])  # max(top)
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])  # min(right = left+width)
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])  # min(bottom = top+height)

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

    def _calculate_center_distance(self, box1_tlwh, box2_tlwh):
        """Calculate Euclidean distance between box centers.

        Args:
            box1_tlwh: [left, top, width, height]
            box2_tlwh: [left, top, width, height]

        Returns:
            float: Euclidean distance between centers
        """
        # tlwh format: [left(x), top(y), width, height]
        c1_x = box1_tlwh[0] + box1_tlwh[2] / 2  # left + width/2
        c1_y = box1_tlwh[1] + box1_tlwh[3] / 2  # top + height/2
        c2_x = box2_tlwh[0] + box2_tlwh[2] / 2  # left + width/2
        c2_y = box2_tlwh[1] + box2_tlwh[3] / 2  # top + height/2

        return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)


    def _try_match_zombie(self, detection, zombie_stracks, max_dist=None):
        """Try to match detection with zombie tracks using nearest distance.

        Args:
            detection: STrack object, unmatched detection
            zombie_stracks: list[STrack], zombie tracks to search (lost > 30 frames)
            max_dist: Maximum distance threshold, defaults to self.zombie_match_max_dist

        Returns:
            STrack or None: Matched zombie track if found within max_dist, None otherwise
        """
        if max_dist is None:
            max_dist = self.zombie_match_max_dist
        if self.zombie_dist_thresh > 0:
            max_dist = min(max_dist, self.zombie_dist_thresh)
        if not self._zombie_enabled():
            return None

        if len(zombie_stracks) == 0:
            return None

        best_dist = float('inf')
        best_zombie = None

        for zombie in zombie_stracks:
            zombie_tlwh = zombie.get_tlwh_for_matching(
                frame_id=self.frame_count,
                max_predict_frames=self.zombie_max_predict_frames
            )

            dist = self._calculate_center_distance(detection.tlwh, zombie_tlwh)
            if dist < best_dist:
                best_dist = dist
                best_zombie = zombie

        if best_dist < max_dist:
            # Reactivate the nearest zombie
            best_zombie.re_activate(detection, self.frame_count, new_id=False)
            return best_zombie

        return None

    def _prune_pending_births(self):
        """Drop stale pending birth candidates that missed too many consecutive frames."""
        if not self.pending_births:
            return
        keep = []
        for pending in self.pending_births:
            if self.frame_count - pending["last_frame"] <= self._birth_pending_max_miss:
                keep.append(pending)
        self.pending_births = keep

    def _collect_birth_reference_tracks(self, *track_groups):
        """Build a deduplicated list of tracks used to suppress duplicate new births."""
        refs = []
        seen = set()
        for group in track_groups:
            for track in group:
                if track is None or track.state == TrackState.Removed:
                    continue
                key = id(track)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(track)
        return refs

    def _is_birth_suppressed(self, det_track, reference_tracks) -> bool:
        """Check whether a new birth candidate is too close to existing tracks."""
        if self.birth_suppress_iou <= 0 and self.birth_suppress_center_dist <= 0:
            return False

        det_tlwh = det_track.tlwh

        for ref_track in reference_tracks:
            if ref_track is det_track:
                continue
            ref_tlwh = ref_track.get_tlwh_for_matching(
                frame_id=self.frame_count,
                max_predict_frames=self.zombie_max_predict_frames
            )
            if self.birth_suppress_iou > 0:
                if self._calculate_iou(det_tlwh, ref_tlwh) >= self.birth_suppress_iou:
                    return True
            if self.birth_suppress_center_dist > 0:
                if self._calculate_center_distance(det_tlwh, ref_tlwh) <= self.birth_suppress_center_dist:
                    return True

        return False

    def _find_pending_birth_idx(self, det_track):
        """Find best pending candidate for current detection."""
        if not self.pending_births:
            return -1

        max_dist = self.birth_suppress_center_dist if self.birth_suppress_center_dist > 0 else 40.0
        best_idx = -1
        best_iou = -1.0
        best_dist = float("inf")

        for idx, pending in enumerate(self.pending_births):
            pending_track = pending["track"]
            iou = self._calculate_iou(det_track.tlwh, pending_track.tlwh)
            dist = self._calculate_center_distance(det_track.tlwh, pending_track.tlwh)
            if iou < self._birth_confirm_iou and dist > max_dist:
                continue

            if iou > best_iou or (np.isclose(iou, best_iou) and dist < best_dist):
                best_idx = idx
                best_iou = iou
                best_dist = dist

        return best_idx

    def _try_activate_new_track(self, det_track, activated_starcks, reference_tracks):
        """Apply duplicate suppression and temporal confirmation before spawning a new ID."""
        if self.birth_confirm_frames <= 1:
            if self._is_birth_suppressed(det_track, reference_tracks):
                return False
            det_track.activate(self.kalman_filter, self.frame_count)
            self._set_exit_pending(det_track, False)
            activated_starcks.append(det_track)
            return True

        pending_idx = self._find_pending_birth_idx(det_track)
        if self._is_birth_suppressed(det_track, reference_tracks):
            return False
        if pending_idx < 0:
            self.pending_births.append(
                {
                    "track": det_track,
                    "hits": 1,
                    "last_frame": self.frame_count,
                }
            )
            return False

        pending = self.pending_births.pop(pending_idx)
        hits = pending["hits"] + 1
        if hits >= self.birth_confirm_frames:
            det_track.activate(self.kalman_filter, self.frame_count)
            self._set_exit_pending(det_track, False)
            activated_starcks.append(det_track)
            return True

        pending["track"] = det_track
        pending["hits"] = hits
        pending["last_frame"] = self.frame_count
        self.pending_births.append(pending)
        return False

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None
    ) -> np.ndarray:

        self.check_inputs(dets, img)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_count += 1
        self._outside_zone_det_inds = set()
        self._prune_pending_births()

        # Store image dimensions for adaptive zone computation
        if img is not None:
            self.img_h, self.img_w = img.shape[:2]
        elif hasattr(self, 'h') and hasattr(self, 'w'):
            self.img_h, self.img_w = self.h, self.w

        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        confs = dets[:, 4]

        remain_inds = confs > self.track_thresh

        inds_low = confs > self.min_conf
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [STrack(det, max_obs=self.max_obs) for det in dets]
        else:
            detections = []

        # Update effective zone before association.
        if self.adaptive_zone_enabled:
            self._update_effective_zone(detections, phase='pre')

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        # IMPORTANT: Include lost tracks for matching (same as original ByteTrack)
        # This allows quick recovery from short-term occlusion (<30 frames)
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_count)
                self._set_exit_pending(track, False)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                self._set_exit_pending(track, False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [
                STrack(det_second, max_obs=self.max_obs) for det_second in dets_second
            ]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_count)
                self._set_exit_pending(track, False)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                self._set_exit_pending(track, False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                # Check exit zone: mark exit-pending for delayed removal after grace frames
                if self.exit_zone_enabled:
                    track_tlwh = track.tlwh
                    if img is not None:
                        current_img_shape = img.shape[:2]
                    elif hasattr(self, 'h') and hasattr(self, 'w'):
                        current_img_shape = (self.h, self.w)
                    else:
                        current_img_shape = (1080, 1920)

                    in_exit_zone = self._is_in_exit_zone(track_tlwh, current_img_shape)
                    if in_exit_zone:
                        self._set_exit_pending(track, True)
                    else:
                        self._set_exit_pending(track, False)
                else:
                    self._set_exit_pending(track, False)

                # Normal flow: mark as lost
                track.mark_lost()
                track.lost_frame_id = self.frame_count  # Record when track became lost
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_count)
            self._set_exit_pending(unconfirmed[itracked], False)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Third association - Zombie track rescue (after normal association) """
        # This is the NEW stage: match remaining detections with zombie tracks
        # Zombie tracks = tracks that have been lost for > 30 frames
        # These tracks would normally be discarded by original ByteTrack
        # Get image dimensions
        if img is not None:
            current_img_shape = img.shape[:2]
        elif hasattr(self, 'h') and hasattr(self, 'w'):
            current_img_shape = (self.h, self.w)
        else:
            current_img_shape = (1080, 1920)

        # Track which zombies were rescued
        rescued_zombies = []

        if self.frame_count == 1:
            # First frame: give all detections new IDs
            for inew in u_detection:
                det_track = detections[inew]
                if det_track.conf < self.new_track_thresh:
                    continue
                det_track.activate(self.kalman_filter, self.frame_count)
                activated_starcks.append(det_track)
        else:
            # Normal frames: process unmatched detections
            for inew in u_detection:
                det_track = detections[inew]
                if det_track.conf < self.new_track_thresh:
                    continue

                # In always_expand mode, unmatched-high trigger can expand zone in step4.
                if (
                    self.adaptive_zone_enabled
                    and self.adaptive_zone_update_mode == 'always_expand'
                    and self.adaptive_zone_expand_trigger in ('unmatched_high', 'outside_high')
                ):
                    self._update_effective_zone([det_track], phase='step4')

                # If this detection was outside effective zone before expansion this frame,
                # allow immediate ID creation (newly entered or newly visible region).
                if (
                    self.adaptive_zone_enabled
                    and self.adaptive_zone_update_mode == 'always_expand'
                    and int(det_track.det_ind) in self._outside_zone_det_inds
                ):
                    birth_refs = self._collect_birth_reference_tracks(
                        self.active_tracks,
                        self.lost_stracks,
                        self.zombie_stracks,
                        activated_starcks,
                        refind_stracks,
                        lost_stracks,
                    )
                    self._try_activate_new_track(det_track, activated_starcks, birth_refs)
                    continue

                det_tlwh = det_track.tlwh
                is_entry = self._is_in_entry_zone(det_tlwh, current_img_shape)

                if is_entry:
                    # In entry zone -> create new ID (original behavior)
                    birth_refs = self._collect_birth_reference_tracks(
                        self.active_tracks,
                        self.lost_stracks,
                        self.zombie_stracks,
                        activated_starcks,
                        refind_stracks,
                        lost_stracks,
                    )
                    self._try_activate_new_track(det_track, activated_starcks, birth_refs)
                else:
                    # In center zone -> try match with zombie tracks
                    # Zombie tracks are those lost for > 30 frames
                    zombie = self._try_match_zombie(det_track, self.zombie_stracks)
                    if zombie:
                        # Rescued a zombie track!
                        refind_stracks.append(zombie)
                        rescued_zombies.append(zombie)
                    else:
                        # Strict entry gate: center-zone detections do not spawn new IDs.
                        # This is the core "严进" policy for fixed-camera deployments.
                        if self.strict_entry_gate and self.entry_margin > 0:
                            continue
                        birth_refs = self._collect_birth_reference_tracks(
                            self.active_tracks,
                            self.lost_stracks,
                            self.zombie_stracks,
                            activated_starcks,
                            refind_stracks,
                            lost_stracks,
                        )
                        self._try_activate_new_track(det_track, activated_starcks, birth_refs)

        """ Step 5: Update lost tracks and transition to zombie """
        # Add newly lost tracks to self.lost_stracks
        for track in lost_stracks:
            self.lost_stracks.append(track)

        zombie_enabled = self._zombie_enabled()

        # Freeze positions for tracks that have been lost too long (zombie mode only)
        if zombie_enabled and self.zombie_max_predict_frames > 0:
            for track in self.lost_stracks:
                if track.frozen_mean is None:
                    frames_lost = self.frame_count - track.lost_frame_id
                    if frames_lost >= self.zombie_max_predict_frames:
                        track.frozen_mean = track.mean.copy()

        """ Step 6: Transition old lost tracks to zombie tracks """
        # In zombie mode: move stale lost tracks into zombie pool.
        # In baseline mode: remove stale lost tracks using original ByteTrack policy.
        new_lost_stracks = []
        for track in self.lost_stracks:
            frames_lost = self.frame_count - track.lost_frame_id
            if self.exit_zone_enabled and bool(getattr(track, 'exit_pending', False)):
                grace = max(1, int(self.exit_zone_remove_grace))
                if frames_lost >= grace:
                    track.mark_removed()
                    removed_stracks.append(track)
                    self._set_exit_pending(track, False)
                    continue
            if zombie_enabled:
                if frames_lost >= self.zombie_transition_frames:
                    # This track would be removed in original ByteTrack
                    # Instead, move it to zombie_stracks for potential later rescue
                    if track.frozen_mean is None:
                        track.frozen_mean = track.mean.copy()
                    self.zombie_stracks.append(track)
                else:
                    new_lost_stracks.append(track)
            else:
                if frames_lost > self.max_time_lost:
                    track.mark_removed()
                    self._set_exit_pending(track, False)
                    removed_stracks.append(track)
                else:
                    new_lost_stracks.append(track)
        self.lost_stracks = new_lost_stracks

        """ Step 7: Remove rescued zombies from zombie_stracks """
        if rescued_zombies:
            rescued_ids = {t.id for t in rescued_zombies}
            self.zombie_stracks = [t for t in self.zombie_stracks if t.id not in rescued_ids]

        """ Step 8: Limit max zombie history """
        if not zombie_enabled and self.zombie_stracks:
            # Ensure baseline mode does not keep any zombie state.
            for track in self.zombie_stracks:
                track.mark_removed()
                removed_stracks.append(track)
            self.zombie_stracks = []
        elif len(self.zombie_stracks) > self.zombie_max_history:
            self.zombie_stracks.sort(key=lambda t: t.lost_frame_id)
            tracks_to_remove = self.zombie_stracks[:-self.zombie_max_history]
            for track in tracks_to_remove:
                track.mark_removed()
                removed_stracks.append(track)
            self.zombie_stracks = self.zombie_stracks[-self.zombie_max_history:]

        """ Step 9: Limit max lost tracks history """
        if self.lost_max_history > 0 and len(self.lost_stracks) > self.lost_max_history:
            self.lost_stracks.sort(key=lambda t: t.lost_frame_id)
            tracks_to_remove = self.lost_stracks[:-self.lost_max_history]
            for track in tracks_to_remove:
                track.mark_removed()
                removed_stracks.append(track)
            self.lost_stracks = self.lost_stracks[-self.lost_max_history:]

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        # Note: lost_stracks are added to self.lost_stracks in Step 5
        # So we don't need to extend again here
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.zombie_stracks = sub_stracks(self.zombie_stracks, self.active_tracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )
        # get confs of lost tracks
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)
        return outputs


# id, class_id, conf


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
