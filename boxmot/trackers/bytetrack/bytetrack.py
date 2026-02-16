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
            np.ndarray: [top, left, width, height] in tlwh format
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
    - track_buffer (int): Number of frames to keep a track alive after it was last detected.
    - frame_rate (int): Frame rate of the video being processed. Used to scale the track buffer size.
    - entry_margin (int): Pixel width of entry zone at frame edges. New IDs are only created within this margin (birth control). Set to 0 to disable (default: 0).
    - zombie_iou_thresh (float): IoU threshold for rescuing lost tracks (zombie rescue). Higher values require better spatial overlap to reactivate a lost track (default: 0.3).

    Attributes:
    - frame_count (int): Counter for the frames processed.
    - active_tracks (list): List to hold active tracks.
    - lost_stracks (list[STrack]): List of lost tracks.
    - removed_stracks (list[STrack]): List of removed tracks.
    - buffer_size (int): Size of the track buffer based on frame rate.
    - max_time_lost (int): Maximum time a track can be lost.
    - kalman_filter (KalmanFilterXYAH): Kalman filter for motion prediction.
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

        # Motion model
        self.kalman_filter = KalmanFilterXYAH()

        # Lifecycle gating configuration (scene-mask based birth control and zombie rescue)
        self.entry_margin = kwargs.get('entry_margin', 0)  # Entry zone pixel width (0 = disabled)
        self.zombie_iou_thresh = kwargs.get('zombie_iou_thresh', 0.3)  # IoU threshold for zombie rescue

        # Zombie track management parameters
        self.zombie_max_history = kwargs.get('zombie_max_history', 100)  # Max number of zombie tracks to keep
        self.zombie_dist_thresh = kwargs.get('zombie_dist_thresh', 150)  # Distance threshold for zombie rescue (pixels)
        self.zombie_max_predict_frames = kwargs.get('zombie_max_predict_frames', 5)  # Max frames to predict zombie position

        self.active_tracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

    def _is_in_entry_zone(self, tlwh, img_shape, margin=None):
        """
        Check if a bounding box is in the entry zone (edge of the frame).

        Args:
            tlwh: [top, left, width, height]
            img_shape: (height, width)
            margin: Edge zone pixel width, defaults to self.entry_margin

        Returns:
            bool: True if the box touches the edge zone
        """
        if margin is None:
            margin = self.entry_margin

        # If entry_margin is 0, consider everything as entry zone (disabled gating)
        if margin <= 0:
            return True

        img_h, img_w = img_shape[0], img_shape[1]
        # tlwh format: [top(y), left(x), width, height]
        y1, x1, w, h = tlwh
        x2, y2 = x1 + w, y1 + h

        # Consider entry if any side touches the edge zone
        if x1 < margin or y1 < margin:
            return True
        if x2 > (img_w - margin) or y2 > (img_h - margin):
            return True

        return False

    def _calculate_iou(self, box1, box2):
        """
        Calculate IoU between two bounding boxes.

        Args:
            box1: [x, y, w, h] (tlwh format)
            box2: [x, y, w, h] (tlwh format)

        Returns:
            float: IoU value [0, 1]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]

        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return iou

    def _calculate_center_distance(self, box1_tlwh, box2_tlwh):
        """Calculate Euclidean distance between box centers.

        Args:
            box1_tlwh: [top, left, width, height]
            box2_tlwh: [top, left, width, height]

        Returns:
            float: Euclidean distance between centers
        """
        # Calculate centers
        c1_x = box1_tlwh[0] + box1_tlwh[2] / 2
        c1_y = box1_tlwh[1] + box1_tlwh[3] / 2
        c2_x = box2_tlwh[0] + box2_tlwh[2] / 2
        c2_y = box2_tlwh[1] + box2_tlwh[3] / 2

        return np.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)

    def _try_rescue_zombie(self, detection, lost_stracks, iou_thresh=None, dist_thresh=None):
        """
        Try to rescue a lost track (zombie) using hybrid matching strategy.

        Stage 1: IoU matching with threshold
        Stage 2: For unmatched, use center Euclidean distance to find nearest zombie

        Args:
            detection: STrack object, unmatched detection
            lost_stracks: list[STrack], lost tracks to search
            iou_thresh: IoU threshold, defaults to self.zombie_iou_thresh
            dist_thresh: Distance threshold, defaults to self.zombie_dist_thresh

        Returns:
            STrack or None: Resurrected track if found, None otherwise
        """
        if iou_thresh is None:
            iou_thresh = self.zombie_iou_thresh
        if dist_thresh is None:
            dist_thresh = self.zombie_dist_thresh

        if len(lost_stracks) == 0:
            return None

        # Stage 1: IoU matching
        best_iou = 0.0
        best_iou_track = None

        for track in lost_stracks:
            # Use get_tlwh_for_matching to respect frozen positions for old zombies
            track_tlwh = track.get_tlwh_for_matching(self.frame_count, self.zombie_max_predict_frames)
            iou = self._calculate_iou(detection.tlwh, track_tlwh)
            if iou > best_iou:
                best_iou = iou
                best_iou_track = track

        if best_iou > iou_thresh:
            # Reactivate the lost track
            best_iou_track.re_activate(detection, self.frame_count, new_id=False)
            return best_iou_track

        # Stage 2: Distance matching - find nearest zombie within threshold
        best_dist = float('inf')
        best_dist_track = None

        for track in lost_stracks:
            # Use get_tlwh_for_matching to respect frozen positions
            track_tlwh = track.get_tlwh_for_matching(self.frame_count, self.zombie_max_predict_frames)
            dist = self._calculate_center_distance(detection.tlwh, track_tlwh)
            if dist < best_dist:
                best_dist = dist
                best_dist_track = track

        if best_dist < dist_thresh:
            # Reactivate the nearest lost track
            best_dist_track.re_activate(detection, self.frame_count, new_id=False)
            return best_dist_track

        return None

    @BaseTracker.setup_decorator
    @BaseTracker.per_class_decorator
    def update(
        self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None
    ) -> np.ndarray:

        self.check_inputs(dets, img)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_count += 1
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

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
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
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
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
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
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
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks with lifecycle gating (strict birth control & zombie rescue) """
        # Get image dimensions
        if img is not None:
            current_img_shape = img.shape[:2]
        elif hasattr(self, 'h') and hasattr(self, 'w'):
            current_img_shape = (self.h, self.w)
        else:
            # Fallback: assume 1080p (should not happen due to setup_decorator)
            current_img_shape = (1080, 1920)

        # Track which zombies were rescued so we can remove them from lost_stracks
        rescued_zombies = []

        if self.frame_count == 1:
            # First frame: give all detections new IDs, skip entry zone check
            for inew in u_detection:
                det_track = detections[inew]
                if det_track.conf < self.det_thresh:
                    continue
                det_track.activate(self.kalman_filter, self.frame_count)
                activated_starcks.append(det_track)
        else:
            # Normal frames: apply entry zone gating and zombie rescue
            for inew in u_detection:
                det_track = detections[inew]
                if det_track.conf < self.det_thresh:
                    continue

                det_tlwh = det_track.tlwh
                is_entry = self._is_in_entry_zone(det_tlwh, current_img_shape)

                if is_entry:
                    # Case A: In entry zone -> allow new ID creation
                    det_track.activate(self.kalman_filter, self.frame_count)
                    activated_starcks.append(det_track)
                else:
                    # Case B: In center of frame -> try zombie rescue
                    zombie = self._try_rescue_zombie(det_track, self.lost_stracks)
                    if zombie:
                        # Rescued! Add to refind_stracks (not activated_starcks)
                        # because it's a reactivation, not a new activation
                        refind_stracks.append(zombie)
                        rescued_zombies.append(zombie)
                    # If rescue fails, do NOT create new ID (prevents IDSW from false detections)

        """ Step 5: Update state (remove rescued zombies from lost_stracks) """
        # Remove rescued tracks from lost_stracks to prevent duplicate tracking
        if rescued_zombies:
            rescued_ids = {t.id for t in rescued_zombies}
            self.lost_stracks = [t for t in self.lost_stracks if t.id not in rescued_ids]

        """ Step 6: Freeze zombie positions after max prediction frames """
        for track in self.lost_stracks:
            if track.lost_frame_id > 0 and track.frozen_mean is None:
                frames_lost = self.frame_count - track.lost_frame_id
                if frames_lost >= self.zombie_max_predict_frames:
                    # Freeze the mean at current position
                    track.frozen_mean = track.mean.copy()

        """ Step 7: Limit max zombie history (prevent memory growth) """
        if len(self.lost_stracks) > self.zombie_max_history:
            # Sort by lost_frame_id (oldest first) and remove oldest
            self.lost_stracks.sort(key=lambda t: t.lost_frame_id)
            tracks_to_remove = self.lost_stracks[:-self.zombie_max_history]
            for track in tracks_to_remove:
                track.mark_removed()
                removed_stracks.append(track)
            self.lost_stracks = self.lost_stracks[-self.zombie_max_history:]

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
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
