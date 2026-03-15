from boxmot import (
    BoostTrack,
    BotSort,
    ByteTrack,
    DeepOcSort,
    OcSort,
    StrongSort,
    HybridSort,
)

MOTION_N_APPEARANCE_TRACKING_NAMES = [
    "botsort",
    "deepocsort",
    "strongsort",
    "boosttrack",
    "hybridsort",
    "bytetrack",
]
MOTION_ONLY_TRACKING_NAMES = ["ocsort"]

MOTION_N_APPEARANCE_TRACKING_METHODS = [StrongSort, BotSort, DeepOcSort, BoostTrack, HybridSort, ByteTrack]
MOTION_ONLY_TRACKING_METHODS = [OcSort]

ALL_TRACKERS = [
    "botsort",
    "deepocsort",
    "ocsort",
    "bytetrack",
    "strongsort",
    "boosttrack",
    "hybridsort",
]
PER_CLASS_TRACKERS = ["botsort", "deepocsort", "ocsort", "bytetrack", "boosttrack", "hybridsort"]
