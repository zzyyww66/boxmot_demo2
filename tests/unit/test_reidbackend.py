import cv2
import numpy as np
import pytest
from pathlib import Path

from boxmot.reid.backends.onnx_backend import ONNXBackend
from boxmot.reid.backends.base_backend import BaseModelBackend
from boxmot.reid.backends.openvino_backend import OpenVinoBackend
from boxmot.reid.backends.pytorch_backend import PyTorchBackend
from boxmot.reid.backends.torchscript_backend import TorchscriptBackend
from boxmot.reid.core.auto_backend import ReidAutoBackend
from boxmot.utils import ROOT, WEIGHTS

# generated in previous job step
EXPORTED_REID_MODELS = [
    WEIGHTS / "osnet_x0_25_msmt17.pt",
    WEIGHTS / "osnet_x0_25_msmt17.torchscript",
    WEIGHTS / "osnet_x0_25_msmt17.onnx",
    WEIGHTS / "osnet_x0_25_msmt17_openvino_model",
]

ASSOCIATED_BACKEND = [
    PyTorchBackend,
    TorchscriptBackend,
    ONNXBackend,
    OpenVinoBackend
]


@pytest.mark.parametrize("reid_model", EXPORTED_REID_MODELS)
def test_reidbackend_output(reid_model):

    rab = ReidAutoBackend(weights=reid_model, device="cpu", half=False)
    b = rab.get_backend()

    img = cv2.imread(
        str(ROOT / "assets/MOT17-mini/train/MOT17-04-FRCNN/img1/000001.jpg")
    )
    dets = np.array([[144, 212, 578, 480, 0.82, 0],
                     [425, 281, 576, 472, 0.56, 65]])

    embs = b.get_features(dets[:, 0:4], img)
    assert embs.shape[0] == 2  # two crops should give two embeddings
    assert embs.shape[1] == 512  # osnet embeddings are of size 512


@pytest.mark.parametrize(
    "exported_reid_model, backend", zip(EXPORTED_REID_MODELS, ASSOCIATED_BACKEND)
)
def test_reidbackend_type(exported_reid_model, backend):

    rab = ReidAutoBackend(weights=exported_reid_model, device="cpu", half=False)
    b = rab.get_backend()

    assert isinstance(b, backend)


def test_reidbackend_resolves_existing_root_weights_before_download(tmp_path, monkeypatch):
    local_root_weight = ROOT / "osnet_x0_25_msmt17.pt"
    if not local_root_weight.exists():
        pytest.skip(f"Expected local weight not found: {local_root_weight}")

    monkeypatch.chdir(tmp_path)
    missing_engine_weight = tmp_path / "boxmot" / "engine" / "weights" / "osnet_x0_25_msmt17.pt"

    resolved = BaseModelBackend.resolve_weights_path(missing_engine_weight)

    assert resolved == local_root_weight
