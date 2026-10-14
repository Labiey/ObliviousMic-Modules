import numpy as np
import pytest

from modules.bsrnn.lib.frame_glue import FrameGlueError
from modules.bsrnn.lib.onnx_frame_backend import OnnxFrameBackend
from tests.models.test_onnx_frame_backend import _SessionDouble


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 31, 256), np.float32), np.full((1, 32, 256), np.nan, np.float32)]
)
def test_invalid_returned_recurrent_state_is_rejected_before_reuse(bad):
    class Broken(_SessionDouble):
        def run(self, names, feeds):
            result = super().run(names, feeds)
            result[2] = bad
            return result

    backend = OnnxFrameBackend(session=Broken(), win=512, hop=128)
    with pytest.raises(FrameGlueError, match="invalid recurrent"):
        backend.separate_frames(np.zeros((2, 512), np.float32), backend.initial_states())
