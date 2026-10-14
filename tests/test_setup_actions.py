import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from modules.campplus.lib.base import model_path
from modules.registration import register_models

from obliviousmic.sdk import AudioFormat, PrepareContext
from obliviousmic.web.service import WebService


def wait(service, job):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        result = service.jobs.get(job["job_id"])
        if result["state"] not in {"queued", "running"}:
            return result
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_module_enrollment_action_and_profile_model_resolution(tmp_path):
    service = WebService(tmp_path)
    register_models(service.registry)
    profile = service.manager.profiles.create("Synthetic")
    pid = profile["profile_id"]
    path = tmp_path / "sample.wav"
    sf.write(path, np.full(48000, 0.03, np.float32), 16000, subtype="FLOAT")
    file_id = service.register_upload(path)
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"synthetic fixture")

    class Extractor:
        def __init__(self, *args, **kwargs):
            pass

        def initialize(self):
            pass

        def embed_enrollment(self, samples):
            assert len(samples) == 48000
            return np.array([1, 0], np.float32)

    lease = SimpleNamespace(value=object(), release=lambda: None)
    with (
        patch("modules.campplus.actions.WindowEmbeddingExtractor", Extractor),
        patch.object(service.manager.assets, "resolve", return_value=model),
        patch.object(service.manager.pool, "acquire_session", return_value=lease),
    ):
        job = service.module_action(
            "org.obliviousmic.campplus",
            "initialize",
            {"profile_id": pid, "file_id": file_id, "speaker_asset": "a" * 64},
        )
        result = wait(service, job)
    assert result["state"] == "completed"
    np.testing.assert_array_equal(service.manager.profiles.artifact(pid, "embedding"), [1, 0])
    assert not path.exists()
    fmt = AudioFormat(16000)
    calls = []
    context = PrepareContext(
        "a",
        {"audio": fmt},
        {},
        "offline",
        {
            "asset.read": lambda asset, owner: calls.append((asset, owner)) or model,
            "profile.read": lambda owner, kind: model,
        },
    )
    assert model_path(context, {"model_asset": "a" * 64, "profile_id": pid}) == model
    assert calls == [("a" * 64, "")]
    assert model_path(context, {"model_asset": "", "profile_id": pid}, private=True) == model
    assert service.catalog()[-1:]  # module-owned action descriptions are discoverable
    service.close()
