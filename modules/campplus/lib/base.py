"""Explicit CAM++ and BSRNN modules; all files/sessions load in prepare only."""

from pathlib import Path

from obliviousmic.sdk.declaration import parameter

COMMON = {
    "model_path": parameter("string", "", private=True),
    "profile_id": parameter("string", "", private=True, format="profile_id"),
    "model_asset": parameter("string", "", private=True, format="asset_id"),
    "engine": parameter("string", "onnxruntime", enum=["onnxruntime", "openvino"]),
    "device": parameter("string", "CPU", enum=["CPU", "GPU", "NPU"]),
    "threads": parameter("integer", 1, minimum=1, maximum=32),
}


def model_path(context, config, *, private=False):
    if private and config["profile_id"] and not config["model_asset"]:
        return Path(context.capabilities["profile.read"](config["profile_id"], "baked_core"))
    if config["model_asset"]:
        return Path(
            context.capabilities["asset.read"](
                config["model_asset"], config["profile_id"] if private else ""
            )
        )
    path = Path(config["model_path"])
    if not path.is_file() or path.suffix.lower() != ".onnx":
        raise ValueError("select a verified external ONNX model")
    return path


def session(context, config, path):
    from .sessions import make_session

    def factory():
        return make_session(config["engine"], path, config["threads"], device=config["device"])

    if "resource.acquire" in context.capabilities:
        lease = context.capabilities["resource.acquire"](path, config, factory)
        return lease.value, lease
    return factory(), None


class ModelModule:
    def __init__(self, factory=None):
        self.factory = factory
        self.lease = None

    def describe(self):
        return self.MANIFEST

    def start(self, stream):
        self.reset("start")

    def update_parameters(self, change):
        raise ValueError("model settings require reprepare")

    def stop(self):
        pass

    def dispose(self):
        if self.lease:
            self.lease.release()
            self.lease = None
        self.processor = None
