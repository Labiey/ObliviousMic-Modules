# SPDX-License-Identifier: GPL-3.0-only
"""Lazy loading of pinned Luna engines inside the existing isolated network worker."""

import importlib
import json
import sys
from pathlib import Path

from obliviousmic.sdk import PublicModuleError

try:
    from . import luna_compat
    from .luna_transport import Session
except ImportError:
    import luna_compat
    from luna_transport import Session

ROOT = Path(__file__).parent
ENGINES = json.loads((ROOT / "engines.json").read_text("utf-8"))


def credentials(config):
    key, secret, app = config["engine_key"], config["engine_secret"], config["engine_app_id"]
    return {
        "key": key,
        "Token": key,
        "apikey": key,
        "DeepL-Auth-Key": key,
        "DeepL-Auth-Key-2": key,
        "APP ID": app,
        "密钥": secret,
        "API Key": key,
        "Secret Key": secret,
        "SecretId": key,
        "SecretKey": secret,
        "APP_KEY": key,
        "APP_SECRET": secret,
        "Access_Key": key,
        "SECRET_KEY": secret,
        "Access Key ID": key,
        "Secret Access Key": secret,
        "ak": key,
        "sk": secret,
        "endpoint": "cn-north-4",
    }


def validate_engine_config(config):
    engine = ENGINES[config["provider"]]
    required = list(engine.get("credentials", []))
    if config["provider"] == "baidu_api" and config["baidu_interface"] == "bce":
        required = ["baidu_api_key", "engine_secret"]
    if any(not config[name].strip() for name in required):
        raise PublicModuleError("translate_api_key_required")


class Engine:
    def __init__(self, config, session=None, opener=None):
        self.config = dict(config)
        self.spec = ENGINES[config["provider"]]
        self.session = session or Session(
            self.spec["hosts"], config["request_timeout_seconds"], opener
        )
        self.engine = None

    def translate(self, text, source, target):
        self.session.begin()
        try:
            if self.engine is None:
                sys.modules["obmic_luna_compat_v1_1"] = luna_compat
                package = (__package__ + "." if __package__ else "") + "luna."
                cls = importlib.import_module(package + self.spec["file"]).TS
                options = {
                    "interface": {"standard": 0, "bce": 1, "llm": 2}[
                        self.config["baidu_interface"]
                    ],
                    "usewhich": 0 if self.config["deepl_plan"] == "free" else 1,
                    "Region": int(self.config["tencent_region"]),
                }
                secrets = credentials(self.config)
                if (
                    self.config["provider"] == "baidu_api"
                    and self.config["baidu_interface"] == "bce"
                ):
                    secrets["API Key"] = self.config["baidu_api_key"]
                self.engine = cls(options, secrets, self.session)
                self.engine.init()
            self.engine.srclang_1, self.engine.tgtlang_1 = source, target
            if self.engine.needzhconv and target == "zh-Hant":
                raise PublicModuleError("translate_language_unsupported")
            return self.engine.translate(text)
        except PublicModuleError:
            raise
        except Exception:
            if self.session.error:
                raise self.session.error from None
            raise PublicModuleError("translate_engine_response_invalid") from None
