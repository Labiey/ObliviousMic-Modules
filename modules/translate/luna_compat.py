# SPDX-License-Identifier: GPL-3.0-only
"""Small host-side interface for vendored Luna engines; no Luna GUI or scheduler."""

from datetime import UTC
from types import SimpleNamespace


class Languages:
    Auto = "auto"
    Chinese = "zh"
    TradChinese = "zh-Hant"
    English = "en"
    Japanese = "ja"
    Korean = "ko"
    Spanish = "es"
    French = "fr"
    Vietnamese = "vi"
    Ukrainian = "uk"
    Arabic = "ar"
    Swedish = "sv"
    Latin = "la"


# Volcengine's copied SDK uses these only as an annotation and a UTC accessor.
requests = SimpleNamespace(Session=object)
pytz = SimpleNamespace(timezone=lambda name: UTC if name == "UTC" else None)


class basetrans:
    needzhconv = False

    def __init__(self, config, credentials, session):
        self.config = config
        self.multiapikeycurrent = credentials
        self.proxysession = session
        self.srclang_1 = "auto"
        self.tgtlang_1 = "en"

    def init(self):
        pass

    def langmap(self):
        return {}

    @property
    def srclang(self):
        return self.langmap().get(self.srclang_1, self.srclang_1)

    @property
    def tgtlang(self):
        return self.langmap().get(self.tgtlang_1, self.tgtlang_1)

    @property
    def is_src_auto(self):
        return self.srclang_1 == "auto"

    def checkempty(self, names):
        if isinstance(names, str):
            names = [names]
        if any(not str(self.multiapikeycurrent.get(name, "")).strip() for name in names):
            raise ValueError("missing engine credential")
