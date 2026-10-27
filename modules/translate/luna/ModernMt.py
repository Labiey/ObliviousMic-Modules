# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
import hashlib
from obmic_luna_compat_v1_1 import basetrans
import time
from obmic_luna_compat_v1_1 import Languages


class TS(basetrans):

    def init(self):
        _ = self.proxysession.get(
            "https://www.modernmt.com/translate",
            headers={
                "Referer": "https://www.modernmt.com/translate",
            },
        )

    def translate(self, content):
        time_stamp = int(time.time() * 1e3)
        form_data = {
            "q": content,
            "source": self.srclang,
            "target": self.tgtlang,
            "ts": time_stamp,
            "verify": hashlib.md5(
                "webkey_E3sTuMjpP8Jez49GcYpDVH7r#{}#{}".format(
                    time_stamp, content
                ).encode()
            ).hexdigest(),
            "hints": "",
            "multiline": "true",
        }
        r = self.proxysession.post(
            "https://webapi.modernmt.com/translate",
            json=form_data,
            headers={
                "Origin": "https://www.modernmt.com",
                "Referer": "https://www.modernmt.com/translate",
                "X-Requested-With": "XMLHttpRequest",
                "X-HTTP-Method-Override": "GET",
            },
        )
        try:
            data = r.json()
            return data["data"]["translation"]
        except:
            raise Exception(r)

    def langmap(self):
        return {
            Languages.Chinese: "zh-CN",
            Languages.TradChinese: "zh-TW",
            Languages.Auto: "",
        }
