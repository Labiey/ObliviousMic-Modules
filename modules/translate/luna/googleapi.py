# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import basetrans
from obmic_luna_compat_v1_1 import Languages
from html import unescape


class TS(basetrans):
    def langmap(self):
        return {Languages.Chinese: "zh-CN", Languages.TradChinese: "zh-TW"}

    def translate(self, query):
        self.checkempty("key")

        key = self.multiapikeycurrent["key"]
        params = {
            "key": key,
            "target": self.tgtlang,
            "q": (query),
        }
        if not self.is_src_auto:
            params["source"] = self.srclang
        response = self.proxysession.get(
            "https://translation.googleapis.com/language/translate/v2/", params=params
        )

        try:
            return unescape(
                response.json()["data"]["translations"][0]["translatedText"]
            )
        except:
            raise Exception(response)
