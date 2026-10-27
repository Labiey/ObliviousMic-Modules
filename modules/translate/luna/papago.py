# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import basetrans
from obmic_luna_compat_v1_1 import Languages


class TS(basetrans):
    def langmap(self):
        return {Languages.Chinese: "zh-CN", Languages.TradChinese: "zh-TW"}

    def translate(self, content):
        data = {
            "dict": "true",
            "dictDisplay": "30",
            "honorific": "false",
            "useGlossary": "false",
            "source": self.srclang,
            "target": self.tgtlang,
            "text": content,
        }

        r = self.proxysession.post(
            "https://papago.naver.com/api/text/translation", data=data
        )

        data = r.json()
        try:
            return data["translatedText"]
        except:
            raise Exception(data)
