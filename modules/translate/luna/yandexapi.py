# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import basetrans


class TS(basetrans):
    needzhconv = True

    def translate(self, content):
        self.checkempty("key")

        key = self.multiapikeycurrent["key"]

        url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        js = {
            "targetLanguageCode": self.tgtlang,
            "texts": [content],
        }
        if not self.is_src_auto:
            js["sourceLanguageCode"] = self.srclang
        response = self.proxysession.post(
            url, json=js, headers={"Authorization": "Api-Key " + key}
        )

        try:
            return response.json()["translations"][0]["text"]
        except:
            raise Exception(response)
