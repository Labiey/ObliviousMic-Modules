# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import basetrans


class TS(basetrans):

    def translate(self, query):
        self.checkempty("apikey")

        apikey = self.multiapikeycurrent["apikey"]

        headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "none",
        }

        params = {
            "from": self.srclang,
            "to": self.tgtlang,
            "src_text": query,
            "apikey": apikey,
        }
        response = self.proxysession.post(
            "https://api.niutrans.com/NiuTransServer/translation",
            headers=headers,
            data=params,
        )

        try:
            return response.json()["tgt_text"]
        except:
            raise Exception(response)
