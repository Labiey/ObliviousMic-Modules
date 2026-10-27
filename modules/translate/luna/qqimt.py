# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import basetrans
import uuid, time


class TS(basetrans):
    needzhconv = True

    def translate(self, query):

        headers = {
            "accept": "application/json, text/plain, */*",
            "accept-language": "zh-CN,zh;q=0.9,ar;q=0.8,sq;q=0.7,ru;q=0.6",
            "origin": "https://transmart.qq.com",
            "priority": "u=1, i",
            "referer": "https://transmart.qq.com/zh-CN/index?sourcelang=zh&targetlang=en&source=%E6%B5%8B%E8%AF%951%0A%E6%B5%8B%E8%AF%951%E8%AF%B7%E6%B1%82",
            "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-requested-with": "XMLHttpRequest",
        }

        json_data = {
            "header": {
                "fn": "auto_translation",
                "session": "",
                "client_key": "browser-chrome-124.0.0-Windows_10-"
                + str(uuid.uuid4())
                + "-"
                + str(int(time.time())),
                "user": "",
            },
            "type": "plain",
            "model_category": "normal",
            "text_domain": "general",
            "source": {
                "lang": self.srclang,
                "text_list": [
                    "",
                    query,
                    "",
                ],
            },
            "target": {
                "lang": self.tgtlang,
            },
        }

        response = self.proxysession.post(
            "https://transmart.qq.com/api/imt", headers=headers, json=json_data
        )

        return "".join(response.json()["auto_translation"])
