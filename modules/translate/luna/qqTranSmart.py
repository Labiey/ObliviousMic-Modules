# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
import uuid
from obmic_luna_compat_v1_1 import basetrans
import time
from obmic_luna_compat_v1_1 import Languages


class TS(basetrans):
    def get_clientKey(self):
        return "browser-firefox-110.0.0-Windows 10-{}-{}".format(
            uuid.uuid4(), int(time.time() * 1e3)
        )

    def split_sentence(self, data):
        index_pair_list = [
            [item["start"], item["start"] + item["len"]]
            for item in data["sentence_list"]
        ]
        index_list = [i for ii in index_pair_list for i in ii]
        return [
            data["text"][index_list[i] : index_list[i + 1]]
            for i in range(len(index_list) - 1)
        ]

    def init(self):
        self.host_url = "https://transmart.qq.com"
        self.api_url = "https://transmart.qq.com/api/imt"
        _ = self.proxysession.get(self.host_url).text

    def translate(self, content):

        client_key = self.get_clientKey()
        api_headers = {"Cookie": "client_key={}".format(client_key)}

        split_form_data = {
            "header": {
                "fn": "text_analysis",
                "client_key": client_key,
            },
            "type": "plain",
            "text": content,
            "normalize": {"merge_broken_line": "false"},
        }
        split_data = self.proxysession.post(self.api_url, json=split_form_data).json()
        text_list = self.split_sentence(split_data)

        api_form_data = {
            "header": {
                "fn": "auto_translation",
                "client_key": client_key,
            },
            "type": "plain",
            "model_category": "normal",
            "source": {
                "lang": self.srclang,
                "text_list": [""] + text_list + [""],
            },
            "target": {"lang": self.tgtlang},
        }
        r = self.proxysession.post(
            self.api_url, json=api_form_data, headers=api_headers
        )
        try:
            data = r.json()
            return "".join(data["auto_translation"])
        except:
            raise Exception(r)

    def langmap(self):
        return {Languages.TradChinese: "zh-tw"}
