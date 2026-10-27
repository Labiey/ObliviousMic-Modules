# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import Languages
import json
from html import unescape
from obmic_luna_compat_v1_1 import basetrans




# https://github.com/bookfere/Ebook-Translator-Calibre-Plugin/blob/master/engines/google.py


class TS(basetrans):
    def langmap(self):
        return {Languages.Chinese: "zh-CN", Languages.TradChinese: "zh-TW"}

    def translate_1(self, content):
        if not content:
            return ""
        response = self.proxysession.post(
            "https://translate-pa.googleapis.com/v1/translateHtml",
            headers={
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept-Language": "en-US,en;q=0.9",
                "Content-Type": "application/json+protobuf",
                "X-Goog-Api-Key": "AIzaSyATBXajvzQLTDHEQbcpq0Ihe0vWDHmO520",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 "
                "Safari/537.36",
            },
            data=json.dumps([[[content], self.srclang, self.tgtlang], "wt_lib"]),
        )
        try:
            return unescape(response.json()[0][0])
        except:
            raise Exception(response)

    def translate(self, content: str):
        return "\n".join((self.translate_1(_) for _ in content.splitlines()))
