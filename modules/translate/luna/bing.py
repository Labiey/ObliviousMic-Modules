# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
import json
import re
from obmic_luna_compat_v1_1 import basetrans
from obmic_luna_compat_v1_1 import Languages


class TS(basetrans):
    def init(self):
        host_html = self.proxysession.get("https://www.bing.com/Translator").text
        self.tk = self.get_tk(host_html)
        self.ig_iid = self.get_ig_iid(host_html)

    def get_ig_iid(self, host_html):

        iid = re.search(
            '<div[ ]+id="tta_outGDCont"[ ]+data-iid="(.*?)">', host_html
        ).groups()[0]
        ig = re.compile('IG:"(.*?)"').findall(host_html)[0]
        return {"IG": ig, "IID": iid, "isVertical": 1}

    def get_tk(self, host_html):
        result_str = re.compile("var params_AbusePreventionHelper = (.*?);").findall(
            host_html
        )[0]
        result = json.loads(result_str)
        return {"key": result[0], "token": result[1]}

    def translate(self, content):

        form_data = {
            "text": content,
            "fromLang": self.srclang,
            "to": self.tgtlang,
            "tryFetchingGenderDebiasedTranslations": "true",
        }
        form_data.update(self.tk)
        r = self.proxysession.post(
            "https://www.bing.com/ttranslatev3",
            params=self.ig_iid,
            data=form_data,
            allow_redirects=False,
        )
        if r.status_code == 302:
            r = self.proxysession.post(r.headers["Location"], data=form_data)
        try:
            data = r.json()
            return data[0]["translations"][0]["text"]
        except:
            raise Exception(r)

    def langmap(self):
        return {
            Languages.Chinese: "zh-Hans",
            Languages.TradChinese: "zh-Hant",
            Languages.Auto: "auto-detect",
        }
