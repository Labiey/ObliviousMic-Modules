# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import Languages
from html import unescape
from obmic_luna_compat_v1_1 import basetrans


class TS(basetrans):

    def init(self):
        _ = self.proxysession.get("https://www.translate.com/machine-translation")

    def translate(self, content):

        from_language = self.srclang
        if self.is_src_auto:
            detect_form = {"text_to_translate": content}
            r_detect = self.proxysession.post(
                "https://www.translate.com/translator/ajax_lang_auto_detect",
                data=detect_form,
            )
            from_language = r_detect.json()["language"]

        form_data = {
            "text_to_translate": content,
            "source_lang": from_language,
            "translated_lang": self.tgtlang,
            "use_cache_only": "false",
        }
        r = self.proxysession.post(
            "https://www.translate.com/translator/translate_mt", data=form_data
        )
        try:
            data = r.json()
            return unescape(data["translated_text"])
        except:
            raise Exception(r)

    def langmap(self):
        return {Languages.TradChinese: "zh-TW"}
