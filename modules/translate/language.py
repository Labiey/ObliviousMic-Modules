# SPDX-License-Identifier: Apache-2.0
"""Conservative local language detection, loaded only for unlabeled input."""

import unicodedata


class LanguageDetector:
    def __init__(self):
        self.identifier = None

    def detect(self, text):
        # Numbers, names and very short messages are poor evidence for bypassing an API.
        letters = [c for c in text if unicodedata.category(c).startswith("L")]
        cjk = sum("\u3040" <= c <= "\u30ff" or "\u3400" <= c <= "\u9fff" for c in letters)
        latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
        if cjk and latin and min(cjk, latin) / len(letters) >= 0.2:
            return "auto"
        if len(letters) < (4 if cjk else 12):
            return "auto"
        if self.identifier is None:
            from langid.langid import LanguageIdentifier, model

            self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        language, confidence = self.identifier.classify(text)
        # Keep all 97 candidates: restricting to the target would force false matches.
        if confidence < 0.95:
            return "auto"
        if language == "zh":
            import hanzidentifier as hanzi

            script = hanzi.identify(text)
            return {
                hanzi.SIMPLIFIED: "zh",
                hanzi.TRADITIONAL: "zh-Hant",
                hanzi.BOTH: "zh-both",
            }.get(script, "auto")
        return language
