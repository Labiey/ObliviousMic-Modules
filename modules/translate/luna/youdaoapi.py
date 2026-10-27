# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
import time
import hashlib
from obmic_luna_compat_v1_1 import basetrans
import uuid
from obmic_luna_compat_v1_1 import Languages


class TS(basetrans):
    def langmap(self):
        return {Languages.Chinese: "zh-CHS"}

    def translate(self, content):
        self.checkempty(["APP_KEY", "APP_SECRET"])

        APP_KEY = self.multiapikeycurrent["APP_KEY"]
        APP_SECRET = self.multiapikeycurrent["APP_SECRET"]
        youdao_url = "https://openapi.youdao.com/api"

        translate_text = content
        if len(translate_text) <= 20:
            input_text = translate_text
        elif len(translate_text) > 20:
            input_text = (
                translate_text[:10] + str(len(translate_text)) + translate_text[-10:]
            )

        time_curtime = int(time.time())
        uu_id = uuid.uuid4()

        sign = hashlib.sha256(
            (APP_KEY + input_text + str(uu_id) + str(time_curtime) + APP_SECRET).encode(
                "utf-8"
            )
        ).hexdigest()
        data = {
            "q": translate_text,  # 翻译文本
            "from": self.srclang,  # 源语言
            "to": self.tgtlang,  # 翻译语言
            "appKey": APP_KEY,  # 应用id
            "salt": uu_id,  # 随机生产的uuid码
            "sign": sign,  # 签名
            "signType": "v3",  # 签名类型，固定值
            "curtime": time_curtime,  # 秒级时间戳
        }
        r = self.proxysession.get(youdao_url, params=data)  # 获取返回的json()内容
        try:
            return r.json()["translation"][0]
        except:
            raise Exception(r)
