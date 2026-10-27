# SPDX-License-Identifier: GPL-3.0-only
# Adapted from HIllya51/LunaTranslator, revision 856ec4d4f270b59dab489120a702ed93ee254148.
# Original authors: LunaTranslator contributors. Modifications: ObliviousMic contributors, 2026-10-27.
# See ../LUNA_PROVENANCE.json for exact source and changes.
from obmic_luna_compat_v1_1 import basetrans
import time
import base64
import random
import hashlib
import hmac
from obmic_luna_compat_v1_1 import Languages


def get_string_to_sign(method, endpoint, params):
    s = method + endpoint + "/?"
    query_str = "&".join("%s=%s" % (k, params[k]) for k in sorted(params))
    return s + query_str


def sign_str(key, s, method):
    hmac_str = hmac.new(key.encode("utf8"), s.encode("utf8"), method).digest()
    return base64.b64encode(hmac_str)


class TS(basetrans):
    def langmap(self):
        return {Languages.TradChinese: "zh-TW"}

    def trans_tencent(self, q, secret_id, secret_key, fromLang="auto", toLang="en"):

        endpoint = "tmt.tencentcloudapi.com"
        data = {
            "SourceText": q,
            "Source": fromLang,
            "Target": toLang,
            "Action": "TextTranslate",
            "Nonce": random.randint(32768, 65536),
            "ProjectId": 0,
            "Region": [
                "ap-beijing",
                "ap-shanghai",
                "ap-chengdu",
                "ap-chongqing",
                "ap-guangzhou",
                "ap-hongkong",
                "ap-mumbai",
                "ap-seoul",
                "ap-shanghai-fsi",
                "ap-shenzhen-fsi",
                "ap-singapore",
                "ap-tokyo",
                "ap-bangkok",
                "eu-frankfurt",
                "na-ashburn",
                "na-siliconvalley",
                "na-toronto",
            ][
                self.config["Region"]
            ],  # https://cloud.tencent.com/document/api/551/15615#.E5.9C.B0.E5.9F.9F.E5.88.97.E8.A1.A8
            "SecretId": secret_id,
            "SignatureMethod": "HmacSHA1",
            "Timestamp": int(time.time()),
            "Version": "2018-03-21",
        }
        s = get_string_to_sign("GET", endpoint, data)
        data["Signature"] = sign_str(secret_key, s, hashlib.sha1)

        # 此处会实际调用，成功后可能产生计费
        r = self.proxysession.get("https://" + endpoint, params=data)
        # print(r.json())
        return r

    def translate(self, query):
        self.checkempty(["SecretId", "SecretKey"])

        appid = self.multiapikeycurrent["SecretId"]
        secretKey = self.multiapikeycurrent["SecretKey"]

        ret = self.trans_tencent(query, appid, secretKey, self.srclang, self.tgtlang)
        try:
            return ret.json()["Response"]["TargetText"]
        except:
            raise Exception(ret)
