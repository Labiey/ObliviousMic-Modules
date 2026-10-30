# TTS — 云端语音合成

模块 ID：`org.obliviousmic.tts`，版本 1.3.0，Apache-2.0。

```text
文本输入／ASR／翻译 → TTS → 音频设备输出
                       → 字幕（原文 text 输出）
```

## 配置

仅保留两个云端后端，默认选择百炼；填写密钥后保存设置、验证并应用图。

| 后端 | 最简配置 |
| --- | --- |
| 阿里云百炼（qwen） | API Key、匹配的北京／新加坡地域；默认模型和音色可直接使用 |
| Azure Speech | Speech Key、区域；可选语言、音色、语速与音调 |

### 百炼模型接口

按模型名称自动选择协议，无需手选 HTTP／WebSocket：

- `qwen3-tts-flash` 等原有模型使用 HTTP，默认音色 `Cherry`。
- `qwen-audio-3.1-tts-flash`／`qwen-audio-3.1-tts-plus` 及 Qwen-Audio 3.0 TTS 使用 WebSocket 任务接口。
  音色留空时 3.1 使用 `longanhuan_v3.1`，3.0 使用 `longanhuan`。
  已填写的系统／复刻音色原样使用，不自动替换。不同模型的音色不可混用。
- 高级参数 `qwen_workspace_id` 可选，用于业务空间专属域名；留空使用对应地域的标准域名。
- WebSocket 接收 24 kHz、16-bit 单声道 PCM，在内存中校验完整任务后交给现有语句播放队列。
  限制总时长、音频字节数、事件数与单次响应大小；中途失败不播放不完整语音，也不自动重试。
- 依赖 `websockets>=15,<17`（BSD-3-Clause），由 Python 依赖安装，不把库代码捆绑进模块 ZIP。

协议依据：[接入流程](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-websocket-api)、
[客户端事件](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-client-events)、
[服务端事件](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-server-events)、
[音色列表](https://help.aliyun.com/zh/model-studio/qwen-audio-tts-voice-list)。客户端为独立实现，未复制示例代码。

语言支持自动判断／中文／英文／日文。Azure 语速和音调均为倍率，默认 1，范围 0.5～2。
输出为 24 kHz 单声道 PCM，框架按后续端口转换格式；text 输出透传原文。
不需要模型、声音档案或参考音频。1.2.0 已移除本地合成后端和自动启动服务的实现。
升级时如原参数表使用已移除的后端，请重新选择百炼或 Azure 并配置密钥，
不会自动把原来的本地语音任务切换为付费云调用。

只有最终且非空的文本段会请求服务；同一段不重复合成。网络请求运行在独立进程，
准备和健康检查不发送测试请求。超时／停止可结束工作进程，不自动重试或切换供应商。
密钥通过有界 IPC 传递，不放在命令行。生成音频在内存中传递，不写入仓库。

## 来源与许可证

原始 Apache-2.0 客户端代码，包含 LICENSE 和 NOTICE，不捆绑模型、供应商 SDK 或密钥。
所选云服务、音色和生成内容的使用另受供应商条款约束。
协议测试覆盖两个后端、Azure SSML／语速音调、音频校验、错误脱敏、文本去重和音频时钟。
