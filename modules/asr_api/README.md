# ASR API — 远端语音识别

模块 ID：`org.obliviousmic.asr_api`，版本 1.1.2，Apache-2.0。
不下载模型或保存录音。将图中的语音段提交给用户选择的服务，输出最终识别文本。
与本地 **ASR Whisper**（原 ID `org.obliviousmic.asr`）是两个独立模块。

```text
麦克风／应用音频 → ASR API → 字幕／文本翻译／TTS
```

模块使用 16 kHz 单声道；框架自动转换输入格式。HTTP 请求运行在独立进程，
不占用设备回调；停止或超时可结束进程。准备图和健康检查不会试发真实或合成语音。

## 接口与最简配置

| 接口 | 配置 |
| --- | --- |
| OpenAI compatible | API Key；默认地址 `https://api.openai.com/v1`、模型 `whisper-1`。其他服务填写相应地址和模型 |
| Azure Speech | Speech Key、区域和识别语言；默认 `eastasia`、`zh-CN`。可填写完整自定义短音频识别端点 |
| 阿里云百炼 | API Key；默认北京、`qwen3-asr-flash`、自动语言。可选择新加坡，密钥须匹配地域 |

OpenAI 兼容接口是 **`/audio/transcriptions` 文件转写协议**，不等同于所有
支持 `/chat/completions` 的音频模型。Groq 等服务可填写兼容地址和对应模型。
支持根地址或完整转写端点，固定请求 JSON 文本结果；不调用翻译接口。
本地无鉴权 OpenAI 兼容服务可留空 Key。远端使用 HTTPS，HTTP 仅限本机回环地址。

Azure 使用独立的 Speech 服务，不是 Azure OpenAI；固定选择中文／英文／日文。
OpenAI 兼容接口可自动识别语言，也可指定中、英、日。

百炼使用 Qwen ASR 的 `/compatible-mode/v1/chat/completions` 音频消息协议，
不是文件转写的 `/audio/transcriptions`。WAV 在内存中转成 Base64 Data URL，
不创建远端文件或网盘分享。可自动识别语言或指定中文／英文／日文。
高级设置 `bailian_enable_itn` 开启中英文数字格式规范化，默认关闭。
`bailian_base_url` 默认留空；如使用业务空间专属域名，可填
`https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`，
新加坡对应 `ap-southeast-1`。也支持完整 `/chat/completions` 地址，仅允许百炼官方 HTTPS 域名。
所选模型需支持此同步 Qwen ASR 协议；不把实时 WebSocket 或异步文件转写模型混用于该接口。

## 分段与参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_segment_seconds` | 6 秒 | 连续语音达到该长度时提交；范围 1～12 秒 |
| `silence_ms` | 500 ms | 讲话后停顿达到此时间，提交当前段 |
| `minimum_speech_ms` | 200 ms | 语音少于此长度不提交，减少点击噪声请求 |
| `silence_dbfs` | -48 dBFS | 本地分段的能量阈值，不是模型 VAD |
| `request_timeout_seconds` | 15 秒 | 每次请求上限，2～30 秒；失败不自动重试 |

所有设置保存后，停止设备、验证并应用处理图，再启动。没有模型档案或烧录步骤。
静音只保留 200 ms 前置缓冲，不持续上传；长语音按段顺序识别，分段交界处可能影响词语。
输出是**每段的最终文本**，不是逐字实时增量；延迟包括分段等待、网络及服务处理。
时间戳对应本地发送的音频段范围，不声称逐字精确对齐。
输出类型为标准 `TextSegment`，可以直接接已有 Translate、Subtitles、TTS。

音频在内存中编码为 PCM16 WAV。只有实际处理语音段时才发送给当前配置的服务；
该服务可能收费或按其规则处理音频。认证失败、超时、限流或异常结果会显示明确错误，
不会自动重试、切换服务或写出识别内容到日志。密钥不放入进程命令行。

## 来源和许可证

参考 [SpeechRecognition](https://github.com/Uberi/speech_recognition)（BSD-3-Clause）
的 OpenAI／Azure 接口。本模块独立实现 HTTP 客户端，不捆绑上游库、服务器、模型或 GPL 源码。
百炼按[官方 Qwen ASR 协议](https://help.aliyun.com/zh/model-studio/qwen-asr-api-reference)独立实现，无供应商 SDK 或源码拷贝。
固定参考版本及官方协议文档见 `provenance.json`，许可证全文见 `LICENSE`，归属说明见 `NOTICE`。

2026-10-29：百炼协议及错误处理测试通过；使用本机系统合成的 3.26 秒英文，
完成北京地域 `qwen3-asr-flash` 的一次真实请求，返回有效识别文本；未使用麦克风。
新加坡／自定义业务空间端点经过模拟协议测试，未逐地域调用。

## 单次请求失败

已识别的服务请求错误（超时、限流、鉴权／参数拒绝及无效服务响应）只跳过当前请求的输出，
不关闭处理图或工作进程；下一次请求可继续处理。失败时不会透传原文冒充译文／识别结果。
主页监控显示原因和失败次数，后续成功会清除当前错误。默认不自动重试付费请求。
准备、IPC、音频契约和程序异常仍按故障处理。
需要同步更新的 ObliviousMic 框架（`sdk.RequestStatus` 与请求状态监控）。
