# 文本翻译模块 1.2.0

模块 ID：`org.obliviousmic.translate`。接在 ASR 或文本输入之后，输出可继续接 TTS、字幕或文本输出。
无需模型权重或声音档案。新版默认 Microsoft 免密钥接口，选择目标语言即可使用。

## 接口一览

| 类型 | 接口 |
| --- | --- |
| 大模型通用 | **OpenAI 兼容**：百炼、其他兼容云服务，以及本机 Ollama／LM Studio 都填写这里 |
| 免密钥网页接口（13 个） | Microsoft、必应 Bing、谷歌 Google、ModernMT、QQ TranSmart、阿里、有道、彩云、Translate.com、Yandex、QQ IMT、火山、Papago |
| 传统 API（12 个） | Azure Translator、百度、彩云、Google Cloud、DeepL、腾讯云、有道、Yandex Cloud、阿里云机器翻译、火山、华为云、小牛 |

没有独立的百炼大模型选项；**阿里网页翻译／阿里云机器翻译 API 是不同服务，不是百炼大模型**。
网页引擎由 LunaTranslator 原始实现复用而来，不需要启动 LunaTranslator 程序。
本模块未包含它的 OCR、Hook、浏览器自动化或需要专有 DLL 的本地翻译器。

## 连线

`文本输入.text → Translate.text → TTS.text → 音频设备输出`

`麦克风输入.audio → ASR.audio → Translate.text → 文本输出.text`

从 Translate 的 text 端口再接一路文本输出即可显示译文；从 ASR／文本输入另接一路可同时看原文。
原有文本输出的悬浮字幕功能可直接使用。默认仅翻译最终句子，避免部分识别反复请求。

## 配置方式

- **与目标语言相同时跳过翻译**：默认开启，可关闭。先采用手动指定的源语言；为 `auto` 时先看上游 ASR 的语言标记，再用本地检测判断未标注的文字。同语言时保留原文（包括空格和换行），继续送给 TTS／字幕，不调用翻译接口。
  本地检测置信度不足、太短、主要由数字组成或明显中西文混合时仍正常翻译。简体与繁体分别判断，需转换时不会跳过。语言识别并非绝对准确；需要始终请求服务时关闭此开关。
  本地检测使用 `langid` 和 `hanzidentifier`，无需额外下载模型或发送检测请求。初次使用新版前，在框架环境安装 `.venv/Scripts/python.exe -m pip install -e ".[translation]"`（框架目录）。
- **免密钥接口**：选择引擎和目标语言即可。默认 Microsoft。源语言 `auto` 优先采用 ASR 的语言，否则交给服务检测。
- **OpenAI 兼容**：填写 `base_url`、真实 `model` 名和 `api_key`。本机回环服务可不填 Key。
  百炼北京兼容地址示例：`https://dashscope.aliyuncs.com/compatible-mode/v1`，模型例如 `qwen-plus`；
  新加坡使用 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`。也可填写业务空间专用兼容地址。
  Ollama 示例：`http://127.0.0.1:11434/v1`，模型填写本机实际已安装的名称。
- **Azure Translator**：填写 `azure_api_key`，区域资源还需 `azure_region`，全局资源可留空。
  Speech/TTS 的 Key 不保证具有文本翻译权限。
- **传统 API**：只显示所选引擎需要的 Key／Token／Access Key ID、Secret 或 APP ID。
  百度支持通用、BCE、大模型三种接口；DeepL 可选 Free／Pro；腾讯云可选择区域名称。
  华为云当前沿用上游的 `cn-north-4` 区域实现。

每套参数表可选择不同引擎、语言和密钥，多个翻译节点可分别链接不同参数表。
保存设置、保存当前图、验证并应用，再启动会话。准备图只验证配置，所有引擎初始化均推迟到首次翻译。

## 从 1.0.0 更新

1.0.0 的 `qwen`、`qwen_api_key`、`qwen_region` 已删除。更新后使用新版默认参数表，或新建表。
需要百炼时选择 OpenAI 兼容，并填写上述兼容地址、模型和密钥。旧表不会被静默改写或丢弃密钥；
固定引用旧版本的图仍使用旧版本，需要显式升级节点并选择新版参数表。

## 行为、限制与验证

- 保留片段 ID、媒体起止时间、修订号和 final 标记，输出语言改成目标语言；重复、过期修订不会重新请求。
- 默认只翻译最终句子。关闭该选项后可逐次处理上游修订，每次仍返回完整片段，不是模型 token 流。
- 使用独立进程和异步队列；网络不会阻塞音频设备回调。默认单句 2000 字符、超时 25 秒。
- 网页接口和官方 API 的配额、语言覆盖不同。遇到验证码、限流或协议变化会明确报错，不会假装翻译成功。
  阿里网页、Yandex、Yandex Cloud、华为云等上游依赖繁简转换的引擎，当前不提供繁体中文目标转换；请选择其他引擎。
- 2026-10-27 本机使用同一合成测试句完成了全部 13 个免密钥接口的中译英真实请求，并完成 Microsoft 中译日独立进程测试。
  需要私有密钥的 12 个 API 有模拟协议/签名测试；未持有相应服务凭据，未逐个完成真实计费调用。
- 不写文本历史或磁盘翻译缓存。密钥由参数表私密存储和 IPC 传递，不进入命令行或公开错误。
  文本只发送到选定引擎；遇到跨域跳转、超大响应或超时会终止，不自动切换服务或重试计费请求。

## 许可证与来源

整体模块改按 **GPL-3.0-only** 分发，因为实际复用了 LunaTranslator 的 24 个引擎源码。
LICENSE 是完整 GPL-3.0 文本；NOTICE、LUNA_PROVENANCE.json 记录固定 revision、源文件及 SHA-256、修改说明。
原有 Apache-2.0 适配层的声明和许可证保留。ObliviousMic 主仓仍为 Apache-2.0。
不随包分发翻译模型、用户密钥或录音；服务条款仍以各供应商规定为准。
