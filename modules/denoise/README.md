# DPDFNet 通用语音降噪

模块 ID：`org.obliviousmic.denoise`，版本 **1.0.0**，Apache-2.0。
用于降低风扇、空调、键盘等环境噪声，保留人声。无需声纹登记，
不会根据说话人身份关门；也不保证去掉另一人的讲话。音乐可能被削弱。

## 安装与连接

完整 `denoise-1.0.0.obmic.zip` 包含已核对许可的两份官方 ONNX 权重。
安装器验证 SHA-256 后导入公共资产缓存，自动填充资源引用；无须填写模型路径。
Git 和开发用 `--code-only` 包不包含权重。代码包首次准备会从固定上游下载。
需要 NumPy 2 和 ONNX Runtime CPU 1.20+，可使用本体的 `models` 可选依赖。

```text
麦克风／应用音频 → DPDFNet 降噪 → Loudness／ASR／音频输出
```

可与 BSRNN 串联，但应试听是否过度处理。图中的每个降噪节点保持独立状态。
模块工作于 **48 kHz、单声道**；框架自动转换其他采样率和声道。
立体声会合为单声道，需要保留立体声空间感时不应串入此模块。

## 参数

| 参数 | 默认值 | 用法 |
| --- | --- | --- |
| `mode` | `light` 轻量 | DPDFNet2 48k HR，优先降低 CPU 占用 |
| `mode` | 可选 `enhanced` 增强 | DPDFNet8 48k HR，使用更大模型，CPU 占用更高；实际音质取决于声音 |
| `mix` | `1` | 0～1；1 为全部降噪声音，0 为原声，中间值混合原声。可实时修改；0 仍保留相同延迟和推理负载 |
| `threads` | `1` | 高级选项，1～16 个 ONNX CPU 线程；更多线程不一定更快 |

模式和线程数修改后，停止设备、保存设置、验证并应用处理图，再启动。
混合比例可用“保存并更新实时参数”。模型资源由软件管理，不提供任意路径输入。

20 ms 分析窗、10 ms 步长；另有 4 个步长的模型内部频谱延迟。
模块消除输出前导填充并保留正确媒体时间，向框架报告 **50 ms 算法缓存延迟**；
实际端到端延迟还包括设备、队列、推理和其他模块。EOS 排空到准确输入长度。
离线和实时共用同一套因果处理，不使用离线居中反射填充。

## 来源与许可

- [官方源代码](https://github.com/ceva-ip/DPDFNet/tree/9bd9844a227bb6aa57e55588d8d0e961fcff1c46)：Apache-2.0。
- [官方模型卡](https://huggingface.co/Ceva-IP/DPDFNet/blob/dd6818d00f50c836fed43a6243ebe49116de5964/README.md)：声明 `license: apache-2.0`，同时列出两份 48 kHz HR ONNX。
- 轻量 `dpdfnet2_48khz_hr.onnx`，增强 `dpdfnet8_48khz_hr.onnx`；固定版本、大小、SHA-256 见 `resources.json`。
- 许可证全文、归属和修改说明见 `LICENSE`、`licenses/dpdfnet/LICENSE`、`NOTICE`、`provenance.json`。

不打包训练数据、录音、个人声纹或供应商二进制运行库。
