# Licensing and redistribution audit

## ASR API (checked 2026-10-28)

Original Apache-2.0 REST client, PCM/WAV segmentation and process adapter.
Reviewed SpeechRecognition (BSD-3-Clause), revision
8075e8f76acdac58fcf6ac41a104172f51349e98, to check public request
formats and response handling. No source from that project is redistributed;
this module interoperates with their documented protocols. No GPL LunaTranslator
code, provider SDK, model, recording or credential is included. Its NOTICE and
provenance.json identify exact references. Service/model terms remain separate.
The pre-existing ASR package is displayed as ASR Whisper with unchanged ID and
package bytes; renaming does not relicense or duplicate its MIT model payloads.


## DPDFNet denoise (checked 2026-10-28)

The official Ceva-IP/DPDFNet Hugging Face model card explicitly declares
`license: apache-2.0` at revision dd6818d00f50c836fed43a6243ebe49116de5964:
https://huggingface.co/Ceva-IP/DPDFNet/blob/dd6818d00f50c836fed43a6243ebe49116de5964/README.md.
Both dpdfnet2_48khz_hr.onnx and dpdfnet8_48khz_hr.onnx are from that same
official model repository, not a third-party conversion. Their sizes and
SHA-256 digests are pinned in modules/denoise/resources.json.

The corresponding source revision 9bd9844a227bb6aa57e55588d8d0e961fcff1c46
at https://github.com/ceva-ip/DPDFNet includes the full Apache-2.0 LICENSE;
no upstream NOTICE was present. The integration retains that license,
identifies adapted files and changes, and adds attribution in its NOTICE.
Full installers may redistribute these two weights with that license and
provenance; Git and code-only archives omit binaries. NumPy and ONNX Runtime
remain separately installed dependencies. No training dataset is bundled.


Checked 2026-10-22 (initial code provenance audit: 2026-10-14). Code, model weights and dependencies are separate licensed works.
The root LICENSE and per-module LICENSE apply to this repository's original adapters and the
Apache-2.0 OwnerMic source supplied by Labiey. They do not assert an Apache license for every weight.

## Adapted code

Source: user-supplied frozen OwnerMic-main.zip, SHA-256
771e8d9c0fc4714dc8ad96f2272cd9499bcd5259eb244aedccc7e29dbc533935.
The supplied OwnerMic root LICENSE is Apache-2.0. No other LICENSE/NOTICE file was found in that
supplied source tree. Reused headers/docstrings remain; each adapted module carries a modification
notice, the complete license and NOTICE. Network architectures/checkpoint source from WeSep are
not bundled in these adapters. See provenance.md for file mapping and hashes.

Apache-2.0 redistribution requirements: retain license and applicable attribution/NOTICE,
identify changed files, and do not imply endorsement. Official text:
https://www.apache.org/licenses/LICENSE-2.0.html (section 4).

## CAM++ public model

Primary source model card, revision v1.0.0:
https://modelscope.cn/api/v1/models/iic/speech_campplus_sv_zh_en_16k-common_advanced/repo?Revision=v1.0.0&FilePath=README.md
The retrieved card explicitly declares `license: Apache License 2.0`.
The model authors are the 3D-Speaker / ModelScope iic project, not ObliviousMic.
Upstream code: https://github.com/modelscope/3D-Speaker (its Apache-2.0 license is separate evidence).

ONNX distribution/conversion source:
https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
File: 3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
Size: 28281164 bytes; SHA-256:
aa3cfc16963a10586a9393f5035d6d6b57e98d358b347f80c2a30bf4f00ceba2.
The downloaded format conversion is unchanged by this repository. Release builds fetch it from upstream, verify its pinned size and digest, and embed it in CAM++ and
BSRNN Release installers. The Git source and development code-only archives omit the binary. Its attribution, license and source remain in resource metadata
and the installed module's NOTICE/licensing documentation. A similarly named Chinese-only CAM++
model is a distinct artifact and is not silently substituted.

## BSRNN template — unresolved checkpoint redistribution

Recorded checkpoint source:
https://github.com/REAL-TSE/wesep-real-tse/tree/2a540977a348fbaa92e623210505430e2cec608d
The current upstream revision checked on 2026-10-14 is the same revision. The recursive tree has no
LICENSE or NOTICE file. Individual Apache source headers do not independently identify the terms
of downloaded pretrained weights. The README links a Google Drive archive; no separate checkpoint
grant has been established. A Hugging Face recollection or a CAM++ model card is not substituted
as evidence for this exact checkpoint.

Checkpoint avg_model.pt SHA-256 (from the earlier local audit):
97c01b79b0cf1a5b914a4a3bc1113c1864752ebdd320f528e77a53202447fef2.
The locally exported public template restores explicit recurrent state and exposes enrollment as input;
its conversion is not a new independent grant of rights in the separator weights.

This release distributes only independently licensed adaptation/preprocessing/inference glue.
It DOES NOT distribute, host, automatically download or label the checkpoint/template Apache-2.0.
Users may import a compatible template for which they have usage authorization. Imported metadata
states that redistribution rights are unverified. Private baked cores and voice references stay local.
To enable a public default template later, record a precise upstream checkpoint license/authorization,
pinned artifact provenance, conversion changes, attribution and permitted distribution scope.

## Dependencies

Declared separately, installed into the user's Python environment (not embedded in code ZIPs):
NumPy (BSD-3-Clause), ONNX (Apache-2.0), ONNX Runtime (MIT), optional OpenVINO (Apache-2.0).
Use the installed distribution's own license files for its bundled third-party notices. Dependency
versions and resource hashes do not override their license terms. The host's audio resampling and
frontend dependency notices remain in the main project.

No user recording, speaker embedding, personalized ONNX core or dataset is redistributed.

## New module candidates (checked 2026-10-21)

ASR, TTS, subtitles and Broadcast contain original Apache-2.0 adapter code. Each archive contains
the complete adapter LICENSE, its own NOTICE, source and resource declarations. Release ASR packages
include the public MIT model resources below; no third-party runtime, reference audio, enrollment,
cloud key or generated speech is included.

ASR imports faster-whisper and CTranslate2 as separately installed MIT dependencies. The pinned
Systran/faster-whisper-small revision 536b0662742c02347bc0e980a01041f333bce120 identifies its model
as MIT. The model is fetched by the build, verified and embedded in the ASR Release installer.
The four resources retain upstream URLs, exact sizes, SHA-256 and MIT attribution in resources.json.
Sources: https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE and
https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120.

TTS 1.2.0 uses original REST adapters for Qwen/Azure only. Service and selected voice terms
remain separate. The adapter does not redistribute cloud SDKs, models or voice references.
Successful tests with a selected cloud account do not grant redistribution rights.

Subtitles uses the user's Python/Tk runtime and original Windows window-control glue. It does not
bundle Python, Tcl/Tk or another project's GUI code. Their installed runtime notices remain applicable.

Broadcast uses an already installed application's documented local gateway. No NVIDIA application,
driver, SDK or model is distributed. The Apache license covers only this adapter, and makes no claim
about NVIDIA component licensing or endorsement. Consult each module's NOTICE for API references.


## Release payload audit (2026-10-22)

The CAM++ model card at ModelScope revision v1.0.0 was retrieved again and explicitly declares
Apache License 2.0; its unchanged ONNX bytes match the pinned sherpa-onnx artifact above.
The pinned Systran model card explicitly declares MIT. ASR includes the original Whisper MIT text,
Copyright (c) 2022 OpenAI, and attributes the SYSTRAN CTranslate2 conversion in NOTICE.
Copied model cards and full license texts are retained under each affected module's licenses/ tree.

Evidence SHA-256:
- CAM++ model card: 5cc4c68f2f5afc255564493686f2a22370ac86f86ba02117989fc5a39a6aff0a
- Systran pinned model card: 329373481008c7c38654aff8ecdcf0163c211557cc7ba8e2ef6f2f84b4f75ec8
- Original Whisper MIT license: b5d65a59060e68c4ff940e1eddfa6f94b2d68fdf58ed7f4dd57721c997e35e9d
  Source: https://github.com/openai/whisper/blob/main/LICENSE

CAM++ 1.2.0 and BSRNN 1.2.0 include only the licensed public speaker model (28,281,164 bytes).
BSRNN's separator template is still NOT redistributed; a local test copy is not a public license grant.
ASR 1.0.0 includes all four public Whisper resources (486,212,372 bytes total), unchanged.
TTS, subtitles and Broadcast packages have no model payloads. NVIDIA software/models
remain external prerequisites; their local availability is not permission for public redistribution.
Private baked cores, recordings, speaker embeddings and API keys remain excluded from every package.

Resource declarations distinguish code license from weight license, identify original source/model revision,
conversion source, exact bytes/hash, bundled path and full model license file. Installation verifies these
before importing the payload into the host's shared public asset cache. Git never stores weight binaries.
