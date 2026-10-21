# Licensing and redistribution audit

Checked 2026-10-14. Code, model weights and dependencies are separate licensed works.
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
The downloaded format conversion is unchanged by this repository. It is fetched from upstream
rather than embedded in code archives. Its attribution, license and source remain in resource metadata
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

ASR, TTS, subtitles and Broadcast contain original Apache-2.0 adapter code. Each code archive contains
the complete LICENSE, its own NOTICE, source and resource declarations. The packages contain no
third-party runtime, model, reference audio, enrollment, cloud key or generated speech.

ASR imports faster-whisper and CTranslate2 as separately installed MIT dependencies. The pinned
Systran/faster-whisper-small revision 536b0662742c02347bc0e980a01041f333bce120 identifies its model
as MIT. The model is downloaded from its upstream repository, not embedded or rehosted here.
The four resources retain upstream URLs, exact sizes, SHA-256 and MIT attribution in resources.json.
Sources: https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE and
https://huggingface.co/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120.

TTS uses original REST adapters for Qwen/Azure and a loopback client for an independently installed
GPT-SoVITS service. Service terms, selected voices, local runtime and model licenses are separate.
The adapter does not redistribute cloud SDKs, GPT-SoVITS, its models or the user's voice reference.
Successful tests with a selected local model or cloud account do not grant redistribution rights.

Subtitles uses the user's Python/Tk runtime and original Windows window-control glue. It does not
bundle Python, Tcl/Tk or another project's GUI code. Their installed runtime notices remain applicable.

Broadcast uses an already installed application's documented local gateway. No NVIDIA application,
driver, SDK or model is distributed. The Apache license covers only this adapter, and makes no claim
about NVIDIA component licensing or endorsement. Consult each module's NOTICE for API references.
