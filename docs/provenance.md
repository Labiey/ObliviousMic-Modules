# Provenance and modifications

DPDFNet denoise adapts the CPU session/state loading, Vorbis STFT and causal
streaming code in ceva-ip/DPDFNet's package/src/dpdfnet/{onnx_backend,stream,audio}.py
at revision 9bd9844a227bb6aa57e55588d8d0e961fcff1c46 (Apache-2.0).
Changes and model provenance are recorded in modules/denoise/provenance.json
and NOTICE. Unlike upstream StreamEnhancer, it compensates four spectral
delay hops, supplies causal left padding, drains to exact input length and
aligns the original waveform before dry/wet mixing. It uses no librosa or
PyTorch and lets the host handle format conversion. The pinned enhanced
ONNX mistakenly reports profile dpdfnet2_48khz_hr; the module also checks the
distinct state sizes (56436 / 90228) and managed model digests.


The source baseline is ObliviousMic commit fa4620e and the independently reusable leaves of the
user-supplied Apache-2.0 OwnerMic snapshot identified in licensing.md. Exact copied-source hashes
are in provenance.json. This is adapter/DSP source redistribution, not checkpoint redistribution.

Campplus: features.py (NumPy fbank), identity_stream.py (sliding observations and embeddings),
identity_seam.py (observation contracts), sessions.py (explicit runtime selection),
and presence.py (unchanged identity EMA/hysteresis, moved from core Gate in 1.3.0).
BSRNN: the same identity helpers for one-step setup, plus frame_glue.py (NumPy STFT/OLA),
onnx_frame_backend.py (explicit recurrent state), excerpt.py (qualified voice excerpt), bake.py
(binding an enrollment tensor into a user-authorized template).

New/modified distribution code: runtime.py splits SDK adapters into independent packages;
lib/base.py contains shared explicit loading; actions.py combines enrollment and local baking;
JSON declarations define defaults, resource roles, setup and licensing; tools build/validate
code-only releases. Runtime/EOS and shape safety checks remain from the from-scratch framework.
Original attribution headers/docstrings are retained, and added modification notices are explicit.
The package build additionally includes provenance and license documentation in each module archive.

Historical performance claims in inherited docstrings describe the frozen project; they are not
new measurements, endorsement or performance guarantees. Validation uses synthetic correctness
fixtures and does not rerun hardware performance baselines.

The later asr/tts/subtitles/broadcast directories are original modules, not copied OwnerMic leaves.
They use the host SDK, separately installed dependencies and documented provider interfaces; their
per-module NOTICE identifies those sources. No provider implementation, installed vendor program,
weight or private enrollment is copied into their archives. The candidate catalog builder preserves
published CAM++/BSRNN entries and records independently rebuilt hashes for the four additions.

Translate 1.0.0 originally implemented only Azure/OpenAI-compatible protocols under Apache-2.0.
Translate 1.1.0 additionally reuses 24 LunaTranslator engine leaves from revision
856ec4d4f270b59dab489120a702ed93ee254148. The combined module is distributed under
GPL-3.0-only, with the upstream full license, original source attribution, exact source
and adapted LF-normalized SHA-256 hashes and per-file modification records in
modules/translate/LUNA_PROVENANCE.json. The pre-existing Apache adapter headers and full
Apache license are preserved. No Qt GUI, browser automation, model weights, user recordings
or credentials are bundled. This per-module exception does not replace other modules' licenses.
