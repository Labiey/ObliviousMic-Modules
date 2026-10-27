# Provenance and modifications

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

The translate module is original Apache-2.0 code implementing Azure Translator v3 and
OpenAI-compatible chat completions from the linked official protocol documentation.
LunaTranslator revision 856ec4d4f270b59dab489120a702ed93ee254148 (GPL-3.0) was read as a
conceptual reference for engine separation. No GPL source, assets, provider SDK or model
is copied into translate. See modules/translate/NOTICE for exact reference URLs.
