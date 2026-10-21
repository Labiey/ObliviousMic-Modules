# ObliviousMic Modules

Official optional modules and distribution catalog for [ObliviousMic](https://github.com/Labiey).
Framework code and basic audio modules remain in the main project. This repository owns the
optional adapters, module-local reusable algorithms, initialization and package declarations.

| Module | Installation | First use | Weight license |
| --- | --- | --- | --- |
| CAM++ | Code package, verified public ONNX prepared automatically | Select/create a profile and provide at least 3 seconds of suitable speech | Upstream ModelScope v1.0.0 declares Apache-2.0; download from sherpa-onnx, SHA-256 pinned |
| BSRNN | Code package | Import an authorized compatible template once; provide suitable speech (at least 4 seconds); registration and baking run together | Current REAL-TSE checkpoint permission is unresolved. No weights, automatic download or checkpoint redistribution is supplied |
| ASR | Candidate code package; pinned Whisper resources prepared separately | CPU/int8 defaults; automatic Chinese/English/Japanese detection | Whisper conversion model card: MIT; faster-whisper/CTranslate2: MIT |
| TTS | Candidate code package | Qwen/Azure key and region, or one local GPT-SoVITS reference initialization | Adapter Apache-2.0; cloud, model and voice terms remain separate |
| Subtitles | Candidate code package | Windows floating window, no voice profile or model | Original Apache-2.0 code; Python/Tk runtime supplied separately |
| Broadcast | Candidate code package | Compatible local NVIDIA Broadcast app; default buffered noise removal | Original Apache-2.0 adapter; no NVIDIA software or models redistributed |

There are no recordings, user embeddings, personalized models or checkpoint weights in Git or code packages.
`LICENSE` applies to project code and Apache-licensed adapted OwnerMic leaves, not to all third-party
weights or dependencies. Read [licensing](docs/licensing.md) and [provenance](docs/provenance.md).

The local first-use and distribution tests require the same framework version; code packages are intended for installation through the framework catalog, not an implicit dependency on the main source tree.

Modules use stable IDs `org.obliviousmic.<name>` and Module API 1.0. Published CAM++/BSRNN require
ObliviousMic 0.2.0 or newer. The four candidate modules require the matching current 0.3.0rc1 framework
with text streams, named endpoints, managed services and public module errors. For ASR, install the
framework's `speech` optional dependencies before preparing its model resources.

Build code-only deterministic packages: `python tools/build_module.py campplus` / `bsrnn`.
Packages are published as GitHub Actions artifacts and per-module GitHub Release attachments.
Catalog entries pin package SHA-256 and size. `resources.json` describes separate model resources;
`setup.json` declares automatic bindings and first-use actions. [Package format](docs/module-format.md).

Prepare the four new modules without advertising unpublished download links:

```powershell
python -m tools.prepare_catalog
python -m tools.validate_catalog --catalog dist/catalog.next.json
```

The generated candidate and ZIPs stay in ignored `dist/`. Review package licenses and acceptance
records, publish matching module releases, verify the attached bytes/digests, then promote the candidate
to `catalog/index.json`. This command does not publish, push, create tags or alter the current catalog.

Development requires an installed ObliviousMic SDK, plus `pip install -e .` and pytest.
Automated regression inputs are synthetic. Separate opt-in local acceptance has exercised real local
models, user-authorized cloud calls, a subtitle window and audio devices; those private artifacts and
credentials stay outside Git. See individual module READMEs and the framework's development record.
