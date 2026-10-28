# ObliviousMic Modules

Official optional modules and distribution catalog for [ObliviousMic](https://github.com/Labiey).
Framework code and basic audio modules remain in the main project. This repository owns the
optional adapters, module-local reusable algorithms, initialization and package declarations.

| Module | Installation | First use | Weight license |
| --- | --- | --- | --- |
| [DPDFNet Denoise](modules/denoise/README.md) | Installer includes both pinned ONNX models | Light / Enhanced, 48 kHz CPU; no enrollment or manual paths | Official Ceva-IP model card and source: Apache-2.0 |
| CAM++ | Release installer includes verified public ONNX | Select/create a profile and provide at least 3 seconds of suitable speech | Upstream ModelScope v1.0.0 declares Apache-2.0; download from sherpa-onnx, SHA-256 pinned |
| BSRNN | Release installer includes public CAM++ only | Import an authorized compatible template once; provide suitable speech (at least 4 seconds); registration and baking run together | Current REAL-TSE checkpoint permission is unresolved. No weights, automatic download or checkpoint redistribution is supplied |
| ASR Whisper | Release installer includes all four pinned Whisper resources | CPU/int8 defaults; automatic Chinese/English/Japanese detection | Whisper conversion model card: MIT; faster-whisper/CTranslate2: MIT |
| [ASR API](modules/asr_api/README.md) | Client-only installer, no models | OpenAI-compatible key/base/model, Azure Speech key/region or Bailian key/region | Original Apache-2.0 client; selected service and model terms remain separate |
| TTS | Release installer | Qwen/Azure key and region; no local synthesis backend or reference initialization | Adapter Apache-2.0; cloud, model and voice terms remain separate |
| Subtitles | Release installer | Windows floating window, no voice profile or model | Original Apache-2.0 code; Python/Tk runtime supplied separately |
| Broadcast | Release installer | Compatible local NVIDIA Broadcast app; default buffered noise removal | Original Apache-2.0 adapter; no NVIDIA software or models redistributed |
| [Text Translation](modules/translate/README.md) | Release installer | Microsoft by default without a key; 13 web engines, 12 traditional API engines, one OpenAI-compatible interface | Combined module GPL-3.0; pinned LunaTranslator engine sources, no model weights |

Git contains no weight binaries. Release installers include only audited public CAM++/Whisper/DPDFNet resources.
Recordings, user embeddings, personalized models and unresolved BSRNN templates are excluded.
`LICENSE` applies to project code and Apache-licensed adapted OwnerMic leaves, except the GPL-3.0
translation module (see its own LICENSE/NOTICE), and not to all third-party
weights or dependencies. Read [licensing](docs/licensing.md) and [provenance](docs/provenance.md).

The local first-use and distribution tests require the same framework version; code packages are intended for installation through the framework catalog, not an implicit dependency on the main source tree.

Modules use stable IDs `org.obliviousmic.<name>` and Module API 1.0. The current installers require the matching current 0.3.0rc1 framework
with text streams, named endpoints, managed services and public module errors. For ASR, install the
framework's `speech` optional dependencies before preparing its model resources.

Build deterministic full installers: `python -m tools.build_module campplus` / `asr`.
Use `--code-only` only for development; those archives omit public model payloads.
Packages are published as GitHub Actions artifacts and per-module GitHub Release attachments.
Catalog entries pin package SHA-256 and size. `resources.json` identifies bundled model payloads, pinned fallback sources and separate weight licenses;
`setup.json` declares automatic bindings and first-use actions. [Package format](docs/module-format.md).

Prepare all nine modules in an ignored staging directory before publishing:

```powershell
python -m tools.prepare_catalog
python -m tools.validate_catalog --catalog dist/catalog.next.json
```

The generated candidate, model cache and ZIPs stay in ignored `dist/`. Model payloads are fetched/verified by the build, never committed. Review package licenses and acceptance
records, publish matching module releases, verify the attached bytes/digests, then promote the candidate
to `catalog/index.json`. This command does not publish, push, create tags or alter the current catalog.

Development requires an installed ObliviousMic SDK, plus `pip install -e .` and pytest.
Automated regression inputs are synthetic. Separate opt-in local acceptance has exercised real local
models, user-authorized cloud calls, a subtitle window and audio devices; those private artifacts and
credentials stay outside Git. See individual module READMEs and the framework's development record.
