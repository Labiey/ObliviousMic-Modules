# ObliviousMic Modules

Official optional modules and distribution catalog for [ObliviousMic](https://github.com/Labiey).
Framework code and basic audio modules remain in the main project. This repository owns the
CAM++ and BSRNN adapters, module-local reusable algorithms, initialization and package declarations.

| Module | Installation | First use | Weight license |
| --- | --- | --- | --- |
| CAM++ | Code package, verified public ONNX prepared automatically | Select/create a profile and provide at least 3 seconds of suitable speech | Upstream ModelScope v1.0.0 declares Apache-2.0; download from sherpa-onnx, SHA-256 pinned |
| BSRNN | Code package | Import an authorized compatible template once; provide suitable speech (at least 4 seconds); registration and baking run together | Current REAL-TSE checkpoint permission is unresolved. No weights, automatic download or checkpoint redistribution is supplied |

There are no recordings, user embeddings, personalized models or checkpoint weights in Git or code packages.
`LICENSE` applies to project code and Apache-licensed adapted OwnerMic leaves, not to all third-party
weights or dependencies. Read [licensing](docs/licensing.md) and [provenance](docs/provenance.md).

The local first-use and distribution tests require the same framework version; code packages are intended for installation through the framework catalog, not an implicit dependency on the main source tree.

Modules keep stable IDs `org.obliviousmic.campplus` and `org.obliviousmic.bsrnn` and use Module API 1.0.
The current setup/catalog integration requires ObliviousMic 0.2.0 or newer.

Build code-only deterministic packages: `python tools/build_module.py campplus` / `bsrnn`.
Packages are published as GitHub Actions artifacts and per-module GitHub Release attachments.
Catalog entries pin package SHA-256 and size. `resources.json` describes separate model resources;
`setup.json` declares automatic bindings and first-use actions. [Package format](docs/module-format.md).

Development requires an installed ObliviousMic SDK, plus `pip install -e .` and pytest.
Test inputs are synthetic; no physical devices or performance benchmarks are used.
