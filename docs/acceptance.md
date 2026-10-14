# Initial official distribution acceptance

2026-10-14. Framework 0.2.0; modules 1.1.2. No device or performance benchmark was repeated.

Both published release attachments match catalog SHA-256 and byte size. Tag workflow validation and
publication succeeded on GitHub Actions; package reconstruction passed on Ubuntu and Windows with
Python 3.12 and locally on Windows with Python 3.14. Portable path ordering, LF/UTF-8 text, fixed ZIP
metadata and ZIP_STORED avoid platform-dependent package identity.

Local module suite: 94 tests + 111 subtests; Ruff check/format and catalog validation passed.
These tests include the migrated frozen DSP/frame/layout parity suites, independent code bundle
installation/restoration, attribution files, and refusal to auto-download unresolved template resources.
Framework integration: 108 tests, frontend 7 tests, TypeScript/Vite build and isolated framework wheel audit.

Edge integration installed both actual GitHub release packages. CAM++ public model preparation checked
the upstream 28,281,164-byte ONNX against the declared SHA-256. One suitable synthetic WAV initialized
CAM++ in a new profile. BSRNN with an independently generated synthetic ONNX template initialized an
empty profile in one job, producing both embedding and private core. Automatic setup bindings compiled
and committed the private DAG; offline processing preserved all 112000 input frames including EOS.
Spectrum ABI checks reject an incompatible template's baked core before profile publication.
No real owner recording, private model or original checkpoint was uploaded to GitHub.

Chinese/English UI was checked; advanced parameters stay folded, desktop/mobile page widths fit the
viewport. Upload copies were removed; profile artifacts remain local. Main framework contains only basic
modules, while these packages contain no model weights, datasets or user data.

The BSRNN template redistribution decision remains unresolved. A recollection of a Hugging Face download
does not establish the license of the recorded exact checkpoint. This release contains Apache-licensed
adapter code and an authorized local import flow; it does not host or auto-download the separator weights.
See [licensing](licensing.md), [provenance](provenance.md) and [minimal setup](minimal-setup.md).
