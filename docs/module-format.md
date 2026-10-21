# Module package format

Official Release installers use <module>-<version>.obmic.zip. GitHub Source ZIPs and development
code-only packages are not the full user installer. Large weight binaries never enter Git.

Each package contains manifest.json, settings.schema.json, actions.schema.json, setup.json,
resources.json, runtime code, LICENSE, NOTICE and README.md. A package containing permitted public
weights additionally includes models/, complete model licenses and upstream model cards under licenses/,
and licensing.md. Code and weight licenses remain independent.

Each bundled resource declares bundled_path, license_file, original source/model revision, license,
redistribution=allowed, exact bytes and SHA-256. Current audited payload licenses are MIT/Apache-2.0.
Unresolved or prohibited weights cannot be bundled or automatically downloaded. BSRNN's template
remains local-import only. User voice assets and credentials are never public package content.

The host verifies the catalog package digest/identity, resource metadata and every model digest before
executing package code. Models are imported into the content-addressed public cache and removed from
the installed code directory to avoid duplicate model storage. Removing/upgrading module code does not
remove shared models or private profiles. Missing public cache resources can still use their pinned
upstream download declaration. Declared Python/native dependencies are installed separately.

Build full installers with python -m tools.build_module asr (or another module name).
Use --cache PATH --offline for local verified-cache builds, and --code-only only for development.
Prepare all release installers and a catalog candidate with python -m tools.prepare_catalog.
Validate the exact artifacts before publishing; publish and verify Release assets before promoting
catalog/index.json. Module tags and versions are independent.

Initialization runs on the control plane, never the audio callback. The minimum framework is the
matching ObliviousMic 0.3.0rc1 development build supporting bundled resources and large packages.
