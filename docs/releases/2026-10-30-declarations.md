# Unified package declarations

All nine maintained module sources now use `obliviousmic.module.json` as their
only package entry. Stable IDs and processing behavior remain unchanged. Each
module advances one patch version because its package bytes change. Existing
graphs are not automatically repinned. This format requires the matching host
development build; there is no legacy entry reader before first release.

Each declaration contains names/descriptions in Chinese, English and Japanese,
author, license, repository, keywords, framework/system/architecture constraints,
and all previous execution metadata. Settings/actions/setup/resources remain
separate definitions. Public model provenance and redistribution restrictions
are unchanged; the unresolved BSRNN separator template remains local-import only.

`tools.prepare_catalog` builds reviewable deterministic candidates and registers
explicit external declaration paths. The published `catalog/index.json` is not
changed by a build. Publish immutable Release ZIPs manually, verify uploaded
bytes/digests, then write their external declarations with verified GitHub
repository/release/asset IDs and promote the catalog. External declarations
contain final ZIP digests; internal declarations do not.

The existing CI retains metadata validation, without automatic official package
building/publication. No new PR checker or automatic merge workflow is introduced.
The separate daily community catalog workflow is independent of official releases.

The nine patch releases have been manually published and their installer assets
verified against the audited size and SHA-256. `catalog/declarations/*/` contains
their complete external declarations, bound to repository ID `1418766723` and
the actual GitHub Release/asset IDs. The promoted official index and the host's
bundled snapshot carry these same declarations. Private voice assets and the
unresolved BSRNN separator template are not distributed.
