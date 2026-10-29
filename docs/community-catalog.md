# Community discovery and popularity snapshots

`catalog-refresh.yml` runs daily at **03:17 Hong Kong time** (`17 19 * * *` UTC)
and supports manual dispatch from the trusted default branch. Scheduled Actions
can be delayed or disabled by GitHub after prolonged inactivity; clients retain
their last valid snapshot and show its timestamp.

The collector reads only GitHub JSON/base64 metadata. It does not check out,
import, install, or execute third-party repositories. It discovers repositories
using `obliviousmic-module` and `obliviousmic-theme` Topics, freezes each repository
at a commit, reads exact root declaration filenames, and verifies repository
numeric ID/owner plus Release/asset IDs, URL, size and GitHub's digest when present.
The official multi-package repository uses explicitly registered declaration
paths. Author text is not publisher identity and listing is not a safety review.

Search pagination checks `incomplete_results`; more than 1,000 matches are split
into creation-time ranges. Request count, elapsed time, response size and
partition depth are bounded. Rate-limit retry headers produce bounded backoff.
An incomplete or rate-limited whole scan fails without publishing a replacement.
A transient per-repository failure retains its prior entry marked stale;
confirmed removal/unavailability stops its recommendation. Maintainers may edit
`catalog/exclusions.json` to exclude numeric repository IDs or exact
`repository_id/type/package_id` keys permanently.

Only explicitly declared installation asset IDs contribute to statistics.
Historical declared IDs are tracked across versions; models, screenshots and
unrelated Release assets are excluded. First observation is a baseline, never a
fabricated seven-day history. Ranking uses observed nonnegative download
increments within the last seven days, with coverage duration recorded. A
decreased/reset counter or changed asset identity is recorded, not counted as new
downloads. Downloads are not installation counts. New asset observations start
their own baseline.

The collect job has `contents: read`. A separate job with `contents: write`
revalidates the artifact and publishes only these files on **catalog-data**:

```
community-modules.json
community-themes.json
popular-modules.json
popular-themes.json
stats/downloads.json
status.json
```

Publication creates one commit for the whole snapshot with a non-force ref
update. Unexpected files or malformed/mixed snapshots fail before writes. The
workflow cannot edit the official identity catalog, source code or workflows.
The scoped token is supplied only to the relevant collection/publication step
and only sent to `api.github.com`; off-domain redirects are rejected. The official
package release process remains manual and PR merging remains a maintainer task.

Clients read all snapshot files by one fixed **catalog-data commit**, verify the
status checksums and snapshot IDs, and replace caches only after complete
validation. Local cached results remain searchable when the network is unavailable.

Protocol references: [GitHub repository search limits](https://docs.github.com/en/rest/search/search),
[Release asset metadata](https://docs.github.com/en/rest/releases/assets), and
[job-scoped token permissions](https://docs.github.com/en/actions/tutorials/authenticate-with-github_token).
