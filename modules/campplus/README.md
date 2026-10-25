# campplus

Optional official module for ObliviousMic Module API 1.0.
Install the official .obmic.zip package through the catalog or verified local ZIP import. Models live in the local managed asset cache;
private registration/baking output belongs to the selected native configuration profile.

Normal settings expose the profile. Paths, model hashes and inference tuning are advanced options.
Defaults are ONNX Runtime / CPU / one inference thread. No unavailable backend is silently substituted.

CAM++ prepares its declared public model with hash verification. Initialization requires at least 3 seconds of suitable speech and creates a normalized speaker embedding.

Source: frozen OwnerMic Apache-2.0 leaves, then ObliviousMic adapters. See LICENSE, NOTICE and ../../docs/provenance.md. Private artifacts and raw voice are never published.

## Release payload policy (2026-10-22)

The release includes the Apache-2.0 public CAM++ speaker model and its upstream license evidence in licenses/campplus/.
Git source contains no weight binaries. Source ZIPs and development code-only bundles are not the full Release installer.

## 1.3.0: identity policy belongs to CAM++

The decision port now has kind `control` and emits a boolean every 10 ms, including
repeated values. Connect `CAM++.decision` to `Gate.control`. Use the matching updated
framework; older frameworks do not support the `apply_next` late-control policy.
No expiration field or extra graph port is added. Raw similarity, veto, smoothed
confidence and decision reasons remain available through `report_monitor()` only.

Normal parameter tables expose open/close thresholds (0.55/0.35), open/close condition
holds (10/200 ms), and minimum open/closed times (300/200 ms). Advanced fields retain
EMA, fast-open, veto and silence protection. All identity policy fields support hot
updates without loading the model again; model/window settings still need reprepare.
Close threshold must be below open threshold. Effective fast-open score is capped at
the open threshold; effective veto score stays strictly below the close threshold.
Fast-open energy now measures CAM++'s input, since this module does not receive the
separated waveform. Threshold scores are similarities, not probabilities.

Gate manages audio fades and optional `control_timeout_ms` (0 disables). Repeated
commands renew its receive timer. For onset protection put Delay between BSRNN and
Gate and specify `delay_ms`; output latency increases by the same amount.

When upgrading an existing graph, replace its old CAM++ node with the new module,
connect decision to Gate.control and choose a new Gate parameter table. Move any
custom identity thresholds from the old Gate table into CAM++ before applying.
Shared voice/model archives remain available; re-enrollment is not needed.
