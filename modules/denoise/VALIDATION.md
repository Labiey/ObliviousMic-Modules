# Local acceptance — 2026-10-28

Both pinned official models were tested with ONNX Runtime CPU, one inference
thread, on the development Windows machine. These are local measurements,
not a cross-device performance guarantee.

| Mode | Audio tested | Per-10 ms call p50 / p95 / p99 | Mean real-time factor |
| --- | --- | --- | --- |
| Light | 30.003 seconds | 1.450 / 2.090 / 2.462 ms | 0.1568 |
| Enhanced | 30.003 seconds | 5.371 / 6.864 / 8.463 ms | 0.5317 |

The input was deterministic generated stationary noise. Both modes produced
finite samples with lower noise energy, preserved exact stream length and
drained the partial final hop. This fixture does not establish speech quality
or performance under a game load. Enhanced had a maximum single-call outlier
of 12.606 ms; buffering and available CPU headroom still matter.

Additional checks:

- Real ONNX output is invariant to input chunk boundaries when the same EOS
  boundary is used; mixtures 0 / 0.35 / 1 match media-aligned reference output.
- Full installer imports both SHA-256-verified payloads; managed bindings are
  ready without enrollment, model paths or any additional network requests.
- Both modes run through the real asynchronous DAG with automatic conversion
  from 16 kHz stereo to 48 kHz mono and back, including an odd-length EOS.
- Independent spectral-delay fixtures verify first/last sample reconstruction,
  exact 50 ms availability latency, dry/wet alignment, bounded buffers,
  per-node recurrent state, shared session release and discontinuity reset.
- A parallel dry/denoised Merge graph preserves continuous media positions
  without inserting an unnecessary physical delay into the dry branch.
- WebUI mode selection, saved parameter tables and automatic resource-field
  hiding were checked against an isolated local installation.

No microphone, private recording or cloud API was used. Human listening,
other hardware, hot-plug and game-load tests were not part of this acceptance.
