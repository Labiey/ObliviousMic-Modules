# NVIDIA Broadcast integration (in development)

The verified MCP gateway client and private audio-file backend are implemented and tested. This directory is not yet an installable graph module: streaming buffering, delay alignment, manifests/settings and device-host acceptance remain in progress. It is not in the official remote catalog.

Gateway discovery reads the documented per-user base64 gateway.json on each new connection, validates the loopback port range and verifies serverInfo.name before sending any audio paths or effect settings. Requests bypass network proxies, reject redirects, bound replies and do not retry non-idempotent submissions. Provider errors exclude private paths and raw messages.

Effect choices follow live microphone capabilities: noise removal, room echo removal, their combination, or Studio Voice with an advertised microphone profile. Studio Voice is never silently combined with incompatible denoisers. Effects are specified per private job; live device/effect configuration is not rewritten. Existing Broadcast processes are not stopped by this backend.

FileProcessor creates an owned temporary input/output pair, checks the exact completed job and output path, validates decoded mono PCM and removes temporary media after reading. A paused queue, user cancellation, skipped effect, failure or timeout raises an explicit error. It never resumes the global queue, retries a cancellation, resubmits a failed job or returns raw input as an enhancement result. On failure while waiting, cancellation targets only this backend's exact owned job.

The result preserves backend frame count and timing. NVIDIA Broadcast 2.3.0 tests found padding beyond input duration and approximately 70 ms content delay for denoising/echo removal, 40 ms for Studio Voice. A graph adapter must compensate these deliberately and drain EOS; truncating output to input length is not sufficient.

Warm performance depends on batch length: on the development machine, 9.66 seconds took 1.83 seconds for denoising and 3.98 seconds for Studio Voice. A 2.35 second Studio Voice batch took 2.98 seconds, so short batches cannot sustain realtime input with that effect. These are actual local measurements, not a latency guarantee or proof of a finished live module. The streaming adapter must choose an appropriate bounded window per algorithm and declare its buffering latency.

Apache-2.0 applies only to this original adapter code. No NVIDIA binaries, models or drivers are packaged; see LICENSE and NOTICE. An already installed/running compatible Broadcast app is required for this backend. The separate native Audio Effects SDK is another documented integration path, but is not installed or bundled by this implementation.
