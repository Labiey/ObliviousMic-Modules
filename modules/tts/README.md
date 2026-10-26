# Text to speech

Accepts private TextSegment v1 metadata and emits 24 kHz mono float audio plus the original text stream.
Only final, nonempty segments are spoken; repeated final IDs are deduplicated within a bounded window.
Generated audio has its own contiguous clock starting at zero, independent of source utterance duration.
A device host must schedule generated audio independently; connecting it to a capture-clock-only route is not sufficient.

## First use

- **Local GPT-SoVITS (default):** select a private profile and initialize one 3–10 second reference WAV/FLAC. Reference transcript and language can be supplied with the initialization action. The framework keeps the reference and prompt metadata in the private profile, outside the module package. The adapter discovers a compatible Windows portable installation in Downloads or D:/BaiduNetdiskDownload; an advanced directory override supports other locations.
- **Qwen Bailian:** select Qwen, supply a DashScope API key and its Beijing/Singapore region. Cherry and qwen3-tts-flash are defaults. Text is sent to the selected cloud provider; billing and provider terms apply.
- **Azure Speech:** select Azure, supply the Speech resource key and matching region. Chinese, English and Japanese voices are chosen automatically, with advanced voice/endpoint overrides. Text is sent to Azure; billing and provider terms apply.

Cloud providers require no reference profile. Secret inputs are marked as password fields. The adapter never logs keys, prompts, reference paths or provider error bodies, never retries billable synthesis automatically, and never forwards credentials to Qwen audio storage.

Azure supports `azure_rate` (0.5–2.0) and `azure_pitch` (0.5–2.0), both defaulting to 1.0 for the original voice. Values are multipliers: rate 1.2 means faster speech; pitch 1.2 requests +20% pitch. The WebUI offers numeric inputs only when Azure is selected. Save the parameter table, then validate and apply the graph. Default settings retain the original SSML without a prosody tag. Support can vary by voice; these controls do not add styles or role selection. Slower speech still obeys `maximum_audio_seconds`; use shorter input if the generated response exceeds it.

Azure SSML specification: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice#adjust-prosody

Local API calls are restricted to loopback. Existing compatible API services are reused and left running. When auto-starting, the adapter copies the installation YAML to a temporary directory and owns only the process it starts. Shared framework service leases keep it alive across graph replacement; the last owner releases it. The local runtime and model licenses remain separate from this adapter.

## Bounded preparation and processing

Local preparation performs a short, discarded synthesis to load lazy language models, in an isolated worker with a 93 second deadline (90 seconds for the request). This avoids the first live utterance timing out during cold initialization. No cloud warmup is sent. API process startup has its own 120 second bound. Normal synthesis defaults to 25 seconds, at most 300 characters and 30 seconds of returned audio, with configurable bounds. Request failure stops the node; there is no raw-audio fallback.

Audio is decoded and checked for finite normalized PCM, converted to mono and resampled to 24 kHz. Output uses reusable shared-memory IPC and 480-frame packets. A realtime dispatch consumes one utterance at a time. No generated audio or reference recording is bundled or committed.

## Validation and distribution

Contract tests cover all three providers, SSML escaping, audio validation, private error handling, deduplication, clock continuity and warmup deadlines. A real local GPT-SoVITS run synthesized Chinese, English and Japanese through the installed ZIP and framework DAG, verified text passthrough, process cleanup and unchanged original installation files.

User-authorized live Qwen (Beijing) and Azure Speech (eastasia) checks also passed for Chinese, English and Japanese through the installed ZIP, isolated worker and named offline Web API. Defaults were used for the model and voices, without a reference profile. Local Whisper independently read back each language and checked known phrase content. Job metrics, exported reports and generated QA configuration files were checked for credential leakage. These short calls verify the tested accounts and paths, not provider availability or latency guarantees. Earlier intermittent Japanese Qwen failures could not be diagnosed from the old generic worker error; later explicit checks passed. Requests are not retried automatically.

Provider failures use SDK PublicModuleError identifiers, preserving a controlled error_code across IPC and task records without exposing provider response bodies, keys, prompts or signed audio URLs. A matching current framework build is required for this SDK addition.

This module is currently a local development package; it is not yet in the published official catalog. Local GPT-SoVITS has also passed a real ASR → TTS → device playback/subtitle route with independent physical microphone capture. Cloud playback, broader device combinations and remote distribution have separate acceptance requirements. See the framework development decision record for current status.

## Licensing

This package contains original Apache-2.0 adapter code only; see LICENSE and NOTICE. It does not redistribute GPT-SoVITS, model weights, voice recordings, cloud SDKs or provider credentials. No license to a cloud service, third-party model or selected voice is inferred from the adapter license. API references are recorded in NOTICE.
