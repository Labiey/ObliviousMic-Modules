# Local ASR

CPU int8 Whisper Small; Chinese, English and Japanese, auto detection by default.
Install code, then prepare the four verified managed model files. No voice enrollment or model paths required.
Audio input is 16 kHz mono (framework adapts formats); text output uses TextSegment v1 metadata.
Silence boundaries finalize utterances; long speech is segmented within the configured bounded window.
This is utterance streaming, not word-by-word partial decoding; timestamps refer to source media.
Processing is asynchronous in realtime graphs, and EOS drains the remaining voiced segment.
No audio is uploaded; transcript packets are private by default and are not written to logs.
Requires the framework's named text endpoints and quantized resource reservations.
