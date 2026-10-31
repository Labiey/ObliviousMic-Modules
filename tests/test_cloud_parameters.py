"""Request-boundary switches preserve worker ownership, clocks and deduplication."""

import json
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import numpy as np
import pytest
from obliviousmic.adapters.python import PythonAdapter, State
from obliviousmic.sdk import (
    AudioFormat,
    Manifest,
    ModuleRegistration,
    PortBundle,
    PrepareContext,
    StreamContext,
    TextSegment,
)

from modules.translate.network import Module as TranslationWorker
from modules.translate.runtime import Module as Translation
from modules.tts.network import Module as SpeechWorker
from modules.tts.providers import TTSError
from modules.tts.runtime import Module as Speech
from tests.test_translate import config as translation_config
from tests.test_tts import config as speech_config


def text(identifier, language="en"):
    return PortBundle(
        {
            "text": (
                TextSegment(identifier, "Hello", Fraction(0), Fraction(1), language).to_packet(
                    "keyboard"
                ),
            )
        }
    )


def test_tts_switch_preserves_audio_clock_completed_ids_and_request_counters():
    calls = []

    class Client:
        def __init__(self, config):
            self.config = dict(config)

        def synthesize(self, *_args):
            calls.append(self.config)
            return np.ones((2400, 1), np.float32) * 0.1

    module = SpeechWorker(Client)
    module.prepare(PrepareContext("tts", {}, {"audio": AudioFormat(24000)}), speech_config())
    module.start(StreamContext())
    first = module.process(text("one"))
    module.update_parameters(
        {"provider": "azure", "azure_voice": "voice-two", "azure_rate": 1.4, "azure_pitch": 1.5}
    )
    assert not module.process(text("one")).ports["audio"]
    second = module.process(text("two"))
    assert first.ports["audio"][0].media_start == 0
    assert second.ports["audio"][0].media_start == Fraction(1, 10)
    assert second.ports["audio"][-1].media_end == Fraction(1, 5)
    assert calls[0]["provider"] == "qwen"
    assert calls[1]["provider"] == "azure" and calls[1]["azure_voice"] == "voice-two"
    assert calls[1]["azure_rate"] == 1.4 and calls[1]["azure_pitch"] == 1.5
    assert module.report_monitor()["request_successes"] == 2


def test_translation_switch_changes_language_without_replaying_completed_text():
    calls = []

    class Client:
        def __init__(self, config):
            self.config = dict(config)

        def translate(self, *_args):
            calls.append(self.config)
            return self.config["target_language"]

    module = TranslationWorker(Client)
    module.prepare(PrepareContext("tr", {}, {}), translation_config(target_language="ja"))
    module.start(StreamContext())
    assert TextSegment.from_packet(module.process(text("one")).ports["text"][0]).language == "ja"
    module.update_parameters({"target_language": "zh", "model": "model-two"})
    assert not module.process(text("one")).ports["text"]
    result = TextSegment.from_packet(module.process(text("two")).ports["text"][0])
    assert result.text == "zh" and result.language == "zh"
    assert calls[0]["model"] != calls[1]["model"]
    assert module.report_monitor()["request_successes"] == 2


def test_invalid_tts_provider_switch_is_rejected_without_failing_the_adapter_or_worker():
    adapter = PythonAdapter(
        ModuleRegistration(Manifest(Speech.MANIFEST), Speech.SCHEMA, Speech), "tts"
    )
    adapter.prepare(PrepareContext("tts", {}, {"audio": AudioFormat(24000)}), speech_config())
    adapter.start(StreamContext())
    worker = adapter._module.worker.worker
    try:
        with pytest.raises(TTSError, match="tts_azure_api_key_required"):
            adapter.update_parameters({"provider": "azure"}, expected_revision=0)
        assert adapter.state == State.RUNNING and worker.poll() is None
        assert adapter._pending is None
        with pytest.raises(ValueError, match="reprepare"):
            adapter.update_parameters({"maximum_audio_seconds": 40}, expected_revision=0)
        adapter.update_parameters(
            {"language": "ja", "request_timeout_seconds": 30}, expected_revision=0
        )
        adapter.process(PortBundle())  # Empty input applies settings without any billable call.
        assert adapter.parameter_revision == 1
        assert adapter._module.config["language"] == "ja"
        assert adapter._module.worker.worker is worker
        assert adapter._module.worker.timeout == 33
    finally:
        adapter.dispose()
    assert worker.poll() is not None


def test_translation_ipc_switch_preserves_pid_and_uses_new_model_key_and_language():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((payload, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {"choices": [{"message": {"content": "translated"}, "finish_reason": "stop"}]}
                ).encode()
            )

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        module = Translation()
        try:
            module.prepare(
                PrepareContext("tr", {}, {}),
                translation_config(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    target_language="ja",
                    request_timeout_seconds=2,
                ),
            )
            worker = module.worker.worker
            module.start(StreamContext())
            module.process(text("one"))
            module.update_parameters(
                {
                    "model": "new-model",
                    "api_key": "new-synthetic-key",
                    "target_language": "zh",
                    "request_timeout_seconds": 30,
                }
            )
            result = module.process(text("two"))
            assert TextSegment.from_packet(result.ports["text"][0]).language == "zh"
            assert len(received) == 2
            assert received[0][0]["model"] != received[1][0]["model"]
            assert received[1][0]["model"] == "new-model"
            assert received[1][1] == "Bearer new-synthetic-key"
            assert "Chinese" in received[1][0]["messages"][0]["content"]
            assert module.worker.worker is worker and worker.poll() is None
            assert module.worker.timeout == 33
        finally:
            module.dispose()
            server.shutdown()
            thread.join(2)
        assert worker.poll() is not None
