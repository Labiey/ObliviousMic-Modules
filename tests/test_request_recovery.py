"""A failed cloud utterance must not poison the worker or following messages."""

from fractions import Fraction

import numpy as np
import pytest
from obliviousmic.sdk import (
    AudioFormat,
    AudioPacket,
    PortBundle,
    PrepareContext,
    StreamContext,
    TextSegment,
)

from modules.asr_api.network import Module as ASR
from modules.asr_api.providers import ASRError
from modules.translate.network import Module as Translate
from modules.translate.providers import TranslationError
from modules.tts.network import Module as TTS
from modules.tts.providers import TTSError
from tests.test_asr_api import config as asr_config
from tests.test_translate import config as translate_config
from tests.test_tts import config as tts_config


def text(index):
    segment = TextSegment(str(index), "hello", Fraction(index), Fraction(index + 1), "en")
    return PortBundle({"text": (segment.to_packet("source"),)})


@pytest.mark.parametrize(
    "code",
    [
        "tts_provider_http_429",
        "tts_provider_http_401",
        "tts_request_timeout",
        "tts_audio_response_invalid",
    ],
)
def test_tts_failed_request_emits_nothing_next_message_works_without_reprepare(code):
    class Client:
        def __init__(self, config):
            self.calls = 0

        def synthesize(self, *_args):
            self.calls += 1
            if self.calls == 1:
                raise TTSError(code)
            return np.full((2400, 1), 0.1, np.float32)

    module = TTS(Client)
    module.prepare(PrepareContext("tts", {}, {"audio": AudioFormat(24000)}), tts_config())
    module.start(StreamContext())
    assert not any(module.process(text(0)).ports.values())
    assert module.report_monitor()["request_error_code"] == code
    assert not any(module.process(text(0)).ports.values())
    result = module.process(text(1))
    assert sum(p.frames for p in result.ports["audio"]) == 2400
    assert result.ports["audio"][0].media_start == 0
    assert len(result.ports["text"]) == 1
    assert module.client.calls == 2
    assert module.report_monitor() == {
        "request_successes": 1,
        "request_failures": 1,
        "request_error_code": None,
    }


def test_translation_failure_does_not_forward_wrong_language_or_repeat_failed_revision():
    class Client:
        def __init__(self, config):
            self.calls = 0

        def translate(self, *_args):
            self.calls += 1
            if self.calls == 1:
                raise TranslationError("translate_service_unavailable")
            return "翻译结果"

    module = Translate(Client)
    module.prepare(PrepareContext("translate", {}, {}), translate_config())
    module.start(StreamContext())
    assert not module.process(text(0)).ports["text"]
    assert module.report_monitor()["request_error_code"] == "translate_service_unavailable"
    assert not module.process(text(0)).ports["text"]
    assert len(module.process(text(1)).ports["text"]) == 1
    assert module.report_monitor()["request_failures"] == 1
    assert module.report_monitor()["request_error_code"] is None
    assert module.client.calls == 2


def test_asr_failure_discards_only_current_utterance_and_keeps_media_timestamps():
    class Client:
        def __init__(self, config):
            self.calls = 0

        def transcribe(self, _samples):
            self.calls += 1
            if self.calls == 1:
                raise ASRError("asr_api_service_unavailable")
            return "hello", "en"

    module = ASR(Client)
    module.prepare(
        PrepareContext("asr", {"audio": AudioFormat(16000)}, {}),
        asr_config(minimum_speech_ms=100, silence_ms=100),
    )
    module.start(StreamContext())
    values = np.concatenate((np.full((3200, 1), 0.2, np.float32), np.zeros((3200, 1), np.float32)))

    def feed(index):
        return module.process(
            PortBundle.audio(AudioPacket(values, AudioFormat(16000), Fraction(index * 2, 5)))
        )

    assert not feed(0).ports["text"]
    assert module.report_monitor()["request_failures"] == 1
    result = feed(1).ports["text"]
    assert len(result) == 1
    # The next utterance retains the legitimate 100 ms silent pre-roll, but
    # contains none of the failed utterance's voiced interval [0, 0.2).
    assert TextSegment.from_packet(result[0]).start == Fraction(3, 10)
    assert TextSegment.from_packet(result[0]).end == Fraction(7, 10)
    assert module.report_monitor()["request_error_code"] is None


def test_unexpected_tts_bug_still_raises_instead_of_silently_muting_forever():
    class Client:
        def __init__(self, config):
            pass

        def synthesize(self, *_args):
            raise RuntimeError("unexpected implementation failure")

    module = TTS(Client)
    module.prepare(PrepareContext("tts", {}, {"audio": AudioFormat(24000)}), tts_config())
    module.start(StreamContext())
    with pytest.raises(RuntimeError):
        module.process(text(0))


def test_real_ipc_worker_reports_failed_request_and_remains_available(tmp_path, monkeypatch):
    import sys
    from pathlib import Path

    from obliviousmic.adapters.python import PythonAdapter, State
    from obliviousmic.sdk import Manifest, ModuleRegistration

    from modules.tts import runtime

    root = Path(__file__).parents[1]
    (tmp_path / "fixture.py").write_text(
        "import sys;sys.path.insert(0," + repr(str(root)) + ")\n"
        "import numpy as np\n"
        "from modules.tts.network import Module as Base\n"
        "from modules.tts.providers import TTSError\n"
        "class Client:\n"
        " def __init__(self, config): self.calls=0\n"
        " def synthesize(self, *args):\n"
        "  self.calls+=1\n"
        "  if self.calls==1: raise TTSError('tts_provider_http_429')\n"
        "  return np.full((2400,1),0.1,np.float32)\n"
        "class Module(Base):\n"
        " def __init__(self): super().__init__(Client)\n",
        "utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "obliviousmic.adapters.worker",
        "--root",
        str(tmp_path),
        "--entrypoint",
        "fixture:Module",
    ]
    monkeypatch.setattr(runtime, "worker_command", lambda: command)
    adapter = PythonAdapter(
        ModuleRegistration(
            Manifest(runtime.Module.MANIFEST), runtime.Module.SCHEMA, runtime.Module
        ),
        "tts",
    )
    try:
        adapter.prepare(PrepareContext("tts", {}, {"audio": AudioFormat(24000)}), tts_config())
        adapter.start(StreamContext())
        assert not any(adapter.process(text(0)).ports.values())
        assert adapter.state == State.RUNNING
        assert adapter.report_monitor()["request_error_code"] == "tts_provider_http_429"
        result = adapter.process(text(1))
        assert sum(p.frames for p in result.ports["audio"]) == 2400
        assert adapter.report_monitor()["request_successes"] == 1
        assert adapter.report_monitor()["request_error_code"] is None
        assert adapter.state == State.RUNNING
    finally:
        adapter.dispose()
