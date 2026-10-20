from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from obliviousmic.sdk import (
    AudioFormat,
    AudioPacket,
    PortBundle,
    PrepareContext,
    StreamContext,
    TextSegment,
)

from modules.asr.runtime import Module


class Recognizer:
    def __init__(self):
        self.calls = []

    def transcribe(self, samples, language):
        self.calls.append(samples.copy())
        return (SimpleNamespace(start=0, end=len(samples) / 16000, text="你好 world 日本語"),), "zh"


def prepare():
    processor = Recognizer()
    module = Module(lambda: processor)
    config = {k: v["default"] for k, v in Module.SCHEMA["properties"].items()}
    context = PrepareContext("asr", {"audio": AudioFormat(16000)}, {})
    module.prepare(context, config)
    module.start(StreamContext())
    return module, processor


def send(module, value, frames, start=0):
    outputs = []
    samples = np.full((160, 1), value, np.float32)
    for offset in range(0, frames, 160):
        packet = AudioPacket(
            samples[: min(160, frames - offset)],
            AudioFormat(16000),
            Fraction(start + offset, 16000),
        )
        outputs.extend(module.process(PortBundle.audio(packet)).ports["text"])
    samples[:] = -1  # Owned capture memory can be reused after process returns.
    return outputs


def test_silence_finalizes_owned_timestamped_transcript_and_eos_drains():
    module, processor = prepare()
    assert not send(module, 0.2, 4000, 16000)
    result = send(module, 0, 8000, 20000)
    assert len(result) == 1 and result[0].private
    text = TextSegment.from_packet(result[0])
    assert text.text == "你好 world 日本語" and text.start == 1
    assert text.end == Fraction(7, 4) and text.final
    np.testing.assert_allclose(processor.calls[0][:4000], 0.2)
    assert not module.flush().outputs.ports["text"]
    send(module, 0.3, 3333, 28000)
    last = module.flush().outputs.ports["text"]
    assert len(last) == 1 and last[0].observed_end == Fraction(31333, 16000)
    assert last[0].value["id"] != result[0].value["id"]
    module.dispose()


def test_silence_and_short_clicks_cannot_accumulate_or_trigger_model():
    module, processor = prepare()
    assert not send(module, 0, 160000)
    assert module.frames <= 3200 and sum(map(len, module.buffers)) <= 3200
    send(module, 0.2, 160, 160000)
    send(module, 0, 16000, 160160)
    module.flush()
    assert not processor.calls
    module.dispose()


def test_long_speech_is_bounded_and_epoch_reset_discards_old_audio():
    module, processor = prepare()
    outputs = send(module, 0.1, 200000)
    assert len(outputs) == 2 and max(map(len, processor.calls)) <= 96000
    module.reset("epoch")
    module.start(StreamContext("main", 1))
    assert not module.flush().outputs.ports["text"]
    send(module, 0.1, 4000)
    assert module.flush().outputs.ports["text"][0].epoch_id == 1
    module.dispose()
