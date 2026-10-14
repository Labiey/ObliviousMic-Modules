# Frozen correctness tests; only imports adapted.
"""The core adapter must be pinned to the exported layout, not merely plausible.

A wrong transpose here still yields finite audio of the right shape, so the only defence
is a session double that records what it was handed, plus an end-to-end comparison
against the development-side torch separator running the same real weights.
"""

from __future__ import annotations

import unittest

import numpy as np
from modules.bsrnn.lib.frame_glue import FrameGlueError, NumpyFrameStream
from modules.bsrnn.lib.onnx_frame_backend import STATE_LAYERS, OnnxFrameBackend

WIN = 512
HOP = 128
BINS = WIN // 2 + 1
STATE_SHAPE = (1, 32, 256)


class _Io:
    def __init__(self, name: str) -> None:
        self.name = name


class _SessionDouble:
    """A stand-in core that echoes a deterministic mask and records the feeds."""

    def __init__(
        self, *, mask_real: float = 0.5, drop_states: bool = False, swap_names: bool = False
    ) -> None:
        self.mask_real = mask_real
        self.calls: list[dict[str, np.ndarray]] = []
        self._drop_states = drop_states
        names = list(("est_imag", "est_real") if swap_names else ("est_real", "est_imag"))
        self._outputs = names + [f"s{index}_out" for index in range(2 * STATE_LAYERS)]
        if drop_states:
            self._outputs = names

    def get_inputs(self) -> list[_Io]:
        return [
            _Io(name)
            for name in (
                "mix_real",
                "mix_imag",
                *(f"s{index}" for index in range(2 * STATE_LAYERS)),
            )
        ]

    def get_outputs(self) -> list[_Io]:
        return [_Io(name) for name in self._outputs]

    def run(self, _names: object, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append(dict(feeds))
        real = feeds["mix_real"] * self.mask_real
        imag = feeds["mix_imag"] * self.mask_real
        produced: list[np.ndarray] = [real.astype(np.float32), imag.astype(np.float32)]
        if not self._drop_states:
            produced.extend(feeds[name] + np.float32(1.0) for name in list(feeds)[2:])
        return produced


def _backend(**kwargs: object) -> tuple[OnnxFrameBackend, _SessionDouble]:
    session = _SessionDouble(**kwargs)  # type: ignore[arg-type]
    return OnnxFrameBackend(session=session, win=WIN, hop=HOP), session  # type: ignore[arg-type]


class LayoutTest(unittest.TestCase):
    def test_mix_is_fed_in_frequency_first_order(self) -> None:
        backend, session = _backend()
        frames = np.random.default_rng(3).uniform(-0.4, 0.4, size=(6, WIN)).astype(np.float32)
        spectra, states = backend.separate_frames(frames, backend.initial_states())
        mix = session.calls[0]["mix_real"]
        self.assertEqual(mix.shape, (BINS, 6), "the graph wants (freq, frames)")
        self.assertEqual(spectra.shape, (6, BINS), "the wrapper wants (frames, freq)")
        self.assertEqual(len(states), 2 * STATE_LAYERS)

    def test_the_window_is_applied_by_the_backend_not_the_glue(self) -> None:
        backend, session = _backend()
        frames = np.zeros((1, WIN), dtype=np.float32)
        frames[0, 10] = 1.0
        backend.separate_frames(frames, backend.initial_states())
        expected = np.fft.rfft(frames[0] * np.array(backend_win(), dtype=np.float32)).real
        np.testing.assert_allclose(session.calls[0]["mix_real"][:, 0], expected, atol=1e-6)

    def test_states_are_threaded_and_returned(self) -> None:
        backend, session = _backend()
        frames = np.ones((2, WIN), dtype=np.float32) * 0.1
        first = backend.initial_states()
        _, second = backend.separate_frames(frames, first)
        self.assertEqual(float(np.max(second[0])), 1.0, "the echoed state must come back advanced")
        _, third = backend.separate_frames(frames, second)
        self.assertEqual(float(np.max(third[0])), 2.0)

    def test_a_masked_core_reduces_energy_without_breaking_the_stream(self) -> None:
        backend, _ = _backend(mask_real=0.25)
        stream = NumpyFrameStream(backend=backend, win=WIN, hop=HOP)
        source = (0.4 * np.sin(np.arange(8_192) * 0.05)).astype(np.float32)
        parts = []
        for start in range(0, source.size, 1_024):
            parts.append(stream.process(source[start : start + 1_024]))
        parts.append(stream.flush())
        produced = np.concatenate([part for part in parts if part.size])
        self.assertEqual(produced.size, source.size)
        self.assertTrue(np.all(np.isfinite(produced)))
        self.assertLess(float(np.max(np.abs(produced))), float(np.max(np.abs(source))) * 1.2)


def backend_win() -> list[float]:
    from modules.bsrnn.lib.frame_glue import periodic_hann

    return periodic_hann(WIN).tolist()


class RefusalTest(unittest.TestCase):
    def test_wrong_io_count_is_refused_at_construction(self) -> None:
        with self.assertRaises(FrameGlueError):
            _backend(drop_states=True)

    def test_swapped_output_names_are_refused(self) -> None:
        with self.assertRaises(FrameGlueError):
            _backend(swap_names=True)

    def test_bad_geometry_and_frame_shape_are_refused(self) -> None:
        with self.assertRaises(FrameGlueError):
            OnnxFrameBackend(session=_SessionDouble(), win=0, hop=128)  # type: ignore[arg-type]
        with self.assertRaises(FrameGlueError):
            OnnxFrameBackend(session=_SessionDouble(), win=512, hop=-1)  # type: ignore[arg-type]
        backend, _ = _backend()
        with self.assertRaises(FrameGlueError):
            backend.separate_frames(np.ones((2, 64), dtype=np.float32), backend.initial_states())
        with self.assertRaises(FrameGlueError):
            backend.separate_frames(np.ones((2, WIN), dtype=np.float32), ())

    def test_a_state_of_the_wrong_shape_is_refused(self) -> None:
        backend, _ = _backend()
        states = list(backend.initial_states())
        states[0] = np.zeros((2, 32, 256), dtype=np.float32)
        with self.assertRaises(FrameGlueError):
            backend.separate_frames(np.ones((2, WIN), dtype=np.float32), tuple(states))


# Real-core verification is deliberately not a skipped-by-default test: a test nobody can
# run is a landmine, not a guard. The shipped graph is exercised directly by
# evaluation/core_smoke.py, whose measured numbers are quoted in the backend docstring.


if __name__ == "__main__":
    unittest.main()
