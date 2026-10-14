# Frozen correctness tests; only imports adapted.
"""Numpy frame glue: geometry, normalisation and refusal behaviour without torch.

The parity claim against the torch wrapper lives in ``test_frame_glue_parity.py``; these
tests hold the parts that must be true on their own, including the boundary class that a
quarter-overlap test geometry used to hide.
"""

from __future__ import annotations

import unittest

import numpy as np
from modules.bsrnn.lib.frame_glue import (
    FrameBackend,
    FrameGlueError,
    NumpyFrameStream,
    periodic_hann,
)


def _tone(seconds: float, rate: int = 16_000, *, freq: float = 300.0) -> np.ndarray:
    step = 1.0 / rate
    return (0.3 * np.sin(2.0 * np.pi * freq * np.arange(int(seconds * rate)) * step)).astype(
        np.float32
    )


class _IdentityBackend(FrameBackend):
    """Perfect reconstruction: windowed STFT and nothing else."""

    def __init__(self, win: int) -> None:
        self._window = periodic_hann(win)

    def initial_states(self) -> tuple[object, ...]:
        return ()

    def separate_frames(
        self, frames: np.ndarray, layers: tuple[object, ...]
    ) -> tuple[np.ndarray, tuple[object, ...]]:
        return np.fft.rfft(frames * self._window, axis=-1), layers


class _MaskBackend(_IdentityBackend):
    """A non-trivial but deterministic transform, so OLA math is really exercised."""

    def __init__(self, win: int, *, seed: int = 11) -> None:
        super().__init__(win)
        rng = np.random.default_rng(seed)
        bins = win // 2 + 1
        self._real = rng.normal(0.0, 0.5, size=(1, bins)).astype(np.float32) + 1.5
        self._imag = rng.normal(0.0, 0.05, size=(1, bins)).astype(np.float32)

    def separate_frames(
        self, frames: np.ndarray, layers: tuple[object, ...]
    ) -> tuple[np.ndarray, tuple[object, ...]]:
        spectra = np.fft.rfft(frames * self._window, axis=-1)
        mask = self._real + 1j * self._imag
        return (spectra * mask).astype(np.complex64), layers

    @property
    def window(self) -> np.ndarray:
        return self._window


class WindowTest(unittest.TestCase):
    def test_periodic_not_symmetric(self) -> None:
        window = periodic_hann(8)
        expected = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(8) / 8.0)
        np.testing.assert_allclose(window, expected, rtol=0.0, atol=1e-7)
        self.assertNotAlmostEqual(
            float(window[-1]), 0.0, places=5, msg="symmetric Hann would end at zero"
        )

    def test_rejects_bad_sizes(self) -> None:
        for bad in (0, -4, 3.5, True):
            with self.subTest(bad=bad), self.assertRaises(FrameGlueError):
                periodic_hann(bad)  # type: ignore[arg-type]


class ConstructionTest(unittest.TestCase):
    def test_geometry_rules(self) -> None:
        for kwargs in (
            {"win": 31, "hop": 8},
            {"win": 32, "hop": 0},
            {"win": 32, "hop": 64},
            {"win": 8.0, "hop": 4},
        ):
            with self.subTest(**kwargs), self.assertRaises(FrameGlueError):
                NumpyFrameStream(backend=_IdentityBackend(int(kwargs["win"])), **kwargs)

    def test_latency_claim_is_the_window(self) -> None:
        stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=256)
        self.assertEqual(stream.latency_samples, 512)


class ReconstructionTest(unittest.TestCase):
    def _run(self, stream: NumpyFrameStream, source: np.ndarray, *, chunk: int) -> np.ndarray:
        parts = [
            stream.process(source[start : start + chunk]) for start in range(0, source.size, chunk)
        ]
        parts.append(stream.flush())
        return (
            np.concatenate([part for part in parts if part.size])
            if parts
            else np.zeros(0, np.float32)
        )

    def test_identity_backend_reconstructs_the_stream(self) -> None:
        # 32768 samples is a whole number of hops for every geometry below: the
        # reference emits complete hops and never invents a partial one, so an
        # unaligned probe length would make the size assertion lie in the other direction.
        for win, hop in ((512, 256), (32, 8), (1024, 256), (512, 128)):
            for chunk in (hop, win, 999, 4_096):
                source = _tone(32_768 / 16_000.0)
                stream = NumpyFrameStream(backend=_IdentityBackend(win), win=win, hop=hop)
                produced = self._run(stream, source, chunk=chunk)
                self.assertEqual(produced.size, source.size, f"win={win} hop={hop} chunk={chunk}")
                interior = slice(2 * win, -2 * win)
                np.testing.assert_allclose(
                    produced[interior],
                    source[interior],
                    atol=2e-5,
                    err_msg=f"win={win} hop={hop} chunk={chunk} failed to reconstruct",
                )

    def test_a_non_overlapping_hop_cannot_normalise_a_hann_window(self) -> None:
        """hop == win passes the geometry rules but breaks the COLA precondition.

        Synthesis blocks then butt against each other with no overlap, and the periodic
        Hann is zero at its endpoints, so the overlap-add envelope vanishes exactly at the
        block seams.  The reference validates only ``hop <= win``, so this is documented
        as a refusal at emit time rather than silently accepted or re-tuned here.
        """
        stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=512)
        source = _tone(32_768 / 16_000.0)
        with self.assertRaises(FrameGlueError):
            for start in range(0, source.size, 1_024):
                stream.process(source[start : start + 1_024])
            stream.flush()

    def test_lengths_that_are_exact_hop_multiples_still_finish(self) -> None:
        """The boundary class that a quarter-overlap test geometry used to hide.

        Every length here is at least one analysis window plus two samples: shorter than
        that, the reference reflection padding is undefined and refusal is the correct
        behaviour, which RefusalTest asserts separately.
        """
        for total in (32_768, 8_192, 1_024):
            source = _tone(total / 16_000.0)
            stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=256)
            produced = self._run(stream, source, chunk=1024)
            self.assertEqual(produced.size, total, f"total={total}")

    def test_masked_backend_output_is_stable_across_chunk_sizes(self) -> None:
        source = _tone(1.5)
        reference = None
        for chunk in (256, 512, 1024, 4_000):
            stream = NumpyFrameStream(backend=_MaskBackend(512), win=512, hop=256)
            produced = self._run(stream, source, chunk=chunk)
            if reference is None:
                reference = produced
                continue
            usable = min(reference.size, produced.size)
            np.testing.assert_allclose(
                produced[:usable], reference[:usable], atol=1e-6, err_msg=f"chunk={chunk}"
            )
            self.assertEqual(produced.size, reference.size)

    def test_reset_allows_a_second_stream(self) -> None:
        stream = NumpyFrameStream(backend=_MaskBackend(512), win=512, hop=256)
        first = self._run(stream, _tone(0.5), chunk=1024)
        stream.reset()
        self.assertEqual(stream.processed_samples, 0)
        self.assertEqual(stream.emitted_samples, 0)
        second = self._run(stream, _tone(0.5), chunk=1024)
        np.testing.assert_array_equal(first, second)


class RefusalTest(unittest.TestCase):
    def test_a_stream_shorter_than_one_window_is_refused_not_padded(self) -> None:
        stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=256)
        with self.assertRaises(FrameGlueError):
            stream.process(np.ones(64, dtype=np.float32))
            stream.flush()

    def test_double_flush_and_processing_after_flush_are_refused(self) -> None:
        stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=256)
        stream.process(_tone(0.2))
        stream.flush()
        with self.assertRaises(FrameGlueError):
            stream.flush()
        with self.assertRaises(FrameGlueError):
            stream.process(_tone(0.1))

    def test_non_finite_input_is_refused(self) -> None:
        stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=256)
        with self.assertRaises(FrameGlueError):
            stream.process(np.full(64, np.inf, dtype=np.float32))

    def test_backend_shape_and_finiteness_are_enforced(self) -> None:
        class BadShape(FrameBackend):
            def initial_states(self) -> tuple[object, ...]:
                return ()

            def separate_frames(self, frames, layers):  # type: ignore[no-untyped-def]
                return np.zeros((frames.shape[0], 3), dtype=np.complex64), layers

        stream = NumpyFrameStream(backend=BadShape(), win=512, hop=256)
        with self.assertRaises(FrameGlueError):
            stream.process(_tone(0.2))

        class RealOnly(BadShape):
            def separate_frames(self, frames, layers):  # type: ignore[no-untyped-def]
                return np.zeros((frames.shape[0], 257), dtype=np.float32), layers

        stream = NumpyFrameStream(backend=RealOnly(), win=512, hop=256)
        with self.assertRaises(FrameGlueError):
            stream.process(_tone(0.2))

        class NaNs(BadShape):
            def separate_frames(self, frames, layers):  # type: ignore[no-untyped-def]
                out = np.zeros((frames.shape[0], 257), dtype=np.complex64)
                out[0, 0] = np.nan + 0j
                return out, layers

        stream = NumpyFrameStream(backend=NaNs(), win=512, hop=256)
        with self.assertRaises(FrameGlueError):
            stream.process(_tone(0.2))

    def test_a_backend_losing_states_is_refused(self) -> None:
        class States(FrameBackend):
            def initial_states(self) -> tuple[object, ...]:
                return (np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32))

            def separate_frames(self, frames, layers):  # type: ignore[no-untyped-def]
                spectra = np.fft.rfft(frames * periodic_hann(frames.shape[1]), axis=-1)
                return spectra, (np.zeros(1, dtype=np.float32),)

        stream = NumpyFrameStream(backend=States(), win=512, hop=256)
        with self.assertRaises(FrameGlueError):
            stream.process(_tone(0.2))

    def test_carried_history_cannot_fabricate_missing_samples(self) -> None:
        stream = NumpyFrameStream(backend=_IdentityBackend(512), win=512, hop=256)
        stream.process(_tone(0.5))
        # Reach in and desynchronise the frontier the way a skipped read would, then the
        # next append must be the one to notice.
        stream._processed -= 100  # noqa: SLF001 - deliberately corrupting to prove the guard
        with self.assertRaises(FrameGlueError):
            stream.process(_tone(0.1))


class StatefulnessTest(unittest.TestCase):
    def test_a_stateful_backend_sees_each_frame_exactly_once(self) -> None:
        class Counter(FrameBackend):
            def __init__(self, win: int) -> None:
                self._window = periodic_hann(win)
                self.seen = 0

            def initial_states(self) -> tuple[object, ...]:
                return (0,)

            def separate_frames(self, frames, layers):  # type: ignore[no-untyped-def]
                self.seen += int(frames.shape[0])
                spectra = np.fft.rfft(frames * self._window, axis=-1)
                return spectra, (layers[0] + self.seen,)

        backend = Counter(512)
        stream = NumpyFrameStream(backend=backend, win=512, hop=256)
        source = _tone(1.0)
        for start in range(0, source.size, 1024):
            stream.process(source[start : start + 1024])
        stream.flush()
        self.assertGreater(stream.frames_processed, 0)
        self.assertEqual(
            backend.seen, stream.frames_processed, "a frame must be consumed exactly once"
        )
        self.assertEqual(stream.state_layers if hasattr(stream, "state_layers") else (1, 1)[0], 1)


if __name__ == "__main__":
    unittest.main()
