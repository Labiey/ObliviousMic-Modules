# SPDX-License-Identifier: Apache-2.0
# Adapted from ceva-ip/DPDFNet, revision 9bd9844a227bb6aa57e55588d8d0e961fcff1c46.
# Modified for ObliviousMic: bounded streaming, explicit delay compensation,
# exact EOS draining, dry/wet alignment and independently owned recurrent state.
"""48 kHz DPDFNet ONNX inference; no PyTorch, network access or device I/O."""

from collections import deque

import numpy as np

RATE = 48000
HOP = 480
WINDOW = 960
MODEL_DELAY_HOPS = 4  # MagnitudeMask(2) followed by DF lookahead(2).
LATENCY_FRAMES = (MODEL_DELAY_HOPS + 1) * HOP
PROFILES = {"light": "dpdfnet2_48khz_hr", "enhanced": "dpdfnet8_48khz_hr"}
STATE_SIZES = {"light": 56436, "enhanced": 90228}


class Session:
    """Share only the immutable model/session, never a node's recurrent state."""

    def __init__(self, path, mode, threads):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        meta = self.session.get_modelmeta().custom_metadata_map
        if (
            meta.get("model_type") != "dpdfnet"
            # The pinned upstream DPDFNet8 ONNX accidentally labels itself
            # dpdfnet2_48khz_hr. Verify its distinct state size as well; managed
            # installation additionally pins the complete model SHA-256.
            or meta.get("profile") not in {PROFILES[mode], "dpdfnet2_48khz_hr"}
            or int(meta.get("sample_rate", 0)) != RATE
            or int(meta.get("n_fft", 0)) != WINDOW
            or int(meta.get("hop_length", 0)) != HOP
        ):
            raise ValueError("incompatible DPDFNet model metadata")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 2 or len(outputs) != 2 or inputs[0].shape[-2:] != [WINDOW // 2 + 1, 2]:
            raise ValueError("incompatible DPDFNet streaming signature")
        size = int(meta["state_size"])
        if inputs[1].shape != [size] or size != STATE_SIZES[mode]:
            raise ValueError("invalid DPDFNet state size")
        erb = np.fromstring(meta["erb_norm_init"], sep=",", dtype=np.float32)
        spec = np.fromstring(meta["spec_norm_init"], sep=",", dtype=np.float32)
        if (
            len(erb) != int(meta["erb_norm_state_size"])
            or len(spec) != int(meta["spec_norm_state_size"])
            or len(erb) + len(spec) > size
            or not np.isfinite(erb).all()
            or not np.isfinite(spec).all()
        ):
            raise ValueError("invalid DPDFNet normalization state")
        self.initial_state = np.zeros(size, dtype=np.float32)
        self.initial_state[: len(erb)] = erb
        self.initial_state[len(erb) : len(erb) + len(spec)] = spec
        self.initial_state.flags.writeable = False
        self.inputs = [x.name for x in inputs]
        self.outputs = [x.name for x in outputs]

    def infer(self, spectrum, state):
        enhanced, next_state = self.session.run(
            self.outputs, {self.inputs[0]: spectrum, self.inputs[1]: state}
        )
        if not np.isfinite(enhanced).all() or not np.isfinite(next_state).all():
            raise ValueError("nonfinite DPDFNet inference output")
        return enhanced, next_state


class Stream:
    def __init__(self, session):
        self.session = session
        n = np.arange(WINDOW)
        self.window = np.sin(0.5 * np.pi * np.sin(np.pi * (n + 0.5) / WINDOW) ** 2)
        self.window = self.window.astype(np.float32)
        self.reset()

    def reset(self):
        self.state = self.session.initial_state.copy()
        # A half-window of left silence supplies the initial OLA contribution.
        self.pending = np.zeros(HOP, dtype=np.float32)
        self.overlap = np.zeros(WINDOW, dtype=np.float32)
        self.skip = LATENCY_FRAMES
        self.received = self.emitted = 0
        self.dry = deque()
        self.finished = False

    def _dry(self, length):
        chunks = []
        while length:
            first = self.dry.popleft()
            used = min(length, len(first))
            chunks.append(first[:used])
            if used < len(first):
                self.dry.appendleft(first[used:])
            length -= used
        return np.concatenate(chunks)

    def _feed(self, samples, mix):
        self.pending = np.concatenate((self.pending, samples))
        result = []
        while len(self.pending) >= WINDOW:
            spectrum = np.fft.rfft(self.pending[:WINDOW] * self.window)
            ri = np.stack((spectrum.real, spectrum.imag), axis=-1).astype(np.float32)
            enhanced, self.state = self.session.infer(ri[None, None], self.state)
            spectrum_out = enhanced[0, 0, :, 0] + 1j * enhanced[0, 0, :, 1]
            frame = (np.fft.irfft(spectrum_out, n=WINDOW) * self.window).astype(np.float32)
            self.overlap += frame
            committed = self.overlap[:HOP].copy()
            self.overlap[:HOP] = self.overlap[HOP:]
            self.overlap[HOP:] = 0
            self.pending = self.pending[HOP:]
            if self.skip:
                self.skip -= HOP
                continue
            count = min(HOP, self.received - self.emitted)
            if count:
                # Dry audio has exactly the same media time as enhanced audio.
                dry = self._dry(count)
                result.append((mix * committed[:count] + (1 - mix) * dry).astype(np.float32))
                self.emitted += count
        return np.concatenate(result) if result else np.empty(0, dtype=np.float32)

    def process(self, samples, mix=1.0):
        if self.finished:
            raise RuntimeError("DPDFNet stream is drained; reset before reuse")
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not np.isfinite(samples).all():
            raise ValueError("nonfinite input audio")
        if not samples.size:
            return np.empty(0, dtype=np.float32)
        self.received += len(samples)
        self.dry.append(samples.copy())
        return self._feed(samples, mix)

    def flush(self, mix=1.0):
        if self.finished or not self.received:
            self.finished = True
            return np.empty(0, dtype=np.float32)
        # At most 5 delayed hops plus one partial hop remain. Bound both the
        # computation and tail even if the caller supplied a sub-hop final packet.
        remaining = self.received - self.emitted
        padding = int(np.ceil((self.skip + remaining) / HOP)) * HOP
        result = self._feed(np.zeros(padding, dtype=np.float32), mix)
        if self.emitted != self.received:
            raise RuntimeError("DPDFNet failed to drain its audio tail")
        self.finished = True
        self.pending = np.empty(0, dtype=np.float32)
        self.dry.clear()
        return result
