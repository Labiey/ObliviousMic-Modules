# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""Re-bake the separation core's enrollment reference without a training stack.

The deployed tfmap core personalizes through one initializer: ``enroll_mag``,
the [257, 376] magnitude spectrum of a 3.0 s excerpt of the owner's enrollment
audio (48000 samples at 16 kHz; 376 frames = 375 hops + the leading frame at
hop 128).  Everything downstream - the L2 normalization, the TF-map attention
against the live mixture, the speaker context - is computed in-graph, so
switching owners is a single-tensor swap plus a manifest digest update, with
no torch and no re-export stack.

The public template exposes enroll_mag as an input and contains no enrollment
tensor. Binding converts that input to a private initializer in the user's
profile directory. Existing private cores can also be re-bound.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

#: The excerpt length the baked reference expects: 3.0 s at 16 kHz.
EXCERPT_SAMPLES: int = 48_000
#: Analysis window and hop of the separation model's STFT (manifest: hop 128).
STFT_WIN: int = 512
STFT_HOP: int = 128
FREQUENCY_BINS: int = STFT_WIN // 2 + 1
PUBLIC_TEMPLATE_FILENAME = "tfmap_context_causal_100-template.onnx"


class CoreRebakeError(RuntimeError):
    """Named failure so the enrollment flow can surface a category-only error."""


def compute_enroll_magnitude(excerpt: np.ndarray) -> np.ndarray:
    """The exact ``enroll_mag`` preprocessing: [F, frames] magnitude spectrum.

    Mirrors ``sep.stft(x)[0]`` from the reference backend: centered framing
    with reflect padding, periodic Hann window, rfft magnitude.  Any deviation
    here changes who the model listens for, so the parity test pins it.
    """
    if excerpt.ndim != 1:
        raise CoreRebakeError("enrollment excerpt must be mono 1-D")
    if excerpt.dtype != np.float32:
        raise CoreRebakeError("enrollment excerpt must be float32")
    if excerpt.shape[0] != EXCERPT_SAMPLES:
        raise CoreRebakeError(
            f"enrollment excerpt must be exactly {EXCERPT_SAMPLES} samples "
            f"(3.0 s at 16 kHz), got {excerpt.shape[0]}"
        )
    window = np.hanning(STFT_WIN + 1)[:-1].astype(np.float32)
    pad = STFT_WIN // 2
    padded = np.pad(excerpt, (pad, pad), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(padded, STFT_WIN)[::STFT_HOP]
    spectrum = np.fft.rfft(frames * window, axis=-1)
    return np.abs(spectrum).T.astype(np.float32)


def rebake_separation_core(core_path: Path, excerpt: np.ndarray, destination: Path) -> str:
    """Write a copy of the core with the new enrollment reference baked in.

    Returns the sha256 of the written file so the caller can update the
    profile manifest atomically after the write succeeds.
    """
    magnitude = compute_enroll_magnitude(excerpt)
    model = onnx.load(str(core_path))
    initializers = {init.name: init for init in model.graph.initializer}
    inputs = {value.name: value for value in model.graph.input}
    if "enroll_mag" in initializers:
        shape = list(initializers["enroll_mag"].dims)
    elif "enroll_mag" in inputs:
        tensor_type = inputs["enroll_mag"].type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.FLOAT:
            raise CoreRebakeError("template enroll_mag input must be float32")
        shape = [dimension.dim_value for dimension in tensor_type.shape.dim]
    else:
        raise CoreRebakeError("core has no enroll_mag input or initializer to bind")
    if shape != [1, FREQUENCY_BINS, magnitude.shape[1]]:
        raise CoreRebakeError(
            "core enroll_mag shape "
            f"{shape} does not match the excerpt spectrum "
            f"[1, {FREQUENCY_BINS}, {magnitude.shape[1]}]"
        )
    replacement = numpy_helper.from_array(
        magnitude[np.newaxis, :, :].astype(np.float32), name="enroll_mag"
    )
    for index, init in enumerate(model.graph.initializer):
        if init.name == "enroll_mag":
            model.graph.initializer[index].CopyFrom(replacement)
            break
    else:
        model.graph.initializer.append(replacement)
    # The runtime core has only mixture/state inputs; its reference is private.
    retained_inputs = [value for value in model.graph.input if value.name != "enroll_mag"]
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    onnx.checker.check_model(model)
    onnx.save(model, str(destination))
    digest = hashlib.sha256()
    with open(destination, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
