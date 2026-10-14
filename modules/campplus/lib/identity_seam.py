# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""The identity-arm seam shared by the inference arms and the presence switch.

This module is deliberately a leaf: it imports nothing from ``ownermic`` besides
the standard library and numpy typing.  Both sides of the seam need these
definitions - :mod:`ownermic.inference.identity_stream` produces observations and
:mod:`ownermic.processing.presence_switch` consumes them - and a shared type must
therefore live below both packages.  Putting it in either one creates an import
cycle: ``processing.presence_switch`` already depends on ``inference.target_presence``
for the gate state machine, so an arm in ``inference`` importing the seam back *out
of* ``processing`` only resolves for whichever import order happens to be loaded
first, and fails as a partially-initialised-module ``ImportError`` otherwise.

Keeping the seam here means the dependency direction is strictly
``processing -> inference.identity_seam`` and ``inference.identity_stream ->
inference.identity_seam``, which is order-independent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class IdentityObservation:
    """One gate frame of identity evidence from a :class:`PresenceIdentityProvider`.

    Deliberately **unvalidated**: a realtime stage must not raise because an
    inference arm produced a surprising number, so the switch screens every field
    itself and falls back to holding the previous gate state.

    Attributes:
        score: Owner-similarity cosine for this frame, in ``[-1, 1]``.  Negative is
            valid and meaningful: it is the strongest available "not the owner".
        veto: Non-owner evidence cosine in the same range.  Note the *direction*: the
            gate treats a **low** smoothed veto score as "actively not the owner", so
            ``None`` means "no veto channel" while ``-1.0`` means "maximally not the
            owner" and will latch the switch shut.  Absence must be signalled with
            ``None``, never with a sentinel.
        mixture_energy_db: Pre-separation mic energy for this frame.  The switch only
            sees the audio entering it, so an arm that also observes the raw mic may
            report the true mixture level; when ``None`` the switch uses its own input
            level, which makes the recorded attenuation conservative rather than
            wrong-signed.
    """

    score: float
    veto: float | None = None
    mixture_energy_db: float | None = None


class PresenceIdentityProvider(Protocol):
    """Slow identity arm feeding the switch (the windowed CAM++/SV arm in production).

    Implementations receive exactly one gate frame of audio (``sample_rate /
    frame_hz`` samples, 160 at 16 kHz) per call and must build their own history.

    Returning an observation is the normal case *including* while the window is
    still filling: the offline reference scores every frame with a held value
    starting from ``0.0``, so a streaming arm must reproduce that per frame.
    ``None`` means "no usable evidence for this frame" and makes the switch hold
    its previous state; using it for "window not full yet" would advance the gate
    differently from the offline runs that produced the shipped thresholds.
    """

    @property
    def algorithmic_latency_ms(self) -> float:
        """Look-ahead-free observation delay of the arm (its window length etc.)."""
        ...

    def observe(
        self, samples: NDArray[np.float32], *, sample_rate: int
    ) -> IdentityObservation | None: ...

    def reset(self) -> None: ...
