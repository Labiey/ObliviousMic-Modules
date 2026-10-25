# SPDX-License-Identifier: Apache-2.0
# Split from ObliviousMic's adapted frozen OwnerMic leaves; see NOTICE.
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from obliviousmic.sdk import ControlEvent, FlushResult, Latency, PortBundle

from .actions import initialize_action as initialize_action
from .actions import validate_action as validate_action
from .lib.base import ModelModule, model_path, session
from .lib.presence import PresenceGateConfig, PresenceGateError, StreamingPresenceGate


class Module(ModelModule):
    POLICY_FIELDS = {
        "open_threshold",
        "close_threshold",
        "open_hold_ms",
        "close_hold_ms",
        "min_open_ms",
        "min_close_ms",
        "ema_open_alpha",
        "ema_close_alpha",
        "fast_open_score",
        "fast_open_energy_db",
        "veto_score",
        "silence_close_energy_db",
        "silence_close_ms",
    }

    @staticmethod
    def policy_config(config):
        opening, closing = config["open_threshold"], config["close_threshold"]
        return PresenceGateConfig(
            open_threshold=opening,
            close_threshold=closing,
            attack_frames=config["open_hold_ms"] // 10,
            release_frames=config["close_hold_ms"] // 10,
            min_open_frames=config["min_open_ms"] // 10,
            min_close_frames=config["min_close_ms"] // 10,
            ema_open_alpha=config["ema_open_alpha"],
            ema_close_alpha=config["ema_close_alpha"],
            fast_open_score=min(config["fast_open_score"], opening),
            fast_open_energy_db=config["fast_open_energy_db"],
            veto_score=min(config["veto_score"], math.nextafter(closing, 0.0)),
            silence_close_energy_db=config["silence_close_energy_db"],
            silence_close_ms=config["silence_close_ms"],
        )

    @staticmethod
    def validate_settings(config):
        try:
            Module.policy_config(config)
        except PresenceGateError as exc:
            raise ValueError(str(exc)) from None

    def prepare(self, context, config):
        from .lib.identity_stream import SlidingIdentityProvider, WindowEmbeddingExtractor

        self.policy_settings = self.policy_config(config)
        self.policy = StreamingPresenceGate(self.policy_settings, frame_hz=100)
        self.context, self.config = context, dict(config)
        if self.factory:
            self.processor = self.factory(context, config)
        else:
            path = model_path(context, config)
            raw, self.lease = session(context, config, path)
            extractor = WindowEmbeddingExtractor(
                path,
                engine=config["engine"],
                device=config["device"],
                num_threads=config["threads"],
                probe_window_samples=config["window_ms"] * 16,
                session_factory=lambda *_: raw,
            )
            extractor.initialize()

            class Checked:
                def embed(self, audio):
                    value = extractor.embed(audio)
                    if value is None:
                        raise ValueError("identity inference produced no valid embedding")
                    return value

            if config["profile_id"]:
                embedding = context.capabilities["profile.read"](config["profile_id"], "embedding")
            else:
                embedding = np.load(config["embedding_path"], allow_pickle=False)
            self.processor = SlidingIdentityProvider(
                embedding,
                Checked(),
                window_ms=config["window_ms"],
                hop_ms=config["hop_ms"],
                veto_window_ms=config["veto_window_ms"],
                silence_floor_dbfs=config["silence_floor_dbfs"],
            )
        self.processor.initialize()
        self.reset("prepare")

    def reset(self, reason):
        if getattr(self, "processor", None):
            self.processor.reset()
        self.silence_frames = 0
        if hasattr(self, "policy"):
            self.policy.reset()
        self.monitor = {
            "score": None,
            "mixture_energy_db": None,
            "opened": False,
            "reason": "waiting",
        }

    def update_parameters(self, change):
        if set(change) - self.POLICY_FIELDS:
            raise ValueError("model settings require reprepare")
        candidate = {**self.config, **change}
        policy = self.policy_config(candidate)
        self.config, self.policy_settings = candidate, policy
        self.policy._config = policy  # preserve EMA and state across hot updates

    def report_monitor(self):
        return {
            **self.monitor,
            "confidence": self.policy.confidence,
            "veto": self.policy._veto_smoothed,
            "open_threshold": self.policy_settings.open_threshold,
            "close_threshold": self.policy_settings.close_threshold,
        }

    def process(self, bundle):
        outputs = []
        for p in bundle.ports.get("audio", ()):
            frame = (
                np.pad(p.samples[:, 0], (0, 160 - p.frames)) if p.frames < 160 else p.samples[:, 0]
            )
            observed = self.processor.observe(frame, sample_rate=16000)
            score, veto, energy = observed.score, observed.veto, observed.mixture_energy_db
            if (
                not math.isfinite(score)
                or not -1 <= score <= 1
                or not math.isfinite(energy)
                or (veto is not None and (not math.isfinite(veto) or not -1 <= veto <= 1))
            ):
                raise ValueError("invalid identity observation")
            self.silence_frames = (
                self.silence_frames + 1
                if energy < self.policy_settings.silence_close_energy_db
                else 0
            )
            silent = self.silence_frames * 10 >= self.policy_settings.silence_close_ms
            opened = self.policy.update(
                0.0 if silent else score, energy, energy, None if silent else veto
            )
            reason = (
                "silence"
                if silent
                else "veto"
                if self.policy._veto_smoothed is not None
                and self.policy._veto_smoothed < self.policy_settings.veto_score
                else "open"
                if opened
                else "low_confidence"
                if self.policy.confidence < self.policy_settings.open_threshold
                else "attack"
            )
            self.monitor = {
                "score": score,
                "mixture_energy_db": energy,
                "opened": opened,
                "reason": reason,
            }
            outputs.append(
                ControlEvent(
                    "audio",
                    p.media_start,
                    opened,
                    p.stream_id,
                    p.epoch_id,
                    late_policy="apply_next",
                )
            )
        return PortBundle({"decision": tuple(outputs)})

    def flush(self):
        return FlushResult()

    def report_latency(self):
        return Latency(Fraction(self.config["window_ms"] + self.config["hop_ms"], 1000))


# Package declarations are the source of truth for both bundled and source use.
Module.MANIFEST = json.loads(Path(__file__).with_name("manifest.json").read_text("utf-8"))
Module.SCHEMA = json.loads(Path(__file__).with_name("settings.schema.json").read_text("utf-8"))
