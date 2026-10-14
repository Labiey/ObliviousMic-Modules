# SPDX-License-Identifier: Apache-2.0
# Adapted for module-local distribution in 2026; see NOTICE and provenance.md.
# Reused from frozen OwnerMic; imports adapted only. Historical measurements in the
# original docstring describe the frozen version, not this refactor.
"""Selectable inference sessions for the streaming core (AGENTS section 28).

The shipped runtime currently runs the exported ONNX graph on ONNX Runtime CPU.
Measurement showed OpenVINO CPU running the *same* fp32 graph is roughly 20
percent faster per block and costs about 0.7 fewer sustained cores under a loaded
game, with output numerically identical to ORT (135 dB SI-SDR, zero maximum
absolute difference).  Sustained cores matter disproportionately here because the
measured game-load bottleneck is a package power/current cap: fewer of our own
cores also means more clock headroom for the game.

This module provides the seam, not a policy change: the default stays
``onnxruntime`` and no call site silently switches engines.  Two rules are
enforced here because both were learned the hard way this session:

* a thread count is only trusted after it is read back from the compiled model;
  a silently ignored property once invalidated an entire thread sweep;
* the device list is enumerated from the runtime and an unavailable device is an
  error naming what was actually found, never a quiet fallback to CPU.

``openvino`` is an optional extra.  When it is not installed the OpenVINO path
raises :class:`SessionUnavailableError` so callers can decide, instead of an
``ImportError`` escaping from inside a realtime path.
"""

from __future__ import annotations

import os
from dataclasses import field
from typing import Any, NamedTuple

SUPPORTED_ENGINES = ("onnxruntime", "openvino")
SUPPORTED_OV_DEVICES = ("CPU", "GPU", "NPU")
ORT_PROVIDERS = ("CPUExecutionProvider",)


class SessionError(RuntimeError):
    """Base for every engine-seam failure, so callers have one thing to catch."""


class SessionUnavailableError(SessionError):
    """The requested engine or device cannot serve this host."""


class SessionConfigError(SessionError, ValueError):
    """The requested engine/device/thread combination is not valid."""


def preferred_port_name(port: Any) -> str:
    """Pick one name for a port deterministically.

    A compiled OpenVINO port can carry two names for the same tensor: the ONNX graph's own
    output name and the internal producer name. ``get_any_name()`` returns either, so a
    session built that way hands back identifiers that drift between builds, and callers
    that feed and read by name silently mismatch - which in the shipped identity arm turned
    into an exception swallowed into a ``None`` embedding and a gate that never opened.
    Names that are not derived from a node output (no path separator) are the graph's own
    vocabulary, so they win, and the remainder is sorted so the choice cannot vary.
    """
    try:
        names = {str(name) for name in port.get_names()}
    except Exception:  # noqa: BLE001 - fall back to whatever the port will say
        return str(port.get_any_name())
    if not names:
        return str(port.get_any_name())
    clean = sorted(name for name in names if "/" not in name)
    return (clean or sorted(names))[0]


def ort_element_name(port: Any) -> str:
    """Describe an OpenVINO port's element type in the words onnxruntime callers expect.

    The shipped identity arm validates ``tensor(float)`` before loading a model, and that
    check is the guard against feeding an int8 or f64 encoder into a float32 pipeline.
    Reporting no type would either defeat the guard or force every caller to branch on the
    engine, so the two vocabularies are mapped once, here.
    """
    try:
        described = str(port.get_element_type()).lower()
    except Exception:  # noqa: BLE001 - an unreadable type is reported as unknown
        return ""
    # OpenVINO renders an element type as "<Type: 'float32'>", so the name has to be lifted
    # out of that repr; a library that ever prints the bare word still resolves.
    quoted = described.split("'")
    element = (quoted[1] if len(quoted) > 1 else described).replace(" ", "")
    # OpenVINO names its element types "float32"; onnxruntime says "tensor(float)". Both
    # spellings are accepted per row so neither library's vocabulary being changed upstream
    # can silently turn a real float32 model into an unrecognised one.
    return {
        "dynamic": "",
        "boolean": "tensor(bool)",
        "bool": "tensor(bool)",
        "bfloat16": "tensor(bfloat16)",
        "bf16": "tensor(bfloat16)",
        "float": "tensor(float)",
        "float32": "tensor(float)",
        "f32": "tensor(float)",
        "double": "tensor(double)",
        "float64": "tensor(double)",
        "f64": "tensor(double)",
        "float16": "tensor(float16)",
        "half": "tensor(float16)",
        "f16": "tensor(float16)",
        "int8": "tensor(int8)",
        "i8": "tensor(int8)",
        "int16": "tensor(int16)",
        "i16": "tensor(int16)",
        "int32": "tensor(int32)",
        "i32": "tensor(int32)",
        "int64": "tensor(int64)",
        "i64": "tensor(int64)",
        "uint8": "tensor(uint8)",
        "u8": "tensor(uint8)",
        "uint16": "tensor(uint16)",
        "u16": "tensor(uint16)",
        "uint32": "tensor(uint32)",
        "u32": "tensor(uint32)",
        "uint64": "tensor(uint64)",
        "u64": "tensor(uint64)",
    }.get(element, "")


def _port_shape(port: Any) -> list[Any]:
    """Static dimensions as ints, dynamic ones as ``None``, like the ORT convention."""
    try:
        partial = port.get_partial_shape()
    except Exception:  # noqa: BLE001 - a descriptor we cannot read is reported, not guessed
        return []
    shape: list[Any] = []
    for dimension in partial:
        try:
            if bool(dimension.is_static):
                shape.append(int(dimension.get_length()))
                continue
        except Exception:  # noqa: BLE001
            pass
        shape.append(None)
    return shape


class _ModelMeta:
    """The ``get_modelmeta()`` contract: one attribute, a plain mapping of strings."""

    __slots__ = ("custom_metadata_map",)

    def __init__(self, values: dict[str, str]) -> None:
        self.custom_metadata_map = values


def _runtime_metadata(model: Any) -> dict[str, str]:
    """Flatten OpenVINO's rt_info into the key/value strings the validators expect.

    onnxruntime publishes an ONNX file's custom metadata flat as
    ``get_modelmeta().custom_metadata_map``; OpenVINO nests the same values under
    ``rt_info["framework"]`` as OVAny wrappers. The shipped identity arm validates framework,
    sample_rate, output_dim and feature_normalize_type before it will load a model - that
    check is what stops a swapped file silently changing the identity domain - so the
    adapter has to answer it instead of letting one engine skip validation.
    """
    collected: dict[str, str] = {}
    try:
        info = model.rt_info
    except Exception:  # noqa: BLE001 - unreadable metadata is reported as none
        return collected
    for section in ("framework", "parameters"):
        try:
            node = info[section]
            keys = list(node)
        except Exception:  # noqa: BLE001 - the section simply may not exist
            continue
        for key in keys:
            try:
                value = node[key]
            except Exception:  # noqa: BLE001
                continue
            collected[str(key)] = str(getattr(value, "value", value))
    return collected


class SessionIo(NamedTuple):
    """An engine-neutral input/output descriptor.

    ``shape`` mirrors the onnxruntime field callers already read: a list with concrete
    integers where the axis is static and ``None`` where it is dynamic. The OpenVINO adapter
    used to report names only, which made every shape-aware caller an onnxruntime caller.
    """

    name: str
    shape: list[Any] = field(default_factory=list)
    type: str = ""
    shape: list[Any] = []


def validate_choice(engine: str, device: str = "CPU", threads: int = 1) -> None:
    """Reject impossible selections before touching any runtime."""
    if engine not in SUPPORTED_ENGINES:
        raise SessionConfigError(f"engine {engine!r} is not one of {', '.join(SUPPORTED_ENGINES)}")
    if int(threads) <= 0:
        raise SessionConfigError(f"threads must be positive, got {threads!r}")
    if engine == "openvino" and device not in SUPPORTED_OV_DEVICES:
        raise SessionConfigError(
            f"OpenVINO device {device!r} is not one of {', '.join(SUPPORTED_OV_DEVICES)}"
        )


class OpenVinoSession:
    """ORT-session-shaped wrapper around a compiled OpenVINO model.

    Callers use only ``get_inputs()``, ``get_outputs()`` and ``run(None, feeds)``,
    which is deliberately the whole public surface: the streaming path must not
    be able to depend on engine-specific behaviour.
    """

    def __init__(self, model_path: Any, threads: int, device: str = "CPU") -> None:
        validate_choice("openvino", device=device, threads=threads)
        try:
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - depends on the venv
            raise SessionUnavailableError(
                f"openvino engine requested but not importable: {exc}"
            ) from exc

        self.engine = "openvino"
        self.device = device
        self.requested_threads = int(threads)
        try:
            core = ov.Core()
            available = list(core.available_devices)
            if device not in available:
                raise SessionUnavailableError(
                    f"OpenVINO device {device!r} unavailable; enumerated "
                    f"{', '.join(available) or 'no devices'}"
                )
            model = core.read_model(os.fspath(model_path))
            self._model = model
            self._compiled = core.compile_model(
                model,
                device,
                {
                    "PERFORMANCE_HINT": "LATENCY",
                    # This build rejects the legacy CPU_THREADS_NUM key and has no
                    # ov.properties.performance_hint attribute, so the portable
                    # generic string keys are used.
                    "INFERENCE_NUM_THREADS": str(int(threads)),
                },
            )
        except SessionUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - engine errors surface verbatim
            raise SessionUnavailableError(
                f"{device} could not compile this model: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc

        applied = self._read_back_threads()
        self.applied_threads = applied
        self._request = self._compiled.create_infer_request()
        self._input_names = [preferred_port_name(value) for value in self._compiled.inputs]
        self._output_names = [preferred_port_name(value) for value in self._compiled.outputs]

    def _read_back_threads(self) -> int:
        try:
            applied = int(self._compiled.get_property("INFERENCE_NUM_THREADS"))
        except Exception as exc:  # noqa: BLE001
            raise SessionUnavailableError(f"cannot read back thread count: {exc}") from exc
        if applied != self.requested_threads:
            raise SessionUnavailableError(
                f"asked for {self.requested_threads} threads but {self.device} reports "
                f"{applied}; refusing to trust an uncontrolled engine"
            )
        return applied

    def get_modelmeta(self) -> _ModelMeta:
        """Model metadata in the shape every shipped validator already reads."""
        return _ModelMeta(_runtime_metadata(self._model))

    def get_inputs(self) -> list[SessionIo]:
        return [self._descriptor(port) for port in self._compiled.inputs]

    def get_outputs(self) -> list[SessionIo]:
        return [self._descriptor(port) for port in self._compiled.outputs]

    @staticmethod
    def _descriptor(port: Any) -> SessionIo:
        return SessionIo(preferred_port_name(port), _port_shape(port), ort_element_name(port))

    def run(self, _output_names: Any, feeds: dict[str, Any]) -> list[Any]:
        missing = [name for name in self._input_names if name not in feeds]
        if missing:
            raise SessionConfigError(f"feeds are missing inputs {', '.join(missing)}")
        try:
            result = self._request.infer({name: feeds[name] for name in self._input_names})
        except Exception as exc:  # noqa: BLE001 - e.g. an op the device cannot run
            raise SessionUnavailableError(
                f"{self.device} inference failed: {type(exc).__name__}: {str(exc)[:300]}"
            ) from exc
        return [result[name] for name in self._output_names]


def ort_session_options(threads: int) -> Any:
    """The tuned onnxruntime session options, in one place.

    Sequential execution and the two ``allow_spinning`` entries are not decoration: the
    identity arm's tail latencies were measured with them, so any code path that builds an
    ORT session has to use this helper or it is silently a different operating point. They
    used to live only inside the speaker encoder, which meant the shared engine seam was
    quietly dropping them.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def make_session(
    engine: str,
    model_path: Any,
    threads: int,
    device: str = "CPU",
) -> Any:
    """Build a session for the requested engine, CPU-only and pinned to one provider."""
    validate_choice(engine, device=device, threads=threads)
    if engine == "openvino":
        return OpenVinoSession(model_path, threads=threads, device=device)

    options = ort_session_options(threads)
    import onnxruntime as ort

    session = ort.InferenceSession(
        os.fspath(model_path), sess_options=options, providers=list(ORT_PROVIDERS)
    )
    used = list(session.get_providers())
    if used != list(ORT_PROVIDERS):
        raise SessionUnavailableError(
            f"onnxruntime resolved providers {', '.join(used)}, expected CPU only"
        )
    return session


def install_session(
    holder: Any,
    engine: str,
    model_path: Any,
    threads: int,
    device: str = "CPU",
) -> Any:
    """Swap a separator's session and refresh its cached input/output names."""
    session = make_session(engine, model_path, threads=threads, device=device)
    holder._session = session
    holder.input_names = tuple(value.name for value in session.get_inputs())
    holder.output_names = tuple(value.name for value in session.get_outputs())
    holder.engine = engine
    return session


def probe_engines() -> dict[str, object]:
    """Ask each runtime what it can actually do, without running anything through it.

    This is a control-path call and costs about a second on the first OpenVINO probe, so it
    belongs behind an endpoint the UI loads once, not behind the metrics ticker. Availability
    is read from the runtime rather than inferred from which import succeeded, because a
    device list is the part the user actually has to choose from.
    """
    engines: dict[str, object] = {}
    try:
        import onnxruntime as ort

        engines["onnxruntime"] = {
            "available": True,
            "version": getattr(ort, "__version__", "unknown"),
            "devices": ["CPU"],
        }
    except ImportError as exc:  # pragma: no cover - onnxruntime is a base dependency
        engines["onnxruntime"] = {
            "available": False,
            "version": "unavailable",
            "devices": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }
    try:
        import openvino
        from openvino import Core

        core = Core()
        engines["openvino"] = {
            "available": True,
            "version": getattr(openvino, "__version__", "unknown"),
            "devices": sorted({str(device) for device in core.available_devices}),
        }
    except Exception as exc:
        engines["openvino"] = {
            "available": False,
            "version": "unavailable",
            "devices": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }
    return {"supported": list(SUPPORTED_ENGINES), "engines": engines}
