import base64
import io
import json
import urllib.error
from email.message import Message

import numpy as np
import pytest
import soundfile as sf

from modules.broadcast.files import FileProcessor
from modules.broadcast.gateway import (
    BroadcastError,
    Gateway,
    NoRedirects,
    audio_effects,
    discover_port,
)


def reply(sequence, result):
    response = io.BytesIO(json.dumps({"jsonrpc": "2.0", "id": sequence, "result": result}).encode())
    response.headers = Message()
    response.headers["Content-Type"] = "application/json"
    return response


def test_identity_is_verified_before_any_file_paths_or_settings_are_sent():
    calls = []

    def open(request, **kwargs):
        call = json.loads(request.data)
        calls.append(call)
        return reply(call["id"], {"serverInfo": {"name": "another-service"}})

    with pytest.raises(BroadcastError, match="identity_mismatch"):
        Gateway(18100, opener=open)
    assert [c["method"] for c in calls] == ["initialize"]
    assert not any("inputPath" in json.dumps(c) for c in calls)


def test_verified_tool_contract_and_private_errors_are_sanitized_without_retry():
    calls = []

    def open(request, **kwargs):
        call = json.loads(request.data)
        calls.append(call)
        if call["method"] == "initialize":
            return reply(call["id"], {"serverInfo": {"name": "nvidia-broadcast", "version": "2.3"}})
        return reply(
            call["id"],
            {
                "isError": True,
                "structuredContent": {"error": {"code": "failed", "message": "PRIVATE_REFERENCE"}},
            },
        )

    gateway = Gateway(18100, opener=open)
    with pytest.raises(BroadcastError, match="^broadcast_tool_failed$"):
        gateway.call("submit_file_processing", _version=1, inputPath="private.wav")
    assert len(calls) == 2 and calls[1]["params"]["arguments"]["_version"] == 1
    with pytest.raises(urllib.error.HTTPError):
        NoRedirects().redirect_request(
            urllib.request.Request("http://127.0.0.1:18100/gateway"),
            None,
            302,
            "",
            {},
            "https://elsewhere.invalid",
        )


@pytest.mark.parametrize(
    "document",
    [b"invalid", base64.b64encode(b'{"port":true}'), base64.b64encode(b'{"port":18099}')],
)
def test_invalid_discovery_never_selects_an_arbitrary_server(tmp_path, document):
    path = tmp_path / "gateway.json"
    path.write_bytes(document)
    with pytest.raises(BroadcastError, match="configuration_invalid"):
        discover_port(path)


def test_discovery_reads_current_document_each_time(tmp_path):
    path = tmp_path / "gateway.json"
    assert discover_port(path) == 18100
    for port in [18102, 18104]:
        path.write_bytes(base64.b64encode(json.dumps({"port": port}).encode()))
        assert discover_port(path) == port


def capabilities():
    return {
        "effects": [
            {
                "id": effect,
                "section": "microphone",
                "accessible": True,
                "params": [{"name": "micProfile", "enum": ["default", "warm"]}],
            }
            for effect in ["noise_removal", "room_echo_removal", "studio_voice"]
        ]
    }


def test_effects_follow_available_capabilities_and_never_combine_studio_voice_with_denoisers():
    assert len(audio_effects(capabilities(), "noise_and_echo")) == 2
    assert audio_effects(capabilities(), "studio_voice", mic_profile="warm") == [
        {"effectId": "studio_voice", "params": {"micProfile": "warm"}}
    ]
    with pytest.raises(BroadcastError, match="profile_unavailable"):
        audio_effects(capabilities(), "studio_voice", mic_profile="bright")
    with pytest.raises(BroadcastError, match="effect_unavailable"):
        audio_effects({"effects": []})
    with pytest.raises(BroadcastError, match="settings_invalid"):
        audio_effects(capabilities(), strength=True)


class Files:
    def __init__(self, status="completed", *, skipped=False, bad_output=False):
        self.calls = []
        self.status = status
        self.skipped = skipped
        self.bad_output = bad_output

    def call(self, name, **args):
        self.calls.append((name, args))
        if name == "submit_file_processing":
            source, rate = sf.read(args["inputPath"], dtype="float32", always_2d=True)
            self.output = args["outputPath"]
            # Preserve backend padding here: timing alignment belongs to the streaming adapter.
            sf.write(
                self.output,
                np.concatenate((np.zeros((960, 1), np.float32), source)),
                rate,
                subtype="FLOAT",
            )
            return {
                "id": "owned-job",
                "batchId": "owned-batch",
                "skippedEffects": ["noise_removal"] if self.skipped else [],
            }
        if name == "wait_for_processing_update":
            return {
                "status": self.status,
                "cursor": "cursor",
                "jobs": [
                    {
                        "id": "owned-job",
                        "status": self.status,
                        "outputPath": "unrelated.wav" if self.bad_output else self.output,
                    }
                ],
            }
        assert (
            name == "control_file_processing_jobs"
            and args["action"] == "cancel"
            and args["jobIds"] == ["owned-job"]
        )
        return {"event": "jobs_cancelled"}


def test_private_files_are_removed_and_backend_timing_is_not_silently_trimmed(tmp_path):
    gateway = Files()
    samples = np.full((2400, 1), 0.2, np.float32)
    result = FileProcessor(gateway, audio_effects(capabilities()), workdir=tmp_path).process(
        samples
    )
    assert result.sample_rate == 48000 and result.input_frames == 2400
    assert len(result.samples) == 3360 and not result.samples[:960].any()
    np.testing.assert_allclose(result.samples[960:], samples, atol=1 / 32768)
    assert not list(tmp_path.iterdir())
    assert [call[0] for call in gateway.calls] == [
        "submit_file_processing",
        "wait_for_processing_update",
    ]


@pytest.mark.parametrize("state", ["paused_by_user", "cancelled", "failed", "mixed", "unknown"])
def test_paused_cancelled_or_failed_jobs_never_resume_retry_resubmit_or_pass_raw_audio(
    tmp_path, state
):
    gateway = Files(state)
    with pytest.raises(BroadcastError):
        FileProcessor(gateway, audio_effects(capabilities()), workdir=tmp_path).process(
            np.zeros((960, 1), np.float32)
        )
    assert sum(name == "submit_file_processing" for name, args in gateway.calls) == 1
    assert not any(
        args.get("action") in {"resume", "retry"} or args.get("all") for name, args in gateway.calls
    )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("options", [{"skipped": True}, {"bad_output": True}])
def test_skipped_effects_and_unrelated_outputs_are_not_treated_as_success(tmp_path, options):
    gateway = Files(**options)
    with pytest.raises(BroadcastError):
        FileProcessor(gateway, audio_effects(capabilities()), workdir=tmp_path).process(
            np.zeros((960, 1), np.float32)
        )
    assert not list(tmp_path.iterdir())


def test_expired_job_is_cancelled_by_exact_owned_id(tmp_path, monkeypatch):
    import modules.broadcast.files as implementation

    values = iter([0, 0, 3])
    monkeypatch.setattr(implementation.time, "monotonic", lambda: next(values))
    gateway = Files("queued")
    with pytest.raises(BroadcastError, match="processing_timeout"):
        FileProcessor(
            gateway, audio_effects(capabilities()), timeout_seconds=2, workdir=tmp_path
        ).process(np.zeros((960, 1), np.float32))
    assert gateway.calls[-1][1]["jobIds"] == ["owned-job"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["wrong_id", "bool_id", "non_json", "oversize", "invalid_session"])
def test_malformed_gateway_replies_do_not_create_a_verified_session(kind):
    def open(request, **kwargs):
        call = json.loads(request.data)
        response = reply(call["id"], {"serverInfo": {"name": "nvidia-broadcast"}})
        if kind == "wrong_id":
            response = reply(999, {})
        if kind == "bool_id":
            response = reply(True, {"serverInfo": {"name": "nvidia-broadcast"}})
        if kind == "non_json":
            response.headers.replace_header("Content-Type", "text/html")
        if kind == "oversize":
            response = io.BytesIO(b"x" * (2 * 1024 * 1024 + 1))
            response.headers = Message()
            response.headers["Content-Type"] = "application/json"
        if kind == "invalid_session":
            response.headers["Mcp-Session-Id"] = "bad session"
        return response

    with pytest.raises(BroadcastError):
        Gateway(18100, opener=open)


def test_valid_session_header_is_reused_for_subsequent_calls():
    headers = []

    def open(request, **kwargs):
        call = json.loads(request.data)
        headers.append(request.get_header("Mcp-session-id"))
        result = (
            {"serverInfo": {"name": "nvidia-broadcast", "version": "2.3.0"}}
            if call["method"] == "initialize"
            else {"structuredContent": {"effects": []}}
        )
        response = reply(call["id"], result)
        response.headers["Mcp-Session-Id"] = "owned-session"
        return response

    client = Gateway(18100, opener=open)
    assert client.version == "2.3.0"
    assert client.call("describe_effects", section="microphone") == {"effects": []}
    assert headers == [None, "owned-session"]


def stream_module(**changes):
    from obliviousmic.sdk import AudioFormat, PrepareContext, StreamContext

    from modules.broadcast.files import FileResult
    from modules.broadcast.network import Module as NetworkModule
    from modules.broadcast.runtime import Module

    class GatewayFixture:
        version = "2.3.0"
        port = 18100

        def __init__(self, *args, **kwargs):
            pass

        def call(self, *args, **kwargs):
            return capabilities()

    class Processor:
        def __init__(self, gateway, effects, **kwargs):
            self.delay = 1920 if effects[0]["effectId"] == "studio_voice" else 3360
            self.calls = []

        def process(self, samples, rate):
            self.calls.append(samples.copy())
            data = np.concatenate((np.zeros((self.delay, 1), np.float32), samples * 0.5))[
                : len(samples) + 960
            ]
            return FileResult(data, rate, len(samples), 0)

    config = {key: p["default"] for key, p in Module.SCHEMA["properties"].items()}
    config.update(changes)
    module = NetworkModule(GatewayFixture, Processor)
    module.prepare(
        PrepareContext(
            "broadcast",
            {"audio": AudioFormat(48000)},
            {"audio": AudioFormat(48000)},
            mode="realtime",
        ),
        config,
    )
    module.start(StreamContext("broadcast-test", 2))
    return module


@pytest.mark.parametrize("algorithm", ["noise_removal", "studio_voice"])
def test_streaming_context_and_delay_compensation_preserve_samples_pts_and_exact_partial_eos(
    algorithm,
):
    from fractions import Fraction

    from obliviousmic.sdk import AudioFormat, AudioPacket, PortBundle

    module = stream_module(algorithm=algorithm)
    frames = module.window + module.lookahead + 1377
    source = np.random.default_rng(3).uniform(-0.5, 0.5, (frames, 1)).astype("float32")
    output = []
    for offset in range(0, frames, 337):
        packet = AudioPacket(
            source[offset : offset + 337].copy(),
            AudioFormat(48000),
            Fraction(5) + Fraction(offset, 48000),
            "broadcast-test",
            2,
            offset,
            frozenset({"private"}),
        )
        output.extend(module.process(PortBundle.audio(packet)).ports["audio"])
    assert output and module.pending < module.window + module.lookahead
    output.extend(module.flush().outputs.ports["audio"])
    data = np.concatenate([packet.samples for packet in output])
    np.testing.assert_array_equal(data, source * 0.5)
    assert len(data) == frames and output[0].media_start == 5
    assert output[-1].media_end == Fraction(5) + Fraction(frames, 48000)
    assert [p.sequence for p in output] == list(range(len(output)))
    assert all(p.flags == frozenset({"private"}) and p.epoch_id == 2 for p in output)
    assert module.flush().drained and not module.flush().outputs.ports["audio"]
    assert (
        module.report_latency().mode == "variable"
        and module.report_latency().content_delay_seconds == 0
    )
    module.dispose()


def test_retained_pcm_detaches_from_ipc_and_epoch_reset_clears_audio():
    from fractions import Fraction

    from obliviousmic.sdk import AudioFormat, AudioPacket, PortBundle, StreamContext

    module = stream_module()
    packet = AudioPacket(
        np.full((960, 1), 0.4, np.float32), AudioFormat(48000), Fraction(0), "broadcast-test", 2
    )
    module.process(PortBundle.audio(packet))
    packet.samples.setflags(write=True)
    packet.samples[:] = 0
    outputs = module.flush().outputs.ports["audio"]
    np.testing.assert_allclose(outputs[0].samples, 0.2)
    module.reset("new epoch")
    module.start(StreamContext("broadcast-test", 3))
    assert not module.flush().outputs.ports["audio"] and module.frames == 0
    module.dispose()


def test_realtime_studio_voice_requires_a_sustainable_window():
    with pytest.raises(BroadcastError, match="window_too_short"):
        stream_module(algorithm="studio_voice", window_seconds=2)


def test_interrupted_worker_cleanup_checks_private_source_and_cancels_only_owned_job(tmp_path):
    from modules.broadcast.runtime import cleanup_jobs

    source = tmp_path / "own" / "input.wav"
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps({"id": "owned-job", "port": 18100, "sourcePath": str(source)}), encoding="utf-8"
    )
    calls = []

    class Client:
        def __init__(self, port, **kwargs):
            assert port == 18100

        def call(self, name, **kwargs):
            calls.append((name, kwargs))
            return {
                "jobs": [{"id": "owned-job", "status": "in_progress", "sourcePath": str(source)}]
            }

    cleanup_jobs(tmp_path, Client)
    assert calls[-1][0] == "control_file_processing_jobs" and calls[-1][1]["jobIds"] == [
        "owned-job"
    ]
    calls.clear()
    path.write_text(
        json.dumps(
            {"id": "unrelated", "port": 18100, "sourcePath": str(tmp_path.parent / "unrelated.wav")}
        ),
        encoding="utf-8",
    )
    cleanup_jobs(tmp_path, Client)
    assert not calls
