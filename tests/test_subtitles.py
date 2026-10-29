from fractions import Fraction

import pytest
from obliviousmic.sdk import PortBundle, PrepareContext, StreamContext, TextSegment

from modules.subtitles.runtime import Module


class Session:
    def __init__(self):
        self.updates = []
        self.closed = False
        self.user_closed = False

    def submit(self, value):
        self.updates.append(value)
        return not self.user_closed

    def close(self):
        self.closed = True


def prepare(mode="realtime", **change):
    sessions = []

    def create():
        session = Session()
        sessions.append(session)
        return session

    module = Module(create)
    config = {k: v["default"] for k, v in Module.SCHEMA["properties"].items()}
    config.update(change)
    module.prepare(PrepareContext("overlay", {}, {}, mode), config)
    module.start(StreamContext())
    return module, sessions


def text(revision=0, final=True, epoch=0, value="你好 world 日本語"):
    packet = TextSegment(
        "test-1", value, Fraction(0), Fraction(1), final=final, revision=revision
    ).to_packet("asr", StreamContext("main", epoch))
    return PortBundle({"text": (packet,)})


def test_overlay_is_lazy_deduplicates_revisions_and_passes_text_unchanged():
    module, sessions = prepare()
    assert not sessions
    original = text(final=False)
    assert module.process(original) is original
    assert len(sessions) == 1 and sessions[0].updates[-1]["text"] == "你好 world 日本語"
    module.process(original)
    assert len(sessions[0].updates) == 1
    module.process(text(1))
    assert sessions[0].updates[-1]["revision"] == 1
    module.update_parameters({"font_size": 40})
    assert sessions[0].updates[-1]["config"]["font_size"] == 40
    assert sessions[0].updates[0]["config"]["font_size"] == 28
    module.reset("epoch")
    assert sessions[0].updates[-1]["text"] == ""
    module.process(text(epoch=1))
    assert sessions[0].updates[-1]["revision"] == 0
    module.dispose()
    assert sessions[0].closed


def test_user_closed_window_does_not_reopen_until_enabled_again():
    module, sessions = prepare()
    module.process(text())
    sessions[0].user_closed = True
    module.process(text(1))
    assert module.user_closed and sessions[0].closed
    module.process(text(2))
    assert len(sessions) == 1
    module.update_parameters({"enabled": False})
    module.update_parameters({"enabled": True})
    module.process(text(3))
    assert len(sessions) == 2
    module.stop()
    assert sessions[-1].closed


def test_offline_jobs_do_not_open_a_window_unless_explicitly_enabled():
    module, sessions = prepare("offline")
    original = text()
    assert module.process(original) is original and not sessions
    module.dispose()
    module, sessions = prepare("offline", show_in_offline=True)
    module.process(original)
    assert len(sessions) == 1
    module.dispose()


def test_bound_shortcut_starts_without_text_and_is_released_on_stop():
    module, sessions = prepare(overlay_hotkey="Ctrl+Alt+C")
    assert len(sessions) == 1
    assert sessions[0].updates[-1]["config"]["overlay_hotkey"] == "Ctrl+Alt+C"
    assert sessions[0].updates[-1]["text"] == ""
    module.start(StreamContext())
    assert len(sessions) == 1  # Do not leak a GUI/hook on repeated stream starts.
    original = text()
    assert module.process(original) is original
    module.stop()
    assert sessions[0].closed
    module, sessions = prepare("offline", overlay_hotkey="Ctrl+Alt+C")
    assert not sessions
    module.dispose()


def test_shortcut_is_validated_before_launching_any_window():
    with pytest.raises(ValueError, match="text_overlay_hotkey_invalid"):
        prepare(overlay_hotkey="t")


def test_subtitle_shortcuts_conflict_with_text_inputs_and_other_subtitle_nodes():
    from obliviousmic.graph.dag import DAGCompiler
    from obliviousmic.graph.primitives import register_primitives
    from obliviousmic.sdk import AudioFormat, Manifest, ModuleRegistration, ModuleRegistry

    registry = ModuleRegistry()
    register_primitives(registry)
    registry.register(ModuleRegistration(Manifest(Module.MANIFEST), Module.SCHEMA, Module))
    document = {
        "graph_schema": 1,
        "graph_id": "captions",
        "revision": 1,
        "execution": "dag",
        "nodes": [
            {
                "id": "text",
                "type": "core.text_source",
                "settings": {"overlay_enabled": True, "overlay_hotkey": "Ctrl+Alt+C"},
            },
            {
                "id": "subtitles",
                "type": "org.obliviousmic.subtitles",
                "settings": {"overlay_hotkey": "ctrl+alt+c"},
            },
            {"id": "sink", "type": "core.text_sink"},
        ],
        "edges": [
            {"from": "text.text", "to": "subtitles.text"},
            {"from": "subtitles.text", "to": "sink.text"},
        ],
    }
    with pytest.raises(ValueError, match="hotkey_duplicate"):
        DAGCompiler(registry).compile(document, AudioFormat(16000))
    document["nodes"][0]["settings"]["overlay_hotkey"] = "Ctrl+Alt+T"
    DAGCompiler(registry).compile(document, AudioFormat(16000))


def test_window_queue_coalesces_old_updates_and_accepts_maximum_unicode_caption():
    import json
    import queue

    from modules.subtitles.window import MAX_LINE, coalesce

    inbox = queue.Queue(maxsize=2)
    coalesce(inbox, "first")
    coalesce(inbox, "second")
    coalesce(inbox, "third")
    assert [inbox.get_nowait(), inbox.get_nowait()] == ["second", "third"]
    encoded = json.dumps({"text": "😀" * 16384}, ensure_ascii=True)
    assert len(encoded) + 2000 < MAX_LINE
