# Windows subtitles

Consumes TextSegment v1 metadata and displays a local always-on-top subtitle window.
Without a shortcut the window is created lazily when text arrives, not during graph preparation.
Set `overlay_hotkey` using the WebUI key recorder to show/hide it globally (blank disables the shortcut).
With a shortcut it starts with the session, even before text arrives. Hidden windows keep receiving
updates without reopening; pressing the shortcut again shows the current caption without taking focus.
Drag its header to move it. With a shortcut, × hides the window so it can be reopened; otherwise × closes it.
Font, opacity, timeout and click-through are configurable. Closing the window suppresses further display until the module restarts.
Text is passed downstream unchanged, allowing captions and TTS on the same graph.
Offline jobs do not open a window unless show_in_offline is explicitly enabled.
The GUI runs in an owned process, using bounded queues; excess display updates coalesce to the latest caption.
No subtitle text is placed on command lines, written to logs or retained in status messages.
Requires Windows Python with Tkinter (included with the normal Python Windows installer).
Shortcuts are released when the session stops. Choose different shortcuts for subtitles,
text input and soundboard playback. Game anti-cheat or exclusive fullscreen support is not guaranteed.
