# Windows subtitles

Consumes TextSegment v1 metadata and displays a local always-on-top subtitle window.
The window is created lazily when text arrives, not during graph preparation. Drag its header to move it; × closes it.
Font, opacity, timeout and click-through are configurable. Closing the window suppresses further display until the module restarts.
Text is passed downstream unchanged, allowing captions and TTS on the same graph.
Offline jobs do not open a window unless show_in_offline is explicitly enabled.
The GUI runs in an owned process, using bounded queues; excess display updates coalesce to the latest caption.
No subtitle text is placed on command lines, written to logs or retained in status messages.
Requires Windows Python with Tkinter (included with the normal Python Windows installer).
