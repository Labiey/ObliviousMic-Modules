# SPDX-License-Identifier: Apache-2.0
"""Owned GUI child. JSON stdin only; stdout contains status, never subtitle text."""

import json
import queue
import sys
import threading

MAX_LINE = 256 * 1024


def coalesce(inbox, value):
    """Retain newest updates without a producer/consumer full-to-empty race."""
    while True:
        try:
            inbox.put_nowait(value)
            return
        except queue.Full:
            try:
                inbox.get_nowait()
            except queue.Empty:
                pass


def main():
    import tkinter as tk

    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    inbox = queue.Queue(maxsize=64)
    root = tk.Tk()
    root.title("ObliviousMic Subtitles")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#101820")
    header = tk.Frame(root, bg="#101820")
    header.pack(fill="x", padx=12, pady=(6, 0))
    brand = tk.Label(header, text="ObliviousMic", fg="#80dfc6", bg="#101820", font=("Segoe UI", 10))
    brand.pack(side="left")
    body = tk.Text(
        root,
        bg="#101820",
        fg="white",
        wrap="word",
        borderwidth=0,
        highlightthickness=0,
        height=3,
        takefocus=0,
        cursor="arrow",
    )
    body.pack(fill="both", expand=True, padx=16, pady=(4, 12))
    body.configure(state="disabled")
    state = {"generation": -1, "clear_job": None, "placed": False}

    def report(**value):
        print(json.dumps(value, ensure_ascii=True), flush=True)

    def close(user=True):
        report(state="closed", user=user)
        root.destroy()

    tk.Button(
        header,
        text="×",
        command=close,
        fg="white",
        bg="#101820",
        relief="flat",
        borderwidth=0,
        takefocus=0,
    ).pack(side="right")
    drag = {}

    def begin(event):
        drag.update(x=event.x_root - root.winfo_x(), y=event.y_root - root.winfo_y())

    def move(event):
        if drag:
            root.geometry(f"+{event.x_root - drag['x']}+{event.y_root - drag['y']}")

    for widget in (header, brand):
        widget.bind("<Button-1>", begin)
        widget.bind("<B1-Motion>", move)
    root.protocol("WM_DELETE_WINDOW", close)

    def set_text(text):
        body.configure(state="normal")
        body.delete("1.0", "end")
        body.insert("1.0", text)
        body.see("end")
        body.configure(state="disabled")

    def reader():
        try:
            while True:
                line = sys.stdin.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    continue
                coalesce(inbox, value)
        finally:
            coalesce(inbox, {"shutdown": True})

    def click_style(enabled):
        if sys.platform != "win32":
            return
        import ctypes
        from ctypes import wintypes

        user = ctypes.WinDLL("user32", use_last_error=True)
        user.GetParent.argtypes = [wintypes.HWND]
        user.GetParent.restype = wintypes.HWND
        get_long = (
            user.GetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.GetWindowLongW
        )
        set_long = (
            user.SetWindowLongPtrW if ctypes.sizeof(ctypes.c_void_p) == 8 else user.SetWindowLongW
        )
        get_long.argtypes, get_long.restype = [wintypes.HWND, ctypes.c_int], ctypes.c_ssize_t
        set_long.argtypes, set_long.restype = (
            [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t],
            ctypes.c_ssize_t,
        )
        hwnd = user.GetParent(root.winfo_id())
        current = get_long(hwnd, -20)
        style = current | 0x08000000  # WS_EX_NOACTIVATE: subtitles must not steal focus.
        style = style | 0x20 if enabled else style & ~0x20  # WS_EX_TRANSPARENT
        ctypes.set_last_error(0)
        previous = set_long(hwnd, -20, style)
        if not previous and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        applied = get_long(hwnd, -20)
        return {"click_through": bool(applied & 0x20), "no_activate": bool(applied & 0x08000000)}

    def tick():
        latest = None
        while True:
            try:
                value = inbox.get_nowait()
            except queue.Empty:
                break
            if value.get("shutdown"):
                close(False)
                return
            if value.get("generation", 0) >= state["generation"]:
                latest = value
                state["generation"] = value.get("generation", 0)
        if latest is not None:
            config = latest["config"]
            font_size = int(config["font_size"])
            width = min(int(config["width"]), root.winfo_screenwidth() - 20)
            height = min(root.winfo_screenheight() - 20, max(130, int(font_size * 5.2 + 38)))
            if not state["placed"]:
                x = (root.winfo_screenwidth() - width) // 2
                y = 35 if config["position"] == "top" else root.winfo_screenheight() - height - 60
                root.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")
                state["placed"] = True
            else:
                root.geometry(f"{width}x{height}")
            root.attributes("-alpha", float(config["opacity"]))
            body.configure(font=("Microsoft YaHei UI", font_size))
            text = latest.get("text", "")
            set_text(text)
            root.update_idletasks()
            native_style = click_style(config["click_through"]) or {}
            root.deiconify()
            root.update_idletasks()
            if state["clear_job"]:
                root.after_cancel(state["clear_job"])
            state["clear_job"] = root.after(
                int(config["display_seconds"]) * 1000, lambda: set_text("")
            )
            report(
                state="displayed",
                segment_id=latest.get("id", ""),
                revision=latest.get("revision", 0),
                characters=len(text),
                mapped=bool(root.winfo_ismapped()),
                width=root.winfo_width(),
                height=root.winfo_height(),
                **native_style,
            )
        root.after(30, tick)

    root.withdraw()
    root.update_idletasks()
    report(state="ready")
    threading.Thread(target=reader, daemon=True, name="subtitle-input").start()
    root.after(30, tick)
    root.mainloop()


if __name__ == "__main__":
    main()
