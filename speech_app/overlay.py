from __future__ import annotations

import math
import time
import tkinter as tk
from collections import deque


class VoiceOverlay:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.transparent = "#ff00ff"
        self.window = tk.Toplevel(root)
        self.window.withdraw()
        self.window.overrideredirect(True)
        self.window.attributes("-topmost", True)
        try:
            self.window.attributes("-alpha", 0.86)
        except tk.TclError:
            pass
        try:
            self.window.attributes("-transparentcolor", self.transparent)
        except tk.TclError:
            pass
        self.window.configure(bg=self.transparent)
        self._set_no_activate()
        self.width = 170
        self.height = 66
        self.canvas = tk.Canvas(
            self.window,
            width=self.width,
            height=self.height,
            bg=self.transparent,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.levels: deque[float] = deque([0.02] * 14, maxlen=14)
        self.mode = "idle"
        self.message = ""
        self._running = False
        self._show_token = 0

    def show_recording(self) -> None:
        self.mode = "recording"
        self.message = ""
        self._show()

    def show_transcribing(self) -> None:
        self.mode = "transcribing"
        self.message = ""
        self._show()

    def show_notice(self, message: str, timeout_ms: int = 1400) -> None:
        self.mode = "notice"
        self.message = message
        self._show()
        token = self._show_token
        self.root.after(timeout_ms, lambda: self._hide_if_current(token))

    def hide(self) -> None:
        self._show_token += 1
        self._hide_now()

    def _hide_if_current(self, token: int) -> None:
        if token == self._show_token:
            self._hide_now()

    def _hide_now(self) -> None:
        self._running = False
        self.window.withdraw()

    def set_level(self, level: float) -> None:
        self.levels.append(max(0.02, min(1.0, level)))

    def _show(self) -> None:
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        x = int((screen_w - self.width) / 2)
        y = int(screen_h - self.height - 86)
        self.window.geometry(f"{self.width}x{self.height}+{x}+{y}")
        self._show_token += 1
        self._set_no_activate()
        self.window.deiconify()
        self._set_no_activate()
        self._running = True
        self._draw()

    def _set_no_activate(self) -> None:
        try:
            self.window.update_idletasks()
            import ctypes

            hwnd = self.window.winfo_id()
            user32 = ctypes.windll.user32
            ex_style = user32.GetWindowLongW(hwnd, -20)
            user32.SetWindowLongW(hwnd, -20, ex_style | 0x08000000 | 0x00000080)
        except Exception:
            pass

    def _draw(self) -> None:
        self.canvas.delete("all")
        self._draw_shadow()
        self._draw_wave_pill()
        if self._running:
            self.root.after(28, self._draw)

    def _draw_shadow(self) -> None:
        self._rounded_rect(16, 16, 154, 58, 21, fill="#d8d3dd", outline="")
        self._rounded_rect(22, 12, 148, 54, 21, fill="#ece8f0", outline="")

    def _draw_wave_pill(self) -> None:
        pill_fill = "#ffffff"
        if self.mode == "transcribing":
            pill_fill = "#f8f8f7"
        self._rounded_rect(24, 10, 146, 54, 22, fill=pill_fill, outline="#e5e1ea")

        if self.mode == "notice":
            self.canvas.create_text(
                85,
                32,
                text=self.message,
                fill="#17151b",
                font=("Segoe UI", 10, "bold"),
            )
            return

        if self.mode == "transcribing":
            self._draw_loading_dots()
            return

        self._draw_bars()

    def _draw_bars(self) -> None:
        center_y = 32
        left = 49
        phase = time.time() * 4.8
        for index, level in enumerate(self.levels):
            pulse = (math.sin(phase + index * 0.65) + 1.0) * 0.08
            energy = max(level, pulse)
            height = 6 + energy * 26
            x = left + index * 6
            self.canvas.create_line(
                x,
                center_y - height / 2,
                x,
                center_y + height / 2,
                fill="#17151b",
                width=4,
                capstyle=tk.ROUND,
            )

    def _draw_loading_dots(self) -> None:
        tick = int(time.time() * 8) % 6
        for index in range(6):
            radius = 3 if index == tick else 2
            color = "#17151b" if index == tick else "#c8bdd4"
            x = 64 + index * 8
            self.canvas.create_oval(
                x - radius,
                32 - radius,
                x + radius,
                32 + radius,
                fill=color,
                outline=color,
            )

    def _rounded_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        radius: int,
        fill: str,
        outline: str,
    ) -> None:
        points = [
            x0 + radius,
            y0,
            x1 - radius,
            y0,
            x1,
            y0,
            x1,
            y0 + radius,
            x1,
            y1 - radius,
            x1,
            y1,
            x1 - radius,
            y1,
            x0 + radius,
            y1,
            x0,
            y1,
            x0,
            y1 - radius,
            x0,
            y0 + radius,
            x0,
            y0,
        ]
        self.canvas.create_polygon(
            points,
            smooth=True,
            splinesteps=18,
            fill=fill,
            outline=outline,
        )
