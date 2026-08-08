from __future__ import annotations

import ctypes
import sys
import shutil
import subprocess
import time
from pathlib import Path


class SystemActions:
    def __init__(
        self,
        user32=None,
        paste_sender=None,
        clipboard_writer=None,
        process_starter=None,
        sleep=time.sleep,
    ) -> None:
        self._keyboard_controller = None
        self._user32 = user32
        self._paste_sender = paste_sender
        self._clipboard_writer = clipboard_writer
        self._process_starter = process_starter
        self._sleep = sleep
        self._target_hwnd = None
        self._target_point: tuple[int, int] | None = None

    def remember_active_window(self) -> None:
        user32 = self._get_user32()
        if user32 is None:
            return
        try:
            hwnd = user32.GetForegroundWindow()
        except Exception:
            return
        if hwnd:
            self._target_hwnd = hwnd
        self._remember_cursor_position()

    def restore_active_window(self) -> None:
        user32 = self._get_user32()
        hwnd = self._target_hwnd
        if user32 is None or not hwnd:
            return
        try:
            if hasattr(user32, "IsWindow") and not user32.IsWindow(hwnd):
                return
            attached = self._attach_threads_for_foreground(hwnd)
            try:
                user32.SetForegroundWindow(hwnd)
            finally:
                self._detach_threads(attached)
            self._sleep(0.18)
            self._click_target_point()
        except Exception:
            return

    def copy_to_clipboard(self, text: str) -> None:
        if self._clipboard_writer is not None:
            self._clipboard_writer(text)
            return

        try:
            import pyperclip

            pyperclip.copy(text)
            return
        except ImportError:
            pass

        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            root.clipboard_clear()
            root.clipboard_append(text)
            root.update()
            root.destroy()
        except Exception as exc:
            raise RuntimeError("Could not write to clipboard.") from exc

    def paste_into_active_input(self, text: str = "") -> None:
        if self._paste_sender is not None:
            self._paste_sender(text)
            return
        if text:
            self.copy_to_clipboard(text)
            self._sleep(0.05)
        self._send_ctrl_v()

    def open_tauri_ui(self, speech_root: Path) -> bool:
        release_exe = (
            speech_root
            / "tauri"
            / "src-tauri"
            / "target"
            / "release"
            / "speech-tauri.exe"
        )
        if release_exe.exists():
            return self._start_process([str(release_exe)], speech_root)

        tauri_dir = speech_root / "tauri"
        package_json = tauri_dir / "package.json"
        if not package_json.exists():
            return False

        npm = shutil.which("npm.cmd") or shutil.which("npm")
        if npm is None:
            return False

        return self._start_process([npm, "run", "tauri:dev"], tauri_dir)

    def _get_user32(self):
        if self._user32 is not None:
            return self._user32
        try:
            self._user32 = ctypes.windll.user32
            return self._user32
        except Exception:
            return None

    def _start_process(self, args: list[str], cwd: Path) -> bool:
        if self._process_starter is not None:
            self._process_starter(args, cwd)
            return True
        try:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                args,
                cwd=str(cwd),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            return True
        except Exception:
            return False

    def _attach_threads_for_foreground(self, hwnd) -> list[tuple[int, int]]:
        user32 = self._get_user32()
        if user32 is None or not hasattr(user32, "GetWindowThreadProcessId"):
            return []
        attached: list[tuple[int, int]] = []
        try:
            kernel32 = ctypes.windll.kernel32
            foreground = user32.GetForegroundWindow()
            current_thread = kernel32.GetCurrentThreadId()
            target_thread = user32.GetWindowThreadProcessId(hwnd, None)
            foreground_thread = (
                user32.GetWindowThreadProcessId(foreground, None)
                if foreground
                else 0
            )
            if target_thread:
                user32.AttachThreadInput(current_thread, target_thread, True)
                attached.append((current_thread, target_thread))
            if foreground_thread:
                user32.AttachThreadInput(current_thread, foreground_thread, True)
                attached.append((current_thread, foreground_thread))
        except Exception:
            return attached
        return attached

    def _detach_threads(self, attached: list[tuple[int, int]]) -> None:
        user32 = self._get_user32()
        if user32 is None or not hasattr(user32, "AttachThreadInput"):
            return
        for current_thread, other_thread in attached:
            try:
                user32.AttachThreadInput(current_thread, other_thread, False)
            except Exception:
                pass

    def _remember_cursor_position(self) -> None:
        user32 = self._get_user32()
        if user32 is None or not hasattr(user32, "GetCursorPos"):
            return
        try:
            point = POINT()
            if user32.GetCursorPos(ctypes.byref(point)):
                self._target_point = (point.x, point.y)
        except Exception:
            return

    def _click_target_point(self) -> None:
        user32 = self._get_user32()
        if user32 is None or self._target_point is None:
            return
        try:
            x, y = self._target_point
            user32.SetCursorPos(x, y)
            mouseeventf_leftdown = 0x0002
            mouseeventf_leftup = 0x0004
            user32.mouse_event(mouseeventf_leftdown, 0, 0, 0, 0)
            user32.mouse_event(mouseeventf_leftup, 0, 0, 0, 0)
            self._sleep(0.12)
        except Exception:
            return

    def _send_ctrl_v(self) -> None:
        user32 = self._get_user32()
        if user32 is None:
            self._send_ctrl_v_with_pynput()
            return

        vk_control = 0x11
        vk_v = 0x56

        self._release_hotkey_modifiers()
        self._sleep(0.05)
        user32.keybd_event(vk_control, 0, 0, 0)
        user32.keybd_event(vk_v, 0, 0, 0)
        user32.keybd_event(vk_v, 0, 0x0002, 0)
        user32.keybd_event(vk_control, 0, 0x0002, 0)

    def release_hotkey_modifiers(self) -> None:
        self._release_hotkey_modifiers()

    def _release_hotkey_modifiers(self) -> None:
        user32 = self._get_user32()
        if user32 is None:
            return
        keyeventf_keyup = 0x0002
        for vk in (0x5B, 0x5C, 0xA2, 0xA3, 0x11):
            user32.keybd_event(vk, 0, keyeventf_keyup, 0)

    def _send_ctrl_v_with_pynput(self) -> None:
        try:
            from pynput.keyboard import Controller, Key
        except ImportError as exc:
            raise RuntimeError(
                "pynput is not installed, so Speech cannot press Ctrl+V."
            ) from exc

        if self._keyboard_controller is None:
            self._keyboard_controller = Controller()

        keyboard = self._keyboard_controller
        paste_modifier = getattr(Key, self._paste_modifier_attr())
        with keyboard.pressed(paste_modifier):
            keyboard.press("v")
            keyboard.release("v")

    def _paste_modifier_attr(self) -> str:
        return "cmd" if sys.platform == "darwin" else "ctrl_l"


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
