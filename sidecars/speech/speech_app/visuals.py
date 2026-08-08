from __future__ import annotations


def enable_dpi_awareness() -> None:
    try:
        import ctypes

        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass

    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
        return
    except Exception:
        pass

    try:
        import ctypes

        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def set_windows_app_id() -> None:
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Speech.Local.Dictation"
        )
    except Exception:
        pass


def create_icon_photo():
    """Tkinter icon photo. Kept for back-compat; the tray uses create_tray_image
    directly (no tkinter dependency). Returns None when tkinter/PIL is absent."""
    try:
        from PIL import ImageTk

        from .tray import create_tray_image

        return ImageTk.PhotoImage(create_tray_image())
    except Exception:
        return None
