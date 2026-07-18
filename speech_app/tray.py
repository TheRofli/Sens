from __future__ import annotations

from typing import Protocol


class TrayApp(Protocol):
    def show_window(self) -> None: ...
    def show_history(self) -> None: ...
    def copy_last_transcript(self) -> None: ...
    def toggle_engine(self) -> None: ...
    def load_model_background(self) -> None: ...
    def unload_model(self) -> None: ...
    def set_device(self, device: str) -> None: ...
    def set_backend(self, backend: str) -> None: ...
    def set_model(self, key: str) -> None: ...
    def quit(self) -> None: ...
    def engine_enabled(self) -> bool: ...
    def current_device(self) -> str: ...
    def current_backend(self) -> str: ...
    def current_model(self) -> str: ...
    def current_model_label(self) -> str: ...
    def available_models(self) -> list[dict[str, object]]: ...
    def model_loaded(self) -> bool: ...
    def model_is_loading(self) -> bool: ...


class TrayController:
    def __init__(self, app: TrayApp) -> None:
        self.app = app
        self.icon = None

    def start(self) -> bool:
        try:
            import pystray
        except ImportError:
            return False

        def item(label, action, **kwargs):
            return pystray.MenuItem(label, lambda icon, menu_item: action(), **kwargs)

        image = create_tray_image()

        menu = pystray.Menu(
            item("Open Speech", self.app.show_window, default=True),
            item("Open History", self.app.show_history),
            item("Copy Last Transcript", self.app.copy_last_transcript),
            pystray.Menu.SEPARATOR,
            item(
                "Engine On",
                self.app.toggle_engine,
                checked=lambda _: self.app.engine_enabled(),
            ),
            pystray.MenuItem(
                lambda _item: f"Load {self.app.current_model_label()}",
                lambda _icon, _item: self.app.load_model_background(),
                enabled=lambda _: not self.app.model_loaded()
                and not self.app.model_is_loading(),
            ),
            pystray.MenuItem(
                lambda _item: f"Loading {self.app.current_model_label()}…",
                lambda _icon, _item: None,
                enabled=False,
                visible=lambda _: self.app.model_is_loading(),
            ),
            pystray.MenuItem(
                lambda _item: f"Unload {self.app.current_model_label()}",
                lambda _icon, _item: self.app.unload_model(),
                enabled=lambda _: self.app.model_loaded()
                and not self.app.model_is_loading(),
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Device",
                pystray.Menu(
                    item(
                        "CPU",
                        lambda: self.app.set_device("cpu"),
                        checked=lambda _: self.app.current_device() == "cpu",
                        radio=True,
                    ),
                    item(
                        "GPU / CUDA",
                        lambda: self.app.set_device("cuda"),
                        checked=lambda _: self.app.current_device() == "cuda",
                        radio=True,
                    ),
                    item(
                        "Auto",
                        lambda: self.app.set_device("auto"),
                        checked=lambda _: self.app.current_device() == "auto",
                        radio=True,
                    ),
                ),
            ),
            pystray.MenuItem(
                "Backend",
                pystray.Menu(
                    item(
                        "Auto",
                        lambda: self.app.set_backend("auto"),
                        checked=lambda _: self.app.current_backend() == "auto",
                        radio=True,
                    ),
                    item(
                        "Transformers",
                        lambda: self.app.set_backend("transformers"),
                        checked=lambda _: self.app.current_backend()
                        == "transformers",
                        radio=True,
                    ),
                    item(
                        "NeMo",
                        lambda: self.app.set_backend("nemo"),
                        checked=lambda _: self.app.current_backend() == "nemo",
                        radio=True,
                    ),
                ),
            ),
            pystray.MenuItem("Model", self._build_model_menu()),
            pystray.Menu.SEPARATOR,
            item("Quit", self.app.quit),
        )
        self.icon = pystray.Icon("Speech", image, "Speech", menu)
        self.icon.run_detached()
        return True

    def stop(self) -> None:
        if self.icon is not None:
            self.icon.stop()
            self.icon = None

    def refresh_menu(self) -> None:
        if self.icon is not None:
            try:
                self.icon.update_menu()
            except Exception:
                pass

    def notify(self, title: str, message: str) -> None:
        if self.icon is not None:
            try:
                self.icon.notify(message, title)
            except Exception:
                pass

    def _build_model_menu(self):
        """Build the Model submenu from the app's available presets.

        Each entry shows the preset label plus a small install/state hint and a
        radio check on the active model. Built lazily on each menu open so the
        active state and install state stay current.
        """
        import functools

        import pystray

        try:
            presets = self.app.available_models()
        except Exception:
            presets = []

        items = []
        for preset in presets:
            key = str(preset["key"])
            label = str(preset["label"])
            installed = bool(preset.get("installed"))
            hint = "" if installed else " (not installed)"
            caption = f"{label}{hint}"

            # pystray's action callback is invoked as action(icon, item) and
            # its arity-checked, while the checked callback is invoked as
            # checked(item) with a single argument. Bind the preset key with
            # functools.partial so we capture per-entry state without extra
            # positional parameters that pystray would reject.
            def _on_select(icon, item, captured_key):
                self.app.set_model(captured_key)

            def _is_active(menu_item, captured_key):
                return self.app.current_model() == captured_key

            items.append(
                pystray.MenuItem(
                    caption,
                    functools.partial(_on_select, captured_key=key),
                    checked=functools.partial(_is_active, captured_key=key),
                    radio=True,
                )
            )
        if not items:
            items.append(pystray.MenuItem("No models", None, enabled=False))
        return pystray.Menu(*items)


def create_tray_image():
    from PIL import Image, ImageDraw

    scale = 4
    size = 64
    image = Image.new("RGBA", (size * scale, size * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def px(value: int) -> int:
        return value * scale

    draw.rounded_rectangle(
        (px(7), px(11), px(57), px(53)),
        radius=px(17),
        fill="#f7f1ff",
        outline="#ded2ef",
        width=px(2),
    )
    draw.rounded_rectangle(
        (px(18), px(19), px(46), px(45)),
        radius=px(12),
        fill="#111111",
    )
    bars = [7, 13, 19, 25, 19, 13, 7]
    x = 23
    for height in bars:
        y0 = 32 - height // 2
        y1 = 32 + height // 2
        draw.rounded_rectangle(
            (px(x), px(y0), px(x + 2), px(y1)),
            radius=px(1),
            fill="#ffffff",
        )
        x += 4

    return image.resize((size, size), Image.Resampling.LANCZOS)
