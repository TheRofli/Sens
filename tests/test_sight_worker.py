import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

_candidates = [
    Path(os.environ.get("SENS_SIDECARS_ROOT", "")) / "sight-worker.py",
    Path(r"D:\Sens\sidecars") / "sight-worker.py",
    Path(__file__).resolve().parent.parent / "sidecars" / "sight-worker.py",
]
_worker_path = next((p for p in _candidates if p.is_file()), None)
if _worker_path is None:
    raise SystemExit("sight-worker.py not found (set SENS_SIDECARS_ROOT)")
_spec = importlib.util.spec_from_file_location("sight_worker", _worker_path)
sight_worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sight_worker)
analyze = sight_worker.analyze
locate_text = sight_worker.locate_text
inspect_target = sight_worker.inspect_target
compare_images = sight_worker.compare_images


def write_test_image(path: Path) -> None:
    """Simple light UI-like image: dark text lines on a light background."""
    import cv2

    image = np.full((400, 700, 3), 250, dtype=np.uint8)
    cv2.putText(image, "SENS LOCAL VISION TEST", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    cv2.putText(image, "Invoice #8472 amount 1234.56", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    cv2.rectangle(image, (40, 200), (660, 260), (200, 200, 200), -1)
    cv2.putText(image, "Button: Submit", (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    cv2.imwrite(str(path), image)



class CacheIsolatedTest(unittest.TestCase):
    """Tests that run analyze() must isolate SENS_CACHE_DIR: the cache key
    hashes file *content*, so two runs of the same synthetic image collide
    in the real worker cache and replay stale dumps."""

    @classmethod
    def setUpClass(cls):
        cls._cache_dir = tempfile.mkdtemp(prefix="sens-test-cache-")
        cls._prev_cache = os.environ.get("SENS_CACHE_DIR")
        os.environ["SENS_CACHE_DIR"] = cls._cache_dir

    @classmethod
    def tearDownClass(cls):
        if cls._prev_cache is None:
            os.environ.pop("SENS_CACHE_DIR", None)
        else:
            os.environ["SENS_CACHE_DIR"] = cls._prev_cache


class SightWorkerTests(CacheIsolatedTest):
    @classmethod
    def setUpClass(cls):
        # All tests share one throwaway cache dir so they never touch the
        # real worker cache under %LOCALAPPDATA%\Sens\cache\sight.
        cls._cache_dir = tempfile.mkdtemp(prefix="sens-test-cache-")
        os.environ["SENS_CACHE_DIR"] = cls._cache_dir

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SENS_CACHE_DIR", None)

    @staticmethod
    def _restore_cache_env(previous):
        if previous is None:
            os.environ.pop("SENS_CACHE_DIR", None)
        else:
            os.environ["SENS_CACHE_DIR"] = previous

    def test_dump_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            dump = analyze(str(path))

            self.assertEqual(dump["image"]["width"], 700)
            self.assertEqual(dump["image"]["height"], 400)
            self.assertGreaterEqual(len(dump["colors"]), 1)
            self.assertGreaterEqual(len(dump["ocr"]), 2)
            self.assertIsInstance(dump["layout"], list)
            self.assertIsInstance(dump["attention"], list)
            self.assertIsInstance(dump["scene"], list)
            self.assertIsInstance(dump["objects"], list)

    def test_ocr_reads_cyrillic_and_latin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            dump = analyze(str(path))
            texts = " ".join(item["text"] for item in dump["ocr"]).lower()

            self.assertIn("sens", texts)
            self.assertIn("submit", texts)
            self.assertIn("button", texts)

    def test_locate_finds_text_deterministically(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            dump = analyze(str(path))
            result = locate_text(dump, "submit")

            self.assertTrue(result["found"])
            box = result["box"]
            self.assertGreaterEqual(box[0], 0)
            self.assertGreaterEqual(box[1], 0)
            self.assertGreater(box[2], box[0])
            self.assertGreater(box[3], box[1])

    def test_locate_missing_returns_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            dump = analyze(str(path))
            result = locate_text(dump, "несуществующий текст")

            self.assertFalse(result["found"])
            self.assertIsNone(result["box"])

    def test_inspect_target_zooms_and_grounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            result = inspect_target(str(path), "submit")

            self.assertTrue(result["found"])
            x0, y0, x1, y1 = result["grounding"]
            self.assertGreaterEqual(x0, 0)
            self.assertGreaterEqual(y0, 0)
            self.assertGreater(x1, x0)
            self.assertGreater(y1, y0)
            self.assertIsNotNone(result["analysis"])
            self.assertIn("ocr", result["analysis"])

    def test_inspect_missing_target_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            result = inspect_target(str(path), "absent-zzz")

            self.assertFalse(result["found"])
            self.assertIsNone(result["grounding"])
            self.assertIsNone(result["analysis"])

    def test_inspect_region_upscales_small_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            dump = analyze(str(path), {"x": 40, "y": 200, "width": 640, "height": 60})

            self.assertIn("image", dump)
            self.assertIsInstance(dump["ocr"], list)

    def test_cross_verify_marks_text_blocks_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ui.png"
            write_test_image(path)

            dump = analyze(str(path))
            verification = dump["verification"]

            self.assertIsInstance(verification["textBlocks"], list)
            self.assertIsInstance(verification["graphicBlocks"], list)
            self.assertIsInstance(verification["labeledObjects"], list)
            self.assertGreaterEqual(verification["attentionTextCoverage"], 0)
            self.assertLessEqual(verification["attentionTextCoverage"], 1)
            self.assertIsInstance(verification["conflicts"], list)

    def test_cache_hits_on_second_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            previous = os.environ.get("SENS_CACHE_DIR")
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "ui.png"
                write_test_image(path)

                first = analyze(str(path))
                second = analyze(str(path))

                self.assertFalse(first["cached"])
                self.assertTrue(second["cached"])
                self.assertEqual(first["ocr"], second["ocr"])
                self.assertGreaterEqual(len(list(cache_dir.rglob("*.json"))), 1)
            finally:
                self._restore_cache_env(previous)

    def test_cache_invalidates_when_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            previous = os.environ.get("SENS_CACHE_DIR")
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "ui.png"
                write_test_image(path)
                first = analyze(str(path))

                write_test_image(path)  # same content -> same key
                same = analyze(str(path))
                self.assertTrue(same["cached"])

                # Different pixels -> different digest -> fresh analysis.
                import cv2
                import numpy as np
                cv2.imwrite(str(path), np.full((200, 300, 3), 10, dtype=np.uint8))
                changed = analyze(str(path))
                self.assertFalse(changed["cached"])
            finally:
                self._restore_cache_env(previous)

    def test_cache_respects_no_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            previous = os.environ.get("SENS_CACHE_DIR")
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "ui.png"
                write_test_image(path)

                analyze(str(path), None, no_store=True)

                self.assertEqual(len(list(cache_dir.rglob("*.json"))), 0)
            finally:
                self._restore_cache_env(previous)

    def test_gaps_report_spaced_sections(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "spaced.png"
                img = np.full((500, 700, 3), 245, dtype=np.uint8)
                cv2.rectangle(img, (30, 30), (320, 240), (30, 30, 30), 2)
                cv2.rectangle(img, (336, 30), (670, 240), (30, 30, 30), 2)
                cv2.rectangle(img, (30, 256), (670, 460), (30, 30, 30), 2)
                cv2.imwrite(str(path), img)

                dump = analyze(str(path))

                self.assertTrue(dump["gaps"])
                self.assertTrue(all(not gap["touching"] for gap in dump["gaps"]))
                self.assertFalse(any(
                    c["kind"] == "sections_touch_without_gap"
                    for c in dump["verification"]["conflicts"]
                ))
            finally:
                self._restore_cache_env(previous)

    def test_gaps_flag_glued_sections(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "glued.png"
                img = np.full((500, 700, 3), 245, dtype=np.uint8)
                cv2.rectangle(img, (30, 30), (320, 240), (30, 30, 30), 2)
                cv2.rectangle(img, (336, 30), (670, 240), (30, 30, 30), 2)
                cv2.rectangle(img, (30, 240), (670, 460), (30, 30, 30), 2)  # no gap
                cv2.imwrite(str(path), img)

                dump = analyze(str(path))

                touching = [g for g in dump["gaps"] if g["touching"]]
                self.assertTrue(touching, "expected touching gaps")
                self.assertTrue(any(
                    c["kind"] == "sections_touch_without_gap"
                    for c in dump["verification"]["conflicts"]
                ))
            finally:
                self._restore_cache_env(previous)

    def test_design_qa_flags_low_contrast(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "lowcontrast.png"
                img = np.full((300, 500, 3), 245, dtype=np.uint8)  # light bg
                cv2.putText(img, "grayish text", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (205, 205, 205), 2)
                cv2.putText(img, "dark text", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
                cv2.imwrite(str(path), img)

                dump = analyze(str(path))

                issues = [i["kind"] for i in dump["design"]["issues"]]
                self.assertIn("low_text_contrast", issues)
                self.assertTrue(any(
                    c["kind"] == "low_text_contrast"
                    for c in dump["verification"]["conflicts"]
                ))
            finally:
                self._restore_cache_env(previous)

    def test_design_qa_flags_text_overflow(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "clipped.png"
                img = np.full((200, 200, 3), 245, dtype=np.uint8)
                # текст уходит за правый край кадра
                cv2.putText(img, "overflowing text runs beyond", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
                cv2.imwrite(str(path), img)

                dump = analyze(str(path))

                issues = [i["kind"] for i in dump["design"]["issues"]]
                self.assertTrue(
                    "text_clipped_at_frame" in issues or "text_overflows_section" in issues,
                    f"expected overflow issue, got {issues}",
                )
            finally:
                self._restore_cache_env(previous)

    def test_design_qa_flags_uneven_cards(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            previous = os.environ.get("SENS_CACHE_DIR")
            os.environ["SENS_CACHE_DIR"] = str(cache_dir)
            try:
                path = Path(tmp) / "uneven.png"
                img = np.full((400, 700, 3), 245, dtype=np.uint8)
                cv2.rectangle(img, (30, 30), (320, 300), (30, 30, 30), 2)   # высокая карточка
                cv2.rectangle(img, (350, 30), (670, 180), (30, 30, 30), 2)  # ниже
                cv2.imwrite(str(path), img)

                dump = analyze(str(path))

                issues = [i["kind"] for i in dump["design"]["issues"]]
                self.assertIn("uneven_card_heights", issues)
            finally:
                self._restore_cache_env(previous)

    def test_compare_reports_zero_for_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.png"
            write_test_image(path)

            result = compare_images(str(path), str(path))

            self.assertEqual(result["mismatchRatio"], 0.0)
            self.assertEqual(result["zones"], [])

    def test_compare_flags_modified_image(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.png"
            cand = Path(tmp) / "cand.png"
            base = np.full((300, 400, 3), 240, dtype=np.uint8)
            cv2.imwrite(str(ref), base)
            modified = base.copy()
            cv2.rectangle(modified, (50, 50), (200, 150), (20, 20, 20), -1)
            cv2.imwrite(str(cand), modified)

            result = compare_images(str(ref), str(cand))

            self.assertGreater(result["mismatchRatio"], 0.01)
            self.assertTrue(result["zones"])
            self.assertEqual(result["zones"][0]["box"][0], 50)
            self.assertEqual(result["zones"][0]["box"][1], 50)

    def test_section_style_extracts_tokens(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cards.png"
            img = np.full((400, 700, 3), 245, dtype=np.uint8)
            cv2.rectangle(img, (30, 30), (320, 240), (30, 30, 30), 2)
            cv2.rectangle(img, (336, 30), (670, 240), (30, 30, 30), 2)
            cv2.putText(img, "Card one", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            styles = dump["sectionStyle"]

            self.assertTrue(styles, "expected section styles")
            style = next((s for s in styles if s["padding"]), styles[0])
            self.assertRegex(style["background"], r"^#[0-9A-F]{6}$")
            self.assertIsInstance(style["cornerRadius"], int)
            self.assertIsInstance(style["padding"], dict)
            self.assertIsNotNone(style["fontSize"])

    def test_section_style_detects_border(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bordered.png"
            img = np.full((300, 500, 3), 250, dtype=np.uint8)
            # светлая карточка с синей обводкой
            cv2.rectangle(img, (40, 40), (300, 160), (248, 248, 248), -1)
            cv2.rectangle(img, (40, 40), (300, 160), (228, 115, 0), 1)
            cv2.putText(img, "Bordered", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            styles = dump["sectionStyle"]

            bordered = next((s for s in styles if s["borderColor"]), None)
            self.assertIsNotNone(bordered, "expected a bordered section")
            self.assertEqual(bordered["borderColor"].upper(), "#0073E4")
            self.assertEqual(bordered["borderWidth"], 1)

    def test_controls_report_small_button_styles(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "btn.png"
            img = np.full((200, 300, 3), 250, dtype=np.uint8)
            # маленькая кнопка-капсула со светлой заливкой и синей обводкой
            cv2.rectangle(img, (60, 80), (160, 110), (248, 248, 248), -1)
            cv2.rectangle(img, (60, 80), (160, 110), (228, 115, 0), 1)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))

            self.assertIsInstance(dump["controls"], list)
            outlined = [c for c in dump["controls"] if c["borderColor"]]
            self.assertTrue(outlined, "expected a control with a border")
            control = outlined[0]
            self.assertEqual(control["borderColor"].upper(), "#0073E4")
            self.assertEqual(control["borderWidth"], 1)
            self.assertEqual(control["background"].upper(), "#F8F8F8")

    def test_icons_detect_cross_and_down_arrow(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "icons.png"
            img = np.full((160, 260, 3), 250, dtype=np.uint8)
            # крест (плюс) в левой части — мелкий, как в реальных кнопках
            cv2.line(img, (56, 41), (64, 41), (0, 115, 229), 2)
            cv2.line(img, (60, 37), (60, 45), (0, 115, 229), 2)
            # стрелка вниз в правой части
            cv2.line(img, (180, 35), (180, 70), (100, 100, 100), 2)
            cv2.line(img, (165, 55), (180, 72), (100, 100, 100), 2)
            cv2.line(img, (195, 55), (180, 72), (100, 100, 100), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            kinds = [icon["kind"] for icon in dump["icons"]]

            self.assertIn("cross", kinds, f"expected a cross icon, got {kinds}")
            self.assertIn("arrow_down", kinds, f"expected a down arrow, got {kinds}")

    def test_controls_split_row_into_buttons(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "row.png"
            img = np.full((200, 500, 3), 250, dtype=np.uint8)
            # ряд из двух капсул с обводкой, разделённых зазором
            cv2.rectangle(img, (40, 80), (180, 110), (248, 248, 248), -1)
            cv2.rectangle(img, (40, 80), (180, 110), (228, 115, 0), 1)
            cv2.rectangle(img, (200, 80), (340, 110), (248, 248, 248), -1)
            cv2.rectangle(img, (200, 80), (340, 110), (228, 115, 0), 1)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))

            outlined = [c for c in dump["controls"] if c["borderColor"]]
            self.assertGreaterEqual(len(outlined), 2, "expected the row to split into buttons")
            boxes = sorted(c["box"][0] for c in outlined)
            self.assertLessEqual(boxes[1] - boxes[0], 200)

    def test_layout_keeps_photo_with_large_internal_zone(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "photo.png"
            img = np.full((400, 600, 3), 250, dtype=np.uint8)
            # «фото»: тёмный блок с большой светлой зоной внутри
            cv2.rectangle(img, (40, 40), (560, 360), (90, 90, 90), -1)
            cv2.rectangle(img, (120, 100), (480, 320), (220, 220, 220), -1)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            boxes = [b["box"] for b in dump["layout"]]

            big = next((b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) > 100000), None)
            self.assertIsNotNone(big, "expected the large photo block to be kept")

    def test_shadows_detect_drop_shadow_below_block(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow.png"
            img = np.full((400, 500, 3), 254, dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (350, 250), (180, 180, 180), -1)  # тёмный блок
            # мягкий градиент-«тень» под блоком: 232 -> 255 (20px)
            for i in range(24):
                value = min(255, 232 + i)
                cv2.rectangle(img, (50, 250 + i), (350, 251 + i), (value, value, value), -1)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))

            shadows = [s for s in dump["shadows"] if s["kind"] == "drop_shadow"]
            self.assertTrue(shadows, "expected a drop shadow band")
            self.assertGreaterEqual(shadows[0]["depth"], 10)

    def test_design_qa_flags_misaligned_text_edges(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "misaligned.png"
            img = np.full((300, 500, 3), 250, dtype=np.uint8)
            # два текста в одной колонке с разным отступом слева
            cv2.putText(img, "Left aligned", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.putText(img, "Indented text", (48, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))

            issues = [i["kind"] for i in dump["design"]["issues"]]
            self.assertIn("text_left_edges_misaligned", issues)

    def test_design_qa_does_not_flag_aligned_text(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "aligned.png"
            img = np.full((300, 500, 3), 250, dtype=np.uint8)
            cv2.putText(img, "Left aligned", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.putText(img, "Aligned too", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))

            issues = [i["kind"] for i in dump["design"]["issues"]]
            self.assertNotIn("text_left_edges_misaligned", issues)


if __name__ == "__main__":
    unittest.main()


class FontMetricsTests(CacheIsolatedTest):
    def test_glyph_metrics_reports_cap_and_width(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "text.png"
            img = np.full((200, 600, 3), 10, dtype=np.uint8)
            cv2.putText(img, "START NOW", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (240, 240, 240), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            line = next((o for o in dump["ocr"] if "START" in o["text"].upper()), None)
            self.assertIsNotNone(line)
            m = line.get("metrics")
            self.assertIsNotNone(m, "OCR line should carry glyph metrics")
            self.assertGreater(m["capHeight"], 8)
            self.assertGreater(m["avgGlyphWidth"], 4)
            self.assertGreater(m["fontSize"], 10)

    def test_glyph_metrics_none_on_low_contrast(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "faint.png"
            img = np.full((200, 400, 3), 12, dtype=np.uint8)
            cv2.putText(img, "FAINT", (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (24, 24, 24), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            self.assertTrue(
                all(o.get("metrics") is None for o in dump["ocr"]),
                "low-contrast text must not fabricate font metrics",
            )


class TextureBlockTests(CacheIsolatedTest):
    def test_texture_blocks_detect_large_pattern(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pattern.png"
            img = np.full((700, 1200, 3), 8, dtype=np.uint8)
            # wavy stripes across a big area
            for x in range(1200):
                y0 = 100 + int(60 * np.sin(x / 40.0))
                cv2.line(img, (x, y0), (x, y0 + 200), (180, 180, 180), 3)
            cv2.putText(img, "OVERLAY", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            textures = [b for b in dump["layout"] if b.get("kind") == "texture"]
            self.assertTrue(textures, "large textured pattern should be a texture block")
            biggest = max(textures, key=lambda b: b["area"])
            self.assertGreaterEqual(biggest["area"], 60000)


class DarkButtonTests(CacheIsolatedTest):
    def test_controls_around_text_find_dark_button(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "darkbtn.png"
            img = np.full((260, 500, 3), 4, dtype=np.uint8)
            cv2.rectangle(img, (120, 80), (340, 150), (36, 36, 36), -1)
            cv2.putText(img, "VIEW WORK", (150, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (215, 176, 93), 2)
            cv2.imwrite(str(path), img)

            dump = analyze(str(path))
            buttons = [c for c in dump["controls"] if c["box"][2] - c["box"][0] > 80]
            self.assertTrue(buttons, "dark-on-dark button should be detected via text ring")
            btn = max(buttons, key=lambda c: c["box"][2] - c["box"][0])
            x0, y0, x1, y1 = btn["box"]
            self.assertLessEqual(x0, 140)
            self.assertGreaterEqual(x1, 320)
            self.assertLessEqual(y0, 90)
            self.assertGreaterEqual(y1, 140)
