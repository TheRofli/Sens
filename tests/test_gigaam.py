"""Tests for the GigaAM engine's overlap seam merging."""

import unittest

from speech_app.engines.gigaam import _merge_transcript_parts


class MergeTranscriptPartsTests(unittest.TestCase):
    def test_duplicated_overlap_is_removed(self):
        parts = ["привет мир как дела", "как дела отлично"]
        self.assertEqual(_merge_transcript_parts(parts), "привет мир как дела отлично")

    def test_match_is_case_and_punctuation_insensitive(self):
        # The seam keeps the next chunk's copy, so its punctuation wins.
        parts = ["Мир, как дела.", "как дела? Отлично"]
        self.assertEqual(_merge_transcript_parts(parts), "Мир, как дела? Отлично")

    def test_no_overlap_keeps_both_parts(self):
        parts = ["привет мир", "как дела"]
        self.assertEqual(_merge_transcript_parts(parts), "привет мир как дела")

    def test_single_word_match_is_not_enough_to_dedupe(self):
        # A naturally repeated word must not be eaten by the dedupe.
        parts = ["я думаю все", "все хорошо"]
        self.assertEqual(_merge_transcript_parts(parts), "я думаю все все хорошо")

    def test_multi_chunk_merge(self):
        parts = ["раз два три четыре", "три четыре пять шесть", "пять шесть семь"]
        self.assertEqual(
            _merge_transcript_parts(parts), "раз два три четыре пять шесть семь"
        )

    def test_empty_parts_are_ignored(self):
        parts = ["привет", "", "мир"]
        self.assertEqual(_merge_transcript_parts(parts), "привет мир")

    def test_single_part_is_returned_unchanged(self):
        self.assertEqual(_merge_transcript_parts(["только один кусок"]), "только один кусок")


if __name__ == "__main__":
    unittest.main()
