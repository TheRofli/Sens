import unittest

from speech_app.longform import merge_overlapping_transcripts


class MergeOverlappingTranscriptsTests(unittest.TestCase):
    def test_drops_multiword_overlap(self):
        self.assertEqual(
            merge_overlapping_transcripts(
                ["привет мир как дела", "как дела отлично"]
            ),
            "привет мир как дела отлично",
        )

    def test_match_ignores_case_and_punctuation(self):
        self.assertEqual(
            merge_overlapping_transcripts(["Мир, как дела.", "как дела? Отлично"]),
            "Мир, как дела? Отлично",
        )

    def test_single_word_match_is_not_removed(self):
        self.assertEqual(
            merge_overlapping_transcripts(["я думаю всё", "всё хорошо"]),
            "я думаю всё всё хорошо",
        )

    def test_empty_parts_are_ignored(self):
        self.assertEqual(
            merge_overlapping_transcripts(["привет", "", "мир"]),
            "привет мир",
        )


if __name__ == "__main__":
    unittest.main()
