import unittest

from speech_app.textpost import postprocess


class PostprocessTests(unittest.TestCase):
    def test_collapses_repeated_whitespace_and_trims(self):
        self.assertEqual(postprocess("  hello   world  "), "Hello world")

    def test_empty_or_whitespace_returns_empty(self):
        self.assertEqual(postprocess(""), "")
        self.assertEqual(postprocess("    \n\t "), "")

    def test_capitalizes_first_alpha_character(self):
        self.assertEqual(postprocess("привет мир"), "Привет мир")
        self.assertEqual(postprocess("hello there"), "Hello there")

    def test_handles_none_input(self):
        self.assertEqual(postprocess(None), "")

    def test_strips_leading_hallucination_phrase(self):
        # A standalone hallucination on silence collapses to empty.
        self.assertEqual(postprocess("Thank you for watching"), "")
        self.assertEqual(postprocess("  thanks for watching. "), "")

    def test_keeps_legitimate_use_of_short_word_in_sentence(self):
        # "you" inside a longer phrase must not be stripped.
        self.assertEqual(postprocess("see you tomorrow"), "See you tomorrow")

    def test_strips_leading_punctuation_artifacts(self):
        self.assertEqual(postprocess("- hello"), "Hello")

    def test_collapses_newlines(self):
        # Newlines are ASR segmentation artifacts; they collapse to a space.
        self.assertEqual(postprocess("line one\n\nline two"), "Line one line two")


if __name__ == "__main__":
    unittest.main()
