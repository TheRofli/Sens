import tempfile
import unittest
from pathlib import Path

from speech_app.single_instance import SingleInstanceLock


class SingleInstanceLockTests(unittest.TestCase):
    def test_second_lock_cannot_be_acquired_until_first_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "speech.lock"
            first = SingleInstanceLock(path)
            second = SingleInstanceLock(path)

            self.assertTrue(first.acquire())
            self.assertFalse(second.acquire())

            first.release()
            self.assertTrue(second.acquire())
            second.release()


if __name__ == "__main__":
    unittest.main()
