import unittest

from speech_app.resources import ResourceSnapshot


class ResourceSnapshotTests(unittest.TestCase):
    def test_formats_resource_labels(self):
        snapshot = ResourceSnapshot(ram_mb=1536.4, cpu_percent=12.3, threads=8)

        self.assertEqual(snapshot.ram_label, "1.50 GB")
        self.assertEqual(snapshot.cpu_label, "12.3%")
        self.assertEqual(snapshot.threads_label, "8")


if __name__ == "__main__":
    unittest.main()
