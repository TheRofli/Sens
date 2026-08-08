import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SENS_ROOT = Path(__file__).resolve().parents[3]
SPEECH_ROOT = SENS_ROOT / "sidecars" / "speech"
WORKER = SENS_ROOT / "sidecars" / "hearing-worker.py"


class HearingWorkerProtocolTests(unittest.TestCase):
    def test_status_is_protocol_only_and_does_not_arm_microphone(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = {
                **os.environ,
                "SENS_SPEECH_ROOT": str(SPEECH_ROOT),
                "SPEECH_DATA_DIR": str(Path(tmp) / "data"),
                "SPEECH_MODELS_DIR": str(Path(tmp) / "models"),
                "SENS_LEGACY_SPEECH_ROOT": str(Path(tmp) / "no-legacy"),
            }
            messages = "\n".join(
                json.dumps(
                    {
                        "requestId": f"status-{index}",
                        "operation": "dictation_status",
                        "input": {},
                    }
                )
                for index in range(2)
            )
            completed = subprocess.run(
                [sys.executable, str(WORKER)],
                input=messages + "\n",
                text=True,
                encoding="utf-8",
                capture_output=True,
                env=environment,
                cwd=SENS_ROOT,
                timeout=15,
                check=True,
            )

        lines = completed.stdout.splitlines()
        self.assertEqual(len(lines), 2)
        responses = [json.loads(line) for line in lines]
        self.assertTrue(all(response["ok"] for response in responses))
        for response in responses:
            status = response["result"]
            self.assertFalse(status["running"])
            self.assertFalse(status["modelControlledMicrophone"])
            self.assertEqual(status["hotkey"], "ctrl+win")


if __name__ == "__main__":
    unittest.main()
