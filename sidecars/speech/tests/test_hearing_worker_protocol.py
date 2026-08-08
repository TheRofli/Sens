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
    def _environment(self, root: Path) -> dict[str, str]:
        return {
            **os.environ,
            "SENS_SPEECH_ROOT": str(SPEECH_ROOT),
            "SPEECH_DATA_DIR": str(root / "data"),
            "SPEECH_MODELS_DIR": str(root / "models"),
            "SENS_LEGACY_SPEECH_ROOT": str(root / "no-legacy"),
        }

    def test_status_is_protocol_only_and_does_not_arm_microphone(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = self._environment(Path(tmp))
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

    def test_model_status_is_side_effect_free_and_reports_required_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            message = json.dumps(
                {
                    "requestId": "model-status",
                    "operation": "model_status",
                    "input": {"model": "gigaam"},
                }
            )
            completed = subprocess.run(
                [sys.executable, str(WORKER)],
                input=message + "\n",
                text=True,
                encoding="utf-8",
                capture_output=True,
                env=self._environment(Path(tmp)),
                cwd=SENS_ROOT,
                timeout=15,
                check=True,
            )
        result = json.loads(completed.stdout)["result"]
        self.assertEqual(result["model"], "gigaam")
        self.assertFalse(result["model_installed"])
        self.assertFalse(result["installing"])
        self.assertEqual(result["install_phase"], "missing")
        self.assertEqual(result["install_bytes_required"], 170_197_019)


if __name__ == "__main__":
    unittest.main()
