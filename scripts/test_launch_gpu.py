"""Offline launch tests. No GPU, model, git, or cloud call is made."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import launch_gpu
from bootstrap_gpu import UPSTREAM_REVISION


class LaunchTests(unittest.TestCase):
    def application_command(self, *flags):
        """Run main() against a stand-in upstream and return the command it starts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "upstream/.venv/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            checkpoint = root / "model.safetensors"
            checkpoint.touch()
            started = []

            def popen(command, **kwargs):
                started.append(command)
                return SimpleNamespace(poll=lambda: 0, returncode=0)

            argv = ["launch_gpu.py", "--asset-dir", str(root / "assets"), "--checkpoint", str(checkpoint),
                    "--upstream-root", str(root / "upstream"), *flags]
            environment = {key: value for key, value in os.environ.items() if key != "RUNPOD_POD_ID"}
            with patch.object(sys, "argv", argv), patch.dict(os.environ, environment, clear=True), \
                    patch.object(launch_gpu.subprocess, "check_output", return_value=UPSTREAM_REVISION + "\n"), \
                    patch.object(launch_gpu.subprocess, "Popen", popen), \
                    patch.object(launch_gpu.signal, "signal"):
                self.assertEqual(launch_gpu.main(), 0)
        self.assertEqual(len(started), 1)
        return started[0]

    def test_default_launch_keeps_the_preset_chunk(self):
        self.assertNotIn("--chunk-frames", self.application_command())

    def test_four_frame_flag_reaches_the_application(self):
        command = self.application_command("--chunk-frames", "4")
        self.assertEqual(command[command.index("--chunk-frames") + 1], "4")


if __name__ == "__main__":
    unittest.main()
