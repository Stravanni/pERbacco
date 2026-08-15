from __future__ import annotations

import contextlib
import io
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXAMPLES = (
    ROOT / "examples" / "python" / "quickstart.py",
    ROOT / "examples" / "python" / "custom_oracle.py",
    ROOT / "examples" / "python" / "snapshot.py",
)


class DocumentationAcceptanceTests(unittest.TestCase):
    def test_python_examples_are_executable_and_offline(self) -> None:
        for example in PYTHON_EXAMPLES:
            with self.subTest(example=example.name):
                output = io.StringIO()
                with (
                    patch(
                        "urllib.request.urlopen",
                        side_effect=AssertionError("documentation attempted live HTTP"),
                    ),
                    patch(
                        "socket.create_connection",
                        side_effect=AssertionError("documentation attempted a live socket"),
                    ),
                    patch(
                        "socket.socket.connect",
                        side_effect=AssertionError("documentation attempted a live socket"),
                    ),
                    patch(
                        "socket.getaddrinfo",
                        side_effect=AssertionError("documentation attempted DNS resolution"),
                    ),
                    contextlib.redirect_stdout(output),
                ):
                    runpy.run_path(str(example), run_name="__main__")
                self.assertTrue(output.getvalue().strip())

    def test_c_quickstart_compiles_and_runs_against_public_library(self) -> None:
        source = ROOT / "examples" / "c" / "quickstart.c"
        library = ROOT / "build" / "libperbacco.a"
        self.assertTrue(library.is_file(), "make all must build the public static library")
        compiler = os.environ.get("CC", "cc")
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "perbacco-c-quickstart"
            subprocess.run(
                [
                    compiler,
                    "-std=c99",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Werror",
                    "-Wshadow",
                    "-Wconversion",
                    "-Wstrict-prototypes",
                    "-I",
                    str(ROOT / "include"),
                    str(source),
                    str(library),
                    "-lm",
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=ROOT,
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
        self.assertIn("matches=2", completed.stdout)


if __name__ == "__main__":
    unittest.main()
