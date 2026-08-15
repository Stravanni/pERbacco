from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
from setuptools.command.build_py import build_py as _build_py

ROOT = Path(__file__).resolve().parent


class BinaryDistribution(Distribution):
    def has_ext_modules(self) -> bool:
        return True


class BuildPy(_build_py):
    def run(self) -> None:
        subprocess.run(["make", "all"], cwd=ROOT, check=True)
        super().run()
        extension = "dylib" if platform.system() == "Darwin" else "so"
        source = ROOT / "build" / f"libperbacco.{extension}"
        destination = Path(self.build_lib) / "perbacco" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


class PlatformWheel(_bdist_wheel):
    def get_tag(self) -> tuple[str, str, str]:
        _python, _abi, platform_tag = super().get_tag()
        return "py3", "none", platform_tag


setup(
    cmdclass={"bdist_wheel": PlatformWheel, "build_py": BuildPy},
    distclass=BinaryDistribution,
    package_data={"perbacco": ["libperbacco.so", "libperbacco.dylib"]},
)
