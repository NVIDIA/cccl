# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

CCCL_PYTHON_PATH = Path(__file__).resolve().parents[1]
CCCL_PATH = CCCL_PYTHON_PATH.parent


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        cmake_args = [
            "-DCCCL_ENABLE_C=YES",
            f"-DCCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_temp_path = Path(self.build_temp)
        build_temp_path.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(["cmake", CCCL_PATH] + cmake_args, cwd=build_temp_path)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "cccl.c.parallel"],
            cwd=build_temp_path,
        )


setup(
    license_files=["../../LICENSE"],
    cmdclass={
        "build_ext": BuildCMakeExtension,
    },
    ext_modules=[CMakeExtension("cuda.parallel.experimental.cccl.c")],
)
