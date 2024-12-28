# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from pathlib import Path
import subprocess

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
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            "-DCCCL_ENABLE_C=YES",
            "-DCCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY=" + extdir,
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", str(CCCL_PATH)] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "cccl.c.parallel"],
            cwd=self.build_temp,
        )


setup(
    license_files=["../../LICENSE"],
    install_requires=[
        f"cuda-cccl @ file://{CCCL_PYTHON_PATH}/cuda_cccl",
        "numba>=0.60.0",
        "cuda-python",
        "jinja2",
    ],
    cmdclass={
        "build_ext": BuildCMakeExtension,
    },
    ext_modules=[CMakeExtension("cuda.parallel.experimental.cccl.c")],
)
