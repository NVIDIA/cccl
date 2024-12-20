# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import subprocess

from setuptools import Extension, setup, find_namespace_packages
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel


project_path = os.path.abspath(os.path.dirname(__file__))
cccl_path = os.path.abspath(os.path.join(project_path, "..", ".."))
cccl_headers = [["cub", "cub"], ["libcudacxx", "include"], ["thrust", "thrust"]]
__version__ = None
with open(os.path.join(project_path, "cuda", "parallel", "_version.py")) as f:
    exec(f.read())
assert __version__ is not None
ver = __version__
del __version__


with open("README.md") as f:
    long_description = f.read()


class CustomBuildCommand(build_py):
    def run(self):
        build_py.run(self)


class CustomWheelBuild(bdist_wheel):
    def run(self):
        super().run()


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

        subprocess.check_call(["cmake", cccl_path] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "cccl.c.parallel"],
            cwd=self.build_temp,
        )


setup(
    name="cuda-parallel",
    version=ver,
    description="Experimental Core Library for CUDA Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    packages=find_namespace_packages(include=["cuda.*"]),
    python_requires=">=3.9",
    install_requires=[
        f"cuda-cccl @ file://{cccl_path}/python/cuda_cccl",
        "numba>=0.60.0",
        "cuda-python",
        "jinja2",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-xdist",
            "cupy-cuda12x",
        ]
    },
    cmdclass={
        "build_py": CustomBuildCommand,
        "bdist_wheel": CustomWheelBuild,
        "build_ext": BuildCMakeExtension,
    },
    ext_modules=[CMakeExtension("cuda.parallel.experimental.cccl.c")],
    include_package_data=True,
    license="Apache-2.0 with LLVM exception",
    license_files=("../../LICENSE",),
)
