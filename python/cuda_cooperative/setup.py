# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


project_path = os.path.abspath(os.path.dirname(__file__))
cccl_path = os.path.abspath(os.path.join(project_path, "..", ".."))
cccl_headers = [["cub", "cub"], ["libcudacxx", "include"], ["thrust", "thrust"]]
__version__ = None
with open(os.path.join(project_path, "cuda", "cooperative", "_version.py")) as f:
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


setup(
    name="cuda-cooperative",
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
        "numba>=0.60.0",
        "pynvjitlink-cu12>=0.2.4",
        "cuda-python",
        "jinja2",
    ],
    extras_require={
        "test": [
            "pytest",
            "pytest-xdist",
        ]
    },
    cmdclass={
        "build_py": CustomBuildCommand,
        "bdist_wheel": CustomWheelBuild,
    },
    include_package_data=True,
    license="Apache-2.0 with LLVM exception",
    license_files=("../../LICENSE",),
)
