# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import os
import glob
import shutil

from setuptools import Command, setup, find_packages, find_namespace_packages
from setuptools.command.build_py import build_py
from wheel.bdist_wheel import bdist_wheel


project_path = os.path.abspath(os.path.dirname(__file__))
cccl_path = os.path.abspath(os.path.join(project_path, "..", '..'))
cccl_headers = [
    ['cub', 'cub'],
    ['libcudacxx', 'include'],
    ['thrust', 'thrust']
]
with open(os.path.join(project_path, 'cuda', 'cooperative', '_version.py')) as f:
    exec(f.read())
ver = __version__
del __version__


with open("README.md") as f:
    long_description = f.read()


class CustomBuildCommand(build_py):
    def run(self):
        self.run_command('package_cccl')
        build_py.run(self)


class CustomWheelBuild(bdist_wheel):

    def run(self):
        self.run_command('package_cccl')
        super().run()


class PackageCCCLCommand(Command):
    description = 'Generate additional files'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for proj_dir, header_dir in cccl_headers:
            src_path = os.path.abspath(
                os.path.join(cccl_path, proj_dir, header_dir))
            dst_path = os.path.join(project_path, 'cuda', '_include', proj_dir)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


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
    packages=find_namespace_packages(include=['cuda.*']),
    python_requires='>=3.9',
    install_requires=[
        "numba>=0.60.0",
        "pynvjitlink-cu12>=0.2.4",
        "cuda-python",
        "jinja2"
    ],
    extras_require={
        "test": [
            "pytest",
        ]
    },
    cmdclass={
        'package_cccl': PackageCCCLCommand,
        'build_py': CustomBuildCommand,
        'bdist_wheel': CustomWheelBuild,
    },
    include_package_data=True,
    license="Apache-2.0 with LLVM exception",
    license_files = ('../../LICENSE',),
)
