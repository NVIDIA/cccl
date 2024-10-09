# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil
import subprocess

from setuptools import Command, Extension, setup, find_packages, find_namespace_packages
from setuptools.command.build_py import build_py
from setuptools.command.build_ext import build_ext
from wheel.bdist_wheel import bdist_wheel


project_path = os.path.abspath(os.path.dirname(__file__))
cccl_path = os.path.abspath(os.path.join(project_path, "..", '..'))
cccl_headers = [
    ['cub', 'cub'],
    ['libcudacxx', 'include'],
    ['thrust', 'thrust']
]
with open(os.path.join(project_path, 'cuda', 'parallel', '_version.py')) as f:
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
            # TODO Extract cccl headers into a standalone package
            dst_path = os.path.join(project_path, 'cuda', '_include', proj_dir)
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCCCL_ENABLE_C=YES',
            '-DCCCL_C_PARALLEL_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', cccl_path] +
                              cmake_args, cwd=self.build_temp)
        subprocess.check_call(
            ['cmake', '--build', '.', '--target', 'cccl.c.parallel'], cwd=self.build_temp)


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
    packages=find_namespace_packages(include=['cuda.*']),
    python_requires='>=3.9',
    install_requires=[
        "numba>=0.60.0",
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
        'build_ext': BuildCMakeExtension
    },
    ext_modules=[CMakeExtension('cuda.parallel.experimental.cccl.c')],
    include_package_data=True,
    license="Apache-2.0 with LLVM exception",
    license_files=('../../LICENSE',),
)
