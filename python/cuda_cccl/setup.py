# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil

from setuptools import setup, find_namespace_packages


project_path = os.path.abspath(os.path.dirname(__file__))
cccl_path = os.path.abspath(os.path.join(project_path, "..", ".."))
cccl_headers = [["cub", "cub"], ["libcudacxx", "include"], ["thrust", "thrust"]]
__version__ = None
with open(os.path.join(project_path, "cuda", "cccl", "_version.py")) as f:
    exec(f.read())
assert __version__ is not None
ver = __version__
del __version__


with open("README.md") as f:
    long_description = f.read()


def copy_cccl_headers_to_cuda_include():
    inc_path = os.path.join(project_path, "cuda", "_include")
    init_py_path = os.path.join(inc_path, "__init__.py")
    with open(init_py_path, "w") as f:
        print("# Intentionally empty.", file=f)
    for proj_dir, header_dir in cccl_headers:
        src_path = os.path.abspath(os.path.join(cccl_path, proj_dir, header_dir))
        dst_path = os.path.join(inc_path, proj_dir)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)


copy_cccl_headers_to_cuda_include()

setup(
    name="cuda-cccl",
    version=ver,
    description="Experimental Package with CCCL headers to support JIT compilation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    packages=find_namespace_packages(include=["cuda.*"]),
    python_requires=">=3.9",
    include_package_data=True,
    license="Apache-2.0 with LLVM exception",
    license_files=("../../LICENSE",),
)
