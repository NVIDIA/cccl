# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import shutil

from setuptools import setup, find_namespace_packages


project_path = os.path.abspath(os.path.dirname(__file__))
cccl_path = os.path.abspath(os.path.join(project_path, "..", ".."))
cccl_headers = [["cub", "cub"], ["libcudacxx", "include"], ["thrust", "thrust"]]
ver = "0.1.2.8.0"


with open("README.md") as f:
    long_description = f.read()


def copy_cccl_headers_to_cuda_include():
    for proj_dir, header_dir in cccl_headers:
        src_path = os.path.abspath(os.path.join(cccl_path, proj_dir, header_dir))
        dst_path = os.path.join(project_path, "cuda", "_include", proj_dir)
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
