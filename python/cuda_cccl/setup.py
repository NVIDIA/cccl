# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from setuptools import setup, find_namespace_packages
import os
import shutil

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
CCCL_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "..", ".."))


def copy_cccl_headers_to_cuda_cccl_include():
    cccl_headers = [["cub", "cub"], ["libcudacxx", "include"], ["thrust", "thrust"]]
    inc_path = os.path.join(PROJECT_PATH, "cuda", "cccl", "include")
    os.makedirs(inc_path, exist_ok=True)
    for proj_dir, header_dir in cccl_headers:
        src_path = os.path.abspath(os.path.join(CCCL_PATH, proj_dir, header_dir))
        dst_path = os.path.join(inc_path, proj_dir)
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
    init_py_path = os.path.join(inc_path, "__init__.py")
    with open(init_py_path, "w") as f:
        f.write("# Intentionally empty.\n")


copy_cccl_headers_to_cuda_cccl_include()

setup(
    license_files=["../../LICENSE"],
    packages=find_namespace_packages(include=["cuda.*"]),
    include_package_data=True,
)
