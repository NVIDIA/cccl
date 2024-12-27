# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from setuptools import setup, find_namespace_packages
import shutil

PROJECT_PATH = Path(__file__).resolve().parent
CCCL_PATH = PROJECT_PATH.parents[1]


def copy_cccl_headers_to_cuda_cccl_include():
    cccl_headers = [
        ("cub", "cub"),
        ("libcudacxx", "include"),
        ("thrust", "thrust"),
    ]

    inc_path = PROJECT_PATH / "cuda" / "cccl" / "include"
    inc_path.mkdir(parents=True, exist_ok=True)

    for proj_dir, header_dir in cccl_headers:
        src_path = CCCL_PATH / proj_dir / header_dir
        dst_path = inc_path / proj_dir
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)

    init_py_path = inc_path / "__init__.py"
    init_py_path.write_text("# Intentionally empty.\n")


copy_cccl_headers_to_cuda_cccl_include()

setup(
    license_files=["../../LICENSE"],
    packages=find_namespace_packages(include=["cuda.*"]),
    include_package_data=True,
)
