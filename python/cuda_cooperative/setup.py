# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from setuptools import setup

CCCL_PATH = Path(__file__).resolve().parents[2]

setup(
    license_files=["../../LICENSE"],
    install_requires=[
        f"cuda-cccl @ file://{CCCL_PATH}/python/cuda_cccl",
        "numpy",
        "numba>=0.60.0",
        "pynvjitlink-cu12>=0.2.4",
        "cuda-python",
        "jinja2",
    ],
)
