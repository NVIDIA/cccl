# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
from setuptools import setup
import shutil

PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))
CCCL_PATH = os.path.abspath(os.path.join(PROJECT_PATH, "..", ".."))


def copy_license():
    src = os.path.abspath(os.path.join(CCCL_PATH, "LICENSE"))
    dst = os.path.join(PROJECT_PATH, "LICENSE")
    shutil.copy(src, dst)


setup(
    install_requires=[
        f"cuda-cccl @ file://{CCCL_PATH}/python/cuda_cccl",
        "numpy",
        "numba>=0.60.0",
        "pynvjitlink-cu12>=0.2.4",
        "cuda-python",
        "jinja2",
    ],
)
