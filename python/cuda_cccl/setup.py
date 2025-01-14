# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

PROJECT_PATH = Path(__file__).resolve().parent
CCCL_PATH = PROJECT_PATH.parents[1]


class CustomBuildPy(build_py):
    """Copy CCCL headers BEFORE super().run()

    Note that the CCCL headers cannot be referenced directly:
    setuptools (and pyproject.toml) does not support relative paths that
    reference files outside the package directory (like ../../).
    This is a restriction designed to avoid inadvertently packaging files
    that are outside the source tree.
    """

    def run(self):
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

        super().run()


setup(
    license_files=["../../LICENSE"],
    cmdclass={"build_py": CustomBuildPy},
)
