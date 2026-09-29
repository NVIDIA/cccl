# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cuda import coop


def test_public_stubs_type_check_a_consumer(tmp_path):
    if importlib.util.find_spec("mypy") is None:
        pytest.skip("mypy is not installed")

    # Copy only stubs so implementation annotations cannot hide missing types.
    package = Path(coop.__file__).parent
    stubs = tmp_path / "stubs" / "cuda" / "coop"
    for source in package.rglob("*.pyi"):
        destination = stubs / source.relative_to(package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    shutil.copyfile(package / "py.typed", stubs / "py.typed")

    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        textwrap.dedent(
            """
            from typing import Literal
            import numpy as np
            from typing_extensions import assert_type
            from cuda import coop

            def kernel(source: object, destination: object) -> None:
                block = coop.this_block()
                warp = coop.this_warp().group_by(8)
                values = coop.ThreadData(2, np.int32, alignment=16)
                scratch = coop.TempStorage(64, auto_sync=True)
                assert_type(block, coop.ThreadGroup[Literal["block"]])
                assert_type(values, coop.ThreadDataLike[np.int32])
                assert_type(scratch, coop.TempStorageLike)
                assert_type(coop.load(
                    block, source, values, algorithm="transpose",
                    valid_items=15, oob_default=np.int32(0), offset=1,
                    temp_storage=scratch,
                ), None)
                assert_type(coop.store(
                    warp, destination, values, algorithm="striped",
                ), None)

                # Strict mypy rejects unused ignores if these become accepted.
                coop.TempStorage(64, 16)  # type: ignore[call-arg]
                coop.load(  # type: ignore[call-overload]
                    block, source, values, algorithm="stripd",
                )
                coop.store(  # type: ignore[call-overload]
                    warp, destination, values, algorithm="warp_transpose",
                )
                coop.load(
                    warp,  # type: ignore[arg-type]
                    source, values, temp_storage=scratch,
                )
                coop.register("numba")  # type: ignore[arg-type]
            """
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["MYPYPATH"] = str(tmp_path / "stubs")
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "--disallow-any-unimported",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(consumer),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
