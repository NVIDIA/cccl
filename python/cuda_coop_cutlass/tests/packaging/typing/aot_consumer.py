# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check public provider pack metadata and context manager types."""

from pathlib import Path

from typing_extensions import assert_type

from cuda.coop.cutlass import aot


def check_pack(path: Path) -> None:
    with aot.capture(path, name="reduction") as capture:
        assert_type(capture, None)
    with aot.use(path, mode="required") as selected:
        assert_type(selected, aot.PackInfo | None)
    info = aot.inspect(path)
    assert_type(info.artifact_bytes, int)
    assert_type(info.entries[0], aot.EntryInfo)
