# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Explicit-only fixture that cannot create its selected evidence item."""

import pytest

pytest.skip("exercise an exact node skipped during collection", allow_module_level=True)


def test_never_collected() -> None:
    pass
