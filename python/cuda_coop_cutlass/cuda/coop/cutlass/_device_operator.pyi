# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type a compiled scalar operator used by Reduce and Scan."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

@dataclass(frozen=True)
class DeviceOperator:
    """A typed device operator produced by compile_operator."""

    dtype: type[Any]
    symbol: str
    cpp_type: str
    arch: str
    _ltoir: bytes = field(repr=False)

    @property
    def cpp_definition(self) -> str: ...

def compile_operator(
    func: Callable[..., object],
    dtype: type[Any] | np.dtype[Any],
    *,
    arch: str | None = None,
) -> DeviceOperator:
    """Compile a pure associative CuTe binary function before kernel tracing."""
