# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable per-thread payload storage."""

from typing import Any, overload

from typing_extensions import TypeVar

from cuda.coop._typing import ThreadDataLike as ThreadDataLike
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar

_PortableNumericT = TypeVar("_PortableNumericT", bound=_PortableNumericScalar)

@overload
def ThreadData(
    items_per_thread: int,
    dtype: type[_PortableNumericT],
) -> ThreadDataLike[_PortableNumericT]:
    """Construct a payload; builtin int and float mean 32-bit dtypes."""

@overload
def ThreadData(
    items_per_thread: int,
    dtype: None = None,
) -> ThreadDataLike[Any]:
    """Omit dtype or use an ``Any``-typed external compiler dtype token."""
