# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private scalar typing shared by CUTLASS primitive families."""

from typing import TypeAlias

from typing_extensions import TypeVar

from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _PortableNumericScalar as _PortableNumericScalar

_ScalarT = TypeVar("_ScalarT", bound=_PortableNumericScalar)
_CutlassNumericT = TypeVar("_CutlassNumericT", bound=_PortableNumericScalar)
_ScalarValueT = TypeVar("_ScalarValueT", bound=_PortableNumericScalar)

_CutlassOrderedItem: TypeAlias = _PortableNumericScalar
_CutlassPairValueT = TypeVar("_CutlassPairValueT", bound=_CutlassOrderedItem)
