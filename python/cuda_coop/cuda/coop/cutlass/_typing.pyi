# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private scalar typing shared by CUTLASS primitive families."""

from typing import TypeAlias

from typing_extensions import TypeVar

from .._typing import PortableNumericScalar

ScalarT = TypeVar("ScalarT", bound=PortableNumericScalar)
CutlassNumericT = TypeVar("CutlassNumericT", bound=PortableNumericScalar)
ScalarValueT = TypeVar("ScalarValueT", bound=PortableNumericScalar)

CutlassOrderedItem: TypeAlias = PortableNumericScalar
CutlassPairValueT = TypeVar("CutlassPairValueT", bound=CutlassOrderedItem)
