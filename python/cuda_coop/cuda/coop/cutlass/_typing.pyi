# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private scalar typing shared by qualified CUTLASS operations."""

from typing_extensions import TypeVar

from .._typing import CommonNumericScalar

ScalarT = TypeVar("ScalarT", bound=CommonNumericScalar)
CutlassNumericT = TypeVar("CutlassNumericT", bound=CommonNumericScalar)
ScalarValueT = TypeVar("ScalarValueT", bound=CommonNumericScalar)
