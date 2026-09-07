# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GPU dataclass registration and launch-time argument marshalling."""

from typing import Any, Protocol

from typing_extensions import TypeVar

_DataclassT = TypeVar("_DataclassT")

class _GpuDataclassArgumentHandler(Protocol):
    """Launch-time marshalling extension for registered GPU dataclasses."""

    def prepare_args(
        self,
        ty: Any,
        val: Any,
        stream: Any = None,
        retr: list[Any] | None = None,
    ) -> tuple[Any, Any]:
        """Flatten a registered dataclass and preserve its compiler type."""

gpu_dataclass_argument_handler: _GpuDataclassArgumentHandler

def gpu_dataclass(
    dc: _DataclassT,
    *,
    compute_temp_storage: bool = True,
) -> _DataclassT:
    """Register a dataclass instance for Numba-CUDA-MLIR device use."""

__all__ = ["gpu_dataclass", "gpu_dataclass_argument_handler"]
