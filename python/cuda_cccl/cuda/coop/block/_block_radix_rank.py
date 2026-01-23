# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Tuple, Union

import numba

from .._common import (
    CUB_BLOCK_SCAN_ALGOS,
    CudaSharedMemConfig,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
)
from .._types import (
    Algorithm,
    Array,
    BasePrimitive,
    CxxFunction,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    numba_type_to_cpp,
)

if TYPE_CHECKING:
    import numpy as np

    from ._rewrite import CoopNode

TEMPLATE_PARAMETERS = [
    TemplateParameter("BLOCK_DIM_X"),
    TemplateParameter("RADIX_BITS"),
    TemplateParameter("IS_DESCENDING"),
    TemplateParameter("MEMOIZE_OUTER_SCAN"),
    TemplateParameter("INNER_SCAN_ALGORITHM"),
    TemplateParameter("SMEM_CONFIG"),
    TemplateParameter("BLOCK_DIM_Y"),
    TemplateParameter("BLOCK_DIM_Z"),
]

TEMPLATE_PARAMETER_DEFAULTS = {
    "RADIX_BITS": 4,
    "IS_DESCENDING": "false",
    "MEMOIZE_OUTER_SCAN": "true",
    "INNER_SCAN_ALGORITHM": CUB_BLOCK_SCAN_ALGOS["warp_scans"],
    "SMEM_CONFIG": str(CudaSharedMemConfig.BankSizeFourByte),
}


def _get_template_parameter_specializations(
    dim: dim3, radix_bits: int, descending: bool
) -> dict:
    specialization = TEMPLATE_PARAMETER_DEFAULTS.copy()
    specialization.update(
        {
            "BLOCK_DIM_X": dim[0],
            "RADIX_BITS": radix_bits,
            "IS_DESCENDING": "true" if descending else "false",
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }
    )
    return specialization


class radix_rank(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        begin_bit: int,
        end_bit: int,
        descending: bool = False,
        exclusive_digit_prefix: Any = None,
        unique_id: int = None,
        temp_storage: Any = None,
        node: "CoopNode" = None,
    ) -> None:
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")
        if begin_bit is None or end_bit is None:
            raise ValueError("begin_bit and end_bit must be provided")
        if end_bit <= begin_bit:
            raise ValueError("end_bit must be greater than begin_bit")

        self.node = node
        self.temp_storage = temp_storage
        self.dim = normalize_dim_param(threads_per_block)
        self.dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.begin_bit = begin_bit
        self.end_bit = end_bit
        self.descending = descending
        self.exclusive_digit_prefix = exclusive_digit_prefix
        self.unique_id = unique_id

        radix_bits = end_bit - begin_bit
        radix_digits = 1 << radix_bits
        block_threads = self.dim[0] * self.dim[1] * self.dim[2]
        bins_per_thread = max(1, (radix_digits + block_threads - 1) // block_threads)
        self.bins_per_thread = bins_per_thread

        specialization_kwds = _get_template_parameter_specializations(
            self.dim, radix_bits, descending
        )
        specialization_kwds["KeyT"] = self.dtype
        specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread

        key_cpp = numba_type_to_cpp(self.dtype)
        digit_extractor_cpp = (
            f"::cub::BFEDigitExtractor<{key_cpp}>({begin_bit}, {radix_bits})"
        )

        method = [
            Array(self.dtype, items_per_thread, name="keys"),
            Array(numba.int32, items_per_thread, name="ranks"),
            CxxFunction(cpp=digit_extractor_cpp, func_dtype=self.dtype),
        ]

        if exclusive_digit_prefix is not None:
            method.append(
                Array(
                    numba.int32,
                    bins_per_thread,
                    name="exclusive_digit_prefix",
                )
            )

        if temp_storage is not None:
            method.insert(
                0,
                TempStoragePointer(
                    numba.types.uint8, is_array_pointer=True, name="temp_storage"
                ),
            )

        parameters = [method]

        self.algorithm = Algorithm(
            "BlockRadixRank",
            "RankKeys",
            "block_radix_rank",
            ["cub/block/block_radix_rank.cuh"],
            TEMPLATE_PARAMETERS,
            parameters,
            self,
            unique_id=unique_id,
        )
        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype: Union[str, type, "np.dtype", "numba.types.Type"],
        threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3],
        items_per_thread: int,
        begin_bit: int,
        end_bit: int,
        descending: bool = False,
    ) -> Invocable:
        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            begin_bit=begin_bit,
            end_bit=end_bit,
            descending=descending,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )
