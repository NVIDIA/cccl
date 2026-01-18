# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Any, Callable

import numba

from .._common import CUB_BLOCK_REDUCE_ALGOS, normalize_dim_param, normalize_dtype_param
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    DependentReference,
    Invocable,
    TemplateParameter,
    Value,
    numba_type_to_wrapper,
)
from .._typing import DimType, DtypeType

if TYPE_CHECKING:
    from ._rewrite import CoopNode


class reduce(BasePrimitive):
    is_one_shot = True
    default_algorithm = CUB_BLOCK_REDUCE_ALGOS["warp_reductions"]
    cub_algorithm_map = CUB_BLOCK_REDUCE_ALGOS

    def __init__(
        self,
        dtype: DtypeType,
        threads_per_block: DimType,
        binary_op: Callable,
        items_per_thread: int = 1,
        algorithm=None,
        methods: dict = None,
        unique_id: int = None,
        temp_storage: Any = None,
        num_valid: Any = None,
        use_array_inputs: bool = False,
        node: "CoopNode" = None,
    ) -> None:
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be greater than or equal to 1")

        self.node = node
        self.items_per_thread = items_per_thread
        self.dim = dim = normalize_dim_param(threads_per_block)
        self.dtype = dtype = normalize_dtype_param(dtype)
        self.unique_id = unique_id
        self.temp_storage = temp_storage
        self.binary_op = binary_op
        self.num_valid = num_valid

        (algorithm_cub, algorithm_enum) = self.resolve_cub_algorithm(
            algorithm,
        )

        specialization_kwds = {
            "T": dtype,
            "BLOCK_DIM_X": dim[0],
            "ALGORITHM": algorithm_cub,
            "BLOCK_DIM_Y": dim[1],
            "BLOCK_DIM_Z": dim[2],
        }

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ]

        use_array_inputs = use_array_inputs or items_per_thread > 1
        self.use_array_inputs = use_array_inputs

        if use_array_inputs:
            specialization_kwds["ITEMS_PER_THREAD"] = items_per_thread
            if num_valid is not None:
                raise ValueError("num_valid is not supported for array inputs")

        if binary_op is None:
            cpp_method_name = "Sum"
            if use_array_inputs:
                parameters = [
                    [
                        DependentArray(
                            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="src"
                        ),
                        DependentReference(Dependency("T"), is_output=True),
                    ],
                ]
            else:
                parameters = [
                    [
                        DependentReference(Dependency("T"), name="src"),
                        DependentReference(Dependency("T"), is_output=True),
                    ],
                ]
                if num_valid is not None:
                    parameters[0].insert(1, Value(numba.int32, name="num_valid"))
        else:
            cpp_method_name = "Reduce"
            specialization_kwds["Op"] = binary_op
            if use_array_inputs:
                parameters = [
                    [
                        DependentArray(
                            Dependency("T"), Dependency("ITEMS_PER_THREAD"), name="src"
                        ),
                        DependentPythonOperator(
                            Dependency("T"),
                            [Dependency("T"), Dependency("T")],
                            Dependency("Op"),
                            name="binary_op",
                        ),
                        DependentReference(Dependency("T"), is_output=True),
                    ],
                ]
            else:
                parameters = [
                    [
                        DependentReference(Dependency("T"), name="src"),
                        DependentPythonOperator(
                            Dependency("T"),
                            [Dependency("T"), Dependency("T")],
                            Dependency("Op"),
                            name="binary_op",
                        ),
                        DependentReference(Dependency("T"), is_output=True),
                    ],
                ]
                if num_valid is not None:
                    parameters[0].insert(2, Value(numba.int32, name="num_valid"))

        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]
        else:
            type_definitions = None

        self.algorithm = Algorithm(
            "BlockReduce",
            cpp_method_name,
            "block_reduce",
            ["cub/block/block_reduce.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            unique_id=unique_id,
        )

        self.specialization = self.algorithm.specialize(specialization_kwds)

    @classmethod
    def create(
        cls,
        dtype: DtypeType,
        threads_per_block: DimType,
        binary_op: Callable,
        items_per_thread: int = 1,
        algorithm=None,
        methods: dict = None,
    ):
        algo = cls(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            binary_op=binary_op,
            algorithm=algorithm,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


def BlockReduce(
    dtype: DtypeType,
    threads_per_block: DimType,
    binary_op: Callable,
    items_per_thread: int = 1,
    algorithm=None,
    methods: dict = None,
):
    return reduce.create(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        binary_op=binary_op,
        algorithm=algorithm,
        methods=methods,
    )


class sum(reduce):
    def __init__(
        self,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int = 1,
        algorithm=None,
        methods: dict = None,
        unique_id: int = None,
        temp_storage: Any = None,
        num_valid: Any = None,
        use_array_inputs: bool = False,
        node: "CoopNode" = None,
    ) -> None:
        return super().__init__(
            dtype=dtype,
            threads_per_block=threads_per_block,
            items_per_thread=items_per_thread,
            binary_op=None,
            algorithm=algorithm,
            methods=methods,
            unique_id=unique_id,
            temp_storage=temp_storage,
            num_valid=num_valid,
            use_array_inputs=use_array_inputs,
            node=node,
        )


def BlockSum(
    dtype: DtypeType,
    threads_per_block: DimType,
    items_per_thread: int = 1,
    algorithm=None,
    methods: dict = None,
):
    return sum.create(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        binary_op=None,
        algorithm=algorithm,
        methods=methods,
    )
