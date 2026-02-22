# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from typing import TYPE_CHECKING

import numba

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
)
from .._enums import (
    BlockLoadAlgorithm,
    BlockStoreAlgorithm,
)
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPointer,
    DependentReference,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    Value,
)
from .._typing import (
    DimType,
    DtypeType,
)

if TYPE_CHECKING:
    from ._rewrite import CoopNode

CUB_BLOCK_LOAD_ALGOS = {
    "direct": "::cub::BLOCK_LOAD_DIRECT",
    "striped": "::cub::BLOCK_LOAD_STRIPED",
    "vectorize": "::cub::BLOCK_LOAD_VECTORIZE",
    "transpose": "::cub::BLOCK_LOAD_TRANSPOSE",
    "warp_transpose": "::cub::BLOCK_LOAD_WARP_TRANSPOSE",
    "warp_transpose_timesliced": "::cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED",
}

CUB_BLOCK_STORE_ALGOS = {
    "direct": "::cub::BLOCK_STORE_DIRECT",
    "striped": "::cub::BLOCK_STORE_STRIPED",
    "vectorize": "::cub::BLOCK_STORE_VECTORIZE",
    "transpose": "::cub::BLOCK_STORE_TRANSPOSE",
    "warp_transpose": "::cub::BLOCK_STORE_WARP_TRANSPOSE",
    "warp_transpose_timesliced": "::cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED",
}


class base_load_store(BasePrimitive):
    is_one_shot = True

    template_parameters = [
        TemplateParameter("T"),
        TemplateParameter("BLOCK_DIM_X"),
        TemplateParameter("ITEMS_PER_THREAD"),
        TemplateParameter("ALGORITHM"),
        TemplateParameter("BLOCK_DIM_Y"),
        TemplateParameter("BLOCK_DIM_Z"),
    ]

    def __init__(
        self,
        dtype: DtypeType,
        dim: DimType,
        items_per_thread: int,
        algorithm=None,
        num_valid_items=None,
        oob_default=None,
        unique_id: int = None,
        node: "CoopNode" = None,
        temp_storage=None,
    ) -> None:
        """
        Create a block load/store primitive backed by ``cub::BlockLoad`` or
        ``cub::BlockStore``.

        Example:
            The snippet below demonstrates using block load and store
            invocables.

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
                :language: python
                :dedent:
                :start-after: example-begin load_store
                :end-before: example-end load_store

        :param dtype: Element dtype for source/destination items.
        :type dtype: DtypeType

        :param dim: CUDA block dimensions as an int or ``(x, y, z)`` tuple.
        :type dim: DimType

        :param items_per_thread: Number of items processed by each thread.
        :type items_per_thread: int

        :param algorithm: Optional load/store algorithm selector.
        :type algorithm: str | int | enum, optional

        :param num_valid_items: Optional valid-item count for guarded loads.
        :type num_valid_items: Any, optional

        :param oob_default: Optional out-of-bounds default value for load APIs.
            Requires ``num_valid_items``.
        :type oob_default: Any, optional

        :param unique_id: Optional unique suffix used for generated symbols.
        :type unique_id: int, optional

        :param node: Internal rewrite node used by single-phase rewriting.
        :type node: CoopNode, optional

        :param temp_storage: Optional explicit temporary storage argument.
        :type temp_storage: Any, optional

        :raises ValueError: If ``algorithm`` is invalid.
        :raises ValueError: If ``oob_default`` is provided for store APIs.
        :raises ValueError: If ``oob_default`` is provided without
            ``num_valid_items``.
        """
        self.node = node
        self.dtype = normalize_dtype_param(dtype)
        self.dim = normalize_dim_param(dim)
        self.items_per_thread = items_per_thread
        self.num_valid_items = num_valid_items
        self.oob_default = oob_default
        self.unique_id = unique_id
        if algorithm is None:
            algorithm_enum = self.default_algorithm
        elif isinstance(algorithm, str):
            enum_cls = self.default_algorithm.__class__
            try:
                algorithm_enum = enum_cls[algorithm.upper()]
            except KeyError as exc:
                raise ValueError(f"Invalid algorithm: {algorithm}") from exc
        elif isinstance(algorithm, int):
            algorithm_enum = self.default_algorithm.__class__(algorithm)
        elif isinstance(algorithm, self.default_algorithm.__class__):
            algorithm_enum = algorithm
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

        self.algorithm_enum = algorithm_enum
        (algorithm_cub, _) = self.resolve_cub_algorithm(algorithm_enum)

        input_is_array_pointer = True

        parameters = [
            [
                DependentPointer(
                    value_dtype=Dependency("T"),
                    restrict=True,
                    is_array_pointer=input_is_array_pointer,
                    name="src",
                ),
                DependentArray(
                    value_dtype=Dependency("T"),
                    size=Dependency("ITEMS_PER_THREAD"),
                    name="dst",
                ),
            ]
        ]
        if num_valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="num_valid_items"))
        if oob_default is not None:
            if self.method_name != "Load":
                raise ValueError("oob_default is only valid for BlockLoad")
            if num_valid_items is None:
                raise ValueError("oob_default requires num_valid_items to be set")
            parameters[0].append(
                DependentReference(Dependency("T"), name="oob_default")
            )
        if temp_storage is not None:
            parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8, is_array_pointer=True, name="temp_storage"
                ),
            )
        self.parameters = parameters

        self.algorithm = Algorithm(
            self.struct_name,
            self.method_name,
            self.c_name,
            self.includes,
            self.template_parameters,
            self.parameters,
            self,
            unique_id=unique_id,
        )
        self.specialization = self.algorithm.specialize(
            {
                "T": self.dtype,
                "BLOCK_DIM_X": self.dim[0],
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": algorithm_cub,
                "BLOCK_DIM_Y": self.dim[1],
                "BLOCK_DIM_Z": self.dim[2],
            }
        )
        self.temp_storage = temp_storage

    @classmethod
    def create(
        cls,
        dtype: DtypeType,
        threads_per_block: DimType,
        items_per_thread: int,
        algorithm=None,
    ):
        algo = cls(dtype, threads_per_block, items_per_thread, algorithm)
        specialization = algo.specialization

        return Invocable(
            ltoir_files=specialization.get_lto_ir(),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class load(base_load_store):
    default_algorithm = BlockLoadAlgorithm.DIRECT
    cub_algorithm_map = CUB_BLOCK_LOAD_ALGOS
    struct_name = "BlockLoad"
    method_name = "Load"
    c_name = "block_load"
    includes = ["cub/block/block_load.cuh"]


class store(base_load_store):
    default_algorithm = BlockStoreAlgorithm.DIRECT
    cub_algorithm_map = CUB_BLOCK_STORE_ALGOS
    struct_name = "BlockStore"
    method_name = "Store"
    c_name = "block_store"
    includes = ["cub/block/block_store.cuh"]


def _normalize_threads_per_block(kwargs, threads_per_block):
    kw = dict(kwargs)
    if threads_per_block is None:
        threads_per_block = kw.pop("dim", None)
    return kw, threads_per_block


def _build_load_spec(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    **kwargs,
):
    kw, threads_per_block = _normalize_threads_per_block(kwargs, threads_per_block)
    spec = {
        "dtype": dtype,
        "threads_per_block": threads_per_block,
        "items_per_thread": items_per_thread,
        "algorithm": algorithm,
    }
    spec.update(kw)
    return spec


def _build_store_spec(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    **kwargs,
):
    kw, threads_per_block = _normalize_threads_per_block(kwargs, threads_per_block)
    spec = {
        "dtype": dtype,
        "threads_per_block": threads_per_block,
        "items_per_thread": items_per_thread,
        "algorithm": algorithm,
    }
    spec.update(kw)
    return spec


def _make_load_two_phase(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    **kwargs,
):
    spec = _build_load_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        **kwargs,
    )
    return load.create(**spec)


def _make_load_rewrite(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    **kwargs,
):
    spec = _build_load_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        **kwargs,
    )
    spec["dim"] = spec.pop("threads_per_block")
    return load(**spec)


def _make_store_two_phase(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    **kwargs,
):
    spec = _build_store_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        **kwargs,
    )
    return store.create(**spec)


def _make_store_rewrite(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    **kwargs,
):
    spec = _build_store_spec(
        dtype=dtype,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        **kwargs,
    )
    spec["dim"] = spec.pop("threads_per_block")
    return store(**spec)
