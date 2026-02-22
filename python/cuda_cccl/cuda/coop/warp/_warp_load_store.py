# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional

import numba

from .._common import normalize_dtype_param
from .._enums import WarpLoadAlgorithm, WarpStoreAlgorithm
from .._types import (
    Algorithm,
    BasePrimitive,
    Dependency,
    DependentArray,
    DependentPointer,
    DependentValue,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    Value,
    numba_type_to_wrapper,
)

CUB_WARP_LOAD_ALGOS = {
    "direct": "::cub::WARP_LOAD_DIRECT",
    "striped": "::cub::WARP_LOAD_STRIPED",
    "vectorize": "::cub::WARP_LOAD_VECTORIZE",
    "transpose": "::cub::WARP_LOAD_TRANSPOSE",
}

CUB_WARP_STORE_ALGOS = {
    "direct": "::cub::WARP_STORE_DIRECT",
    "striped": "::cub::WARP_STORE_STRIPED",
    "vectorize": "::cub::WARP_STORE_VECTORIZE",
    "transpose": "::cub::WARP_STORE_TRANSPOSE",
}


def _normalize_algorithm_enum(algorithm, default_algorithm, cub_map):
    if algorithm is None:
        return default_algorithm
    if isinstance(algorithm, str):
        if algorithm not in cub_map:
            raise ValueError(f"Invalid algorithm: {algorithm}")
        try:
            return default_algorithm.__class__[algorithm.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid algorithm: {algorithm}") from exc
    if isinstance(algorithm, int):
        return default_algorithm.__class__(algorithm)
    if isinstance(algorithm, default_algorithm.__class__):
        return algorithm
    raise ValueError(f"Invalid algorithm: {algorithm}")


class load(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpLoadAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        oob_default=None,
        methods: Optional[dict] = None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """
        Loads items from global memory into a warp-striped layout.

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
                :language: python
                :dedent:
                :start-after: example-begin load-store
                :end-before: example-end load-store
        """
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")
        if oob_default is not None and num_valid_items is None:
            raise ValueError("oob_default requires num_valid_items to be set")

        self.node = node
        self.temp_storage = temp_storage

        self.dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.threads_in_warp = threads_in_warp
        self.num_valid_items = num_valid_items
        self.oob_default = oob_default
        self.methods = methods
        self.algorithm_enum = _normalize_algorithm_enum(
            algorithm, WarpLoadAlgorithm.DIRECT, CUB_WARP_LOAD_ALGOS
        )
        algorithm_cub = str(self.algorithm_enum)

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
        ]

        parameters = [
            [
                DependentPointer(
                    value_dtype=Dependency("T"),
                    restrict=True,
                    is_array_pointer=True,
                    name="src",
                ),
                DependentArray(
                    value_dtype=Dependency("T"),
                    size=Dependency("ITEMS_PER_THREAD"),
                    name="dst",
                ),
            ]
        ]
        if temp_storage is not None:
            parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8,
                    is_array_pointer=True,
                    name="temp_storage",
                ),
            )
        if num_valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="num_valid_items"))
        if oob_default is not None:
            parameters[0].append(DependentValue(Dependency("T"), name="oob_default"))

        type_definitions = None
        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]

        template = Algorithm(
            "WarpLoad",
            "Load",
            "warp_load",
            ["cub/warp/warp_load.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            threads=threads_in_warp,
            unique_id=unique_id,
        )

        self.algorithm = template
        self.specialization = template.specialize(
            {
                "T": self.dtype,
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": algorithm_cub,
                "LOGICAL_WARP_THREADS": threads_in_warp,
            }
        )

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpLoadAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        oob_default=None,
        methods: Optional[dict] = None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class store(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpStoreAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        methods: Optional[dict] = None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """
        Stores items from a warp-striped layout into global memory.

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_load_store_api.py
                :language: python
                :dedent:
                :start-after: example-begin load-store
                :end-before: example-end load-store
        """
        if items_per_thread < 1:
            raise ValueError("items_per_thread must be >= 1")

        self.node = node
        self.temp_storage = temp_storage

        self.dtype = normalize_dtype_param(dtype)
        self.items_per_thread = items_per_thread
        self.threads_in_warp = threads_in_warp
        self.num_valid_items = num_valid_items
        self.methods = methods
        self.algorithm_enum = _normalize_algorithm_enum(
            algorithm, WarpStoreAlgorithm.DIRECT, CUB_WARP_STORE_ALGOS
        )
        algorithm_cub = str(self.algorithm_enum)

        template_parameters = [
            TemplateParameter("T"),
            TemplateParameter("ITEMS_PER_THREAD"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("LOGICAL_WARP_THREADS"),
        ]

        parameters = [
            [
                DependentPointer(
                    value_dtype=Dependency("T"),
                    restrict=True,
                    is_array_pointer=True,
                    name="dst",
                ),
                DependentArray(
                    value_dtype=Dependency("T"),
                    size=Dependency("ITEMS_PER_THREAD"),
                    name="src",
                ),
            ]
        ]
        if temp_storage is not None:
            parameters[0].insert(
                0,
                TempStoragePointer(
                    numba.types.uint8,
                    is_array_pointer=True,
                    name="temp_storage",
                ),
            )
        if num_valid_items is not None:
            parameters[0].append(Value(numba.types.int32, name="num_valid_items"))

        type_definitions = None
        if methods is not None:
            type_definitions = [numba_type_to_wrapper(dtype, methods=methods)]

        template = Algorithm(
            "WarpStore",
            "Store",
            "warp_store",
            ["cub/warp/warp_store.cuh"],
            template_parameters,
            parameters,
            self,
            type_definitions=type_definitions,
            threads=threads_in_warp,
            unique_id=unique_id,
        )

        self.algorithm = template
        self.specialization = template.specialize(
            {
                "T": self.dtype,
                "ITEMS_PER_THREAD": items_per_thread,
                "ALGORITHM": algorithm_cub,
                "LOGICAL_WARP_THREADS": threads_in_warp,
            }
        )

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread: int,
        threads_in_warp: int = 32,
        algorithm: Optional[WarpStoreAlgorithm] = None,
        num_valid_items: Optional[int] = None,
        methods: Optional[dict] = None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


def _build_load_spec(
    dtype,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Optional[WarpLoadAlgorithm] = None,
    num_valid_items: Optional[int] = None,
    oob_default=None,
    methods: Optional[dict] = None,
):
    return {
        "dtype": dtype,
        "items_per_thread": items_per_thread,
        "threads_in_warp": threads_in_warp,
        "algorithm": algorithm,
        "num_valid_items": num_valid_items,
        "oob_default": oob_default,
        "methods": methods,
    }


def _build_store_spec(
    dtype,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Optional[WarpStoreAlgorithm] = None,
    num_valid_items: Optional[int] = None,
    methods: Optional[dict] = None,
):
    return {
        "dtype": dtype,
        "items_per_thread": items_per_thread,
        "threads_in_warp": threads_in_warp,
        "algorithm": algorithm,
        "num_valid_items": num_valid_items,
        "methods": methods,
    }


def _make_load_two_phase(
    dtype,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Optional[WarpLoadAlgorithm] = None,
    num_valid_items: Optional[int] = None,
    oob_default=None,
    methods: Optional[dict] = None,
):
    return load.create(
        **_build_load_spec(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            oob_default=oob_default,
            methods=methods,
        )
    )


def _make_load_rewrite(
    dtype,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Optional[WarpLoadAlgorithm] = None,
    num_valid_items: Optional[int] = None,
    oob_default=None,
    methods: Optional[dict] = None,
    unique_id=None,
    temp_storage=None,
    node=None,
):
    spec = _build_load_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        methods=methods,
    )
    spec.update(
        {
            "unique_id": unique_id,
            "temp_storage": temp_storage,
            "node": node,
        }
    )
    return load(**spec)


def _make_store_two_phase(
    dtype,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Optional[WarpStoreAlgorithm] = None,
    num_valid_items: Optional[int] = None,
    methods: Optional[dict] = None,
):
    return store.create(
        **_build_store_spec(
            dtype=dtype,
            items_per_thread=items_per_thread,
            threads_in_warp=threads_in_warp,
            algorithm=algorithm,
            num_valid_items=num_valid_items,
            methods=methods,
        )
    )


def _make_store_rewrite(
    dtype,
    items_per_thread: int = 1,
    threads_in_warp: int = 32,
    algorithm: Optional[WarpStoreAlgorithm] = None,
    num_valid_items: Optional[int] = None,
    methods: Optional[dict] = None,
    unique_id=None,
    temp_storage=None,
    node=None,
):
    spec = _build_store_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        threads_in_warp=threads_in_warp,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        methods=methods,
    )
    spec.update(
        {
            "unique_id": unique_id,
            "temp_storage": temp_storage,
            "node": node,
        }
    )
    return store(**spec)
