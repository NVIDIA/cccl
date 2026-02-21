# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numba

from .._common import normalize_dtype_param
from .._types import (
    Algorithm,
    BasePrimitive,
    Constant,
    Dependency,
    DependentArray,
    DependentPythonOperator,
    Invocable,
    TemplateParameter,
    TempStoragePointer,
    numba_type_to_wrapper,
)


class merge_sort_keys(BasePrimitive):
    is_one_shot = True

    def __init__(
        self,
        dtype,
        items_per_thread,
        compare_op,
        value_dtype=None,
        threads_in_warp=32,
        methods=None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        """
        Performs a warp-wide merge sort over blocked keys.

        Example:
            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_merge_sort_api.py
                :language: python
                :dedent:
                :start-after: example-begin imports
                :end-before: example-end imports

            .. literalinclude:: ../../python/cuda_cccl/tests/coop/test_warp_merge_sort_api.py
                :language: python
                :dedent:
                :start-after: example-begin merge-sort
                :end-before: example-end merge-sort
        """
        self.node = node
        self.temp_storage = temp_storage
        self.dtype = normalize_dtype_param(dtype)
        self.value_dtype = (
            normalize_dtype_param(value_dtype) if value_dtype is not None else None
        )
        self.items_per_thread = items_per_thread
        self.compare_op = compare_op
        self.threads_in_warp = threads_in_warp
        self.methods = methods

        method = [
            DependentArray(Dependency("KeyT"), Dependency("ITEMS_PER_THREAD")),
        ]
        if temp_storage is not None:
            method.insert(
                0,
                TempStoragePointer(
                    numba.types.uint8,
                    is_array_pointer=True,
                    name="temp_storage",
                ),
            )
        if self.value_dtype is not None:
            method.append(
                DependentArray(Dependency("ValueT"), Dependency("ITEMS_PER_THREAD"))
            )
        method.append(
            DependentPythonOperator(
                Constant(numba.int8),
                [Dependency("KeyT"), Dependency("KeyT")],
                Dependency("Op"),
                name="compare_op",
            )
        )

        template = Algorithm(
            "WarpMergeSort",
            "Sort",
            "warp_merge_sort",
            ["cub/warp/warp_merge_sort.cuh"],
            [
                TemplateParameter("KeyT"),
                TemplateParameter("ITEMS_PER_THREAD"),
                TemplateParameter("VIRTUAL_WARP_THREADS"),
                TemplateParameter("ValueT"),
            ],
            [method],
            self,
            type_definitions=[numba_type_to_wrapper(self.dtype, methods=methods)]
            if methods is not None
            else None,
            threads=threads_in_warp,
            unique_id=unique_id,
        )
        self.algorithm = template
        self.specialization = template.specialize(
            {
                "KeyT": self.dtype,
                "VIRTUAL_WARP_THREADS": threads_in_warp,
                "ITEMS_PER_THREAD": items_per_thread,
                "ValueT": self.value_dtype or "::cub::NullType",
                "Op": compare_op,
            }
        )

    @classmethod
    def create(
        cls,
        dtype,
        items_per_thread,
        compare_op,
        value_dtype=None,
        threads_in_warp=32,
        methods=None,
    ):
        algo = cls(
            dtype=dtype,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            value_dtype=value_dtype,
            threads_in_warp=threads_in_warp,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


class merge_sort_pairs(merge_sort_keys):
    def __init__(
        self,
        keys,
        values,
        items_per_thread,
        compare_op,
        threads_in_warp=32,
        methods=None,
        unique_id=None,
        temp_storage=None,
        node=None,
    ):
        super().__init__(
            dtype=keys,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            value_dtype=values,
            threads_in_warp=threads_in_warp,
            methods=methods,
            unique_id=unique_id,
            temp_storage=temp_storage,
            node=node,
        )

    @classmethod
    def create(
        cls,
        keys,
        values,
        items_per_thread,
        compare_op,
        threads_in_warp=32,
        methods=None,
    ):
        algo = cls(
            keys=keys,
            values=values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            methods=methods,
        )
        specialization = algo.specialization
        return Invocable(
            ltoir_files=specialization.get_lto_ir(threads=threads_in_warp),
            temp_storage_bytes=specialization.temp_storage_bytes,
            temp_storage_alignment=specialization.temp_storage_alignment,
            algorithm=specialization,
        )


def _build_merge_sort_keys_spec(
    dtype,
    items_per_thread,
    compare_op,
    value_dtype=None,
    threads_in_warp=32,
    methods=None,
):
    return {
        "dtype": dtype,
        "items_per_thread": items_per_thread,
        "compare_op": compare_op,
        "value_dtype": value_dtype,
        "threads_in_warp": threads_in_warp,
        "methods": methods,
    }


def _build_merge_sort_pairs_spec(
    keys,
    values,
    items_per_thread,
    compare_op,
    threads_in_warp=32,
    methods=None,
):
    return {
        "keys": keys,
        "values": values,
        "items_per_thread": items_per_thread,
        "compare_op": compare_op,
        "threads_in_warp": threads_in_warp,
        "methods": methods,
    }


def _make_merge_sort_keys_two_phase(
    dtype,
    items_per_thread,
    compare_op,
    value_dtype=None,
    threads_in_warp=32,
    methods=None,
):
    return merge_sort_keys.create(
        **_build_merge_sort_keys_spec(
            dtype=dtype,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            value_dtype=value_dtype,
            threads_in_warp=threads_in_warp,
            methods=methods,
        )
    )


def _make_merge_sort_keys_rewrite(
    dtype,
    items_per_thread,
    compare_op,
    value_dtype=None,
    threads_in_warp=32,
    methods=None,
    unique_id=None,
    temp_storage=None,
    node=None,
):
    spec = _build_merge_sort_keys_spec(
        dtype=dtype,
        items_per_thread=items_per_thread,
        compare_op=compare_op,
        value_dtype=value_dtype,
        threads_in_warp=threads_in_warp,
        methods=methods,
    )
    spec.update(
        {
            "unique_id": unique_id,
            "temp_storage": temp_storage,
            "node": node,
        }
    )
    return merge_sort_keys(**spec)


def _make_merge_sort_pairs_two_phase(
    keys,
    values,
    items_per_thread,
    compare_op,
    threads_in_warp=32,
    methods=None,
):
    return merge_sort_pairs.create(
        **_build_merge_sort_pairs_spec(
            keys=keys,
            values=values,
            items_per_thread=items_per_thread,
            compare_op=compare_op,
            threads_in_warp=threads_in_warp,
            methods=methods,
        )
    )


def _make_merge_sort_pairs_rewrite(
    keys,
    values,
    items_per_thread,
    compare_op,
    threads_in_warp=32,
    methods=None,
    unique_id=None,
    temp_storage=None,
    node=None,
):
    spec = _build_merge_sort_pairs_spec(
        keys=keys,
        values=values,
        items_per_thread=items_per_thread,
        compare_op=compare_op,
        threads_in_warp=threads_in_warp,
        methods=methods,
    )
    spec.update(
        {
            "unique_id": unique_id,
            "temp_storage": temp_storage,
            "node": node,
        }
    )
    return merge_sort_pairs(**spec)
