# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict Pyright fixture for common, CUTLASS, and Numba public surfaces."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import assert_type


def _integer_sum(left: int, right: int) -> int:
    return left + right


def _integer_difference(left: int, right: int) -> int:
    return left - right


def _integer_flag(left: int, right: int) -> bool:
    return left != right


def _integer_less(left: int, right: int) -> bool:
    return left < right


if TYPE_CHECKING:
    import numpy as np

    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class _PrefixOp:
        pass

    stateful_op = numba_coop.StatefulFunction(
        _PrefixOp,
        object(),
        name=None,
    )
    assert_type(stateful_op.op, type[_PrefixOp])
    assert_type(stateful_op.dtype, object)
    assert_type(stateful_op.name, str | None)

    common_group = coop.this_block()
    common_warp = coop.this_warp()
    external_dtype: Any = object()
    common_payload = coop.ThreadData(2, int)
    common_numpy_payload = coop.ThreadData(2, np.int32)
    compiler_dtype_payload = coop.ThreadData(2, external_dtype)
    common_scalar: int = 1
    common_numpy_scalar = np.int32(1)
    compiler_scalar: Any = object()
    common_payload[0] = 1
    assert_type(common_payload[0], int)
    assert_type(common_numpy_payload, coop.ThreadDataLike[np.int32])
    assert_type(compiler_dtype_payload, coop.ThreadDataLike[Any])
    assert_type(
        coop.load(common_group, object(), common_payload), coop.ThreadDataLike[int]
    )
    assert_type(coop.reduce(common_group, common_payload), int)
    assert_type(coop.sum(common_group, common_payload), int)
    assert_type(coop.scan(common_group, common_payload), coop.ThreadDataLike[int])
    assert_type(
        coop.exclusive_sum(common_group, common_payload), coop.ThreadDataLike[int]
    )
    assert_type(
        coop.inclusive_sum(common_group, common_payload), coop.ThreadDataLike[int]
    )
    assert_type(
        coop.exclusive_scan(common_group, common_payload), coop.ThreadDataLike[int]
    )
    assert_type(
        coop.inclusive_scan(common_group, common_payload), coop.ThreadDataLike[int]
    )
    assert_type(coop.scan(common_group, common_scalar), int)
    assert_type(coop.exclusive_sum(common_group, common_scalar), int)
    assert_type(coop.inclusive_sum(common_group, common_scalar), int)
    assert_type(coop.exclusive_scan(common_group, common_scalar), int)
    assert_type(coop.inclusive_scan(common_group, common_scalar), int)
    assert_type(coop.scan(common_warp, common_scalar), int)
    assert_type(coop.exclusive_sum(common_warp, common_numpy_scalar), np.int32)
    assert_type(coop.inclusive_sum(common_warp, common_numpy_scalar), np.int32)
    assert_type(coop.exclusive_scan(common_warp, common_scalar), int)
    assert_type(coop.inclusive_scan(common_warp, compiler_scalar), Any)
    assert_type(coop.exchange(common_group, common_payload), coop.ThreadDataLike[int])
    assert_type(
        coop.exchange(common_warp, common_payload, mode="blocked_to_striped"),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.adjacent_difference(common_group, common_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.discontinuity(common_group, common_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(coop.shuffle(common_group, common_payload), coop.ThreadDataLike[int])
    assert_type(
        coop.merge_sort_keys(common_group, common_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_sort_keys(common_group, common_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(common_group, common_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(common_group, common_numpy_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.run_length_decode(
            common_group,
            common_payload,
            common_payload,
            decoded_items_per_thread=2,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.histogram(common_group, common_payload, bins=16),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.histogram(
            common_group,
            common_payload,
            bins=16,
            counter_dtype=np.int64,
        ),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(
        coop.topk_max_keys(common_group, common_payload, 1),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.topk_min_keys(common_group, common_payload, 1),
        coop.ThreadDataLike[int],
    )
    coop.store(common_group, object(), common_payload, algorithm="direct")
    coop.load(common_group, object(), common_payload, algorithm="transpose")
    assert_type(
        coop.reduce(
            common_group,
            common_scalar,
            broadcast=False,
            algorithm="raking",
        ),
        int,
    )
    common_group.sync()
    common_group.sync_aligned()
    assert_type(
        common_group.group_by(1),
        coop.ThreadGroup[Literal["warps_within_block"]],
    )
    assert_type(common_group.rank_as(int, level="block"), int)

    # Backend-only callbacks, pair controls, constructors, and routing knobs do
    # not leak into the common root contract.
    coop.ThreadData.from_values(1)  # pyright: ignore[reportFunctionMemberAccess]
    coop.ThreadData(2, int, alignas=16)  # pyright: ignore[reportCallIssue]
    coop.ThreadData(2, int, values=[1, 2])  # pyright: ignore[reportCallIssue]

    def _add(a: int, b: int) -> int:
        return a + b

    coop.reduce(common_group, common_payload, binary_op=_add)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_group, common_payload, scan_op=_add)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_scan(common_group, common_payload, scan_op=_add)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.inclusive_scan(common_group, common_payload, scan_op=_add)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.radix_sort_keys(common_group, common_payload, values=common_payload)  # pyright: ignore[reportCallIssue]
    coop.load(common_group, object(), common_payload, launch_metadata={})  # pyright: ignore[reportCallIssue]
    coop.reduce(common_group, common_payload, "extra")  # pyright: ignore[reportCallIssue]
    coop.exchange(common_group, common_scalar)  # pyright: ignore[reportArgumentType]
    coop.exchange(common_group, common_payload, common_payload)  # pyright: ignore[reportCallIssue]
    coop.exchange(common_group, common_payload, launch_metadata={})  # pyright: ignore[reportCallIssue]
    coop.exchange(common_group, common_payload, mode="blocked_to_warp_striped")  # pyright: ignore[reportArgumentType]
    coop.exchange(common_group, common_payload, mode="warp_striped_to_blocked")  # pyright: ignore[reportArgumentType]
    coop.load(common_group, object(), common_payload, algorithm="invalid")  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_group, common_payload, algorithm="invalid")  # pyright: ignore[reportCallIssue, reportArgumentType]

    cutlass_group = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_payload = cutlass_coop.ThreadData(2, int, values=[1, 2])
    cutlass_inferred_payload = cutlass_coop.ThreadData(2, values=[1, 2])
    cutlass_external_payload = cutlass_coop.ThreadData(2, external_dtype)
    cutlass_numpy_payload = cutlass_coop.ThreadData(2, np.float32)
    assert_type(cutlass_payload, cutlass_coop.ThreadData[int])
    assert_type(cutlass_inferred_payload, cutlass_coop.ThreadData[int])
    assert_type(cutlass_external_payload, cutlass_coop.ThreadData[Any])
    assert_type(cutlass_numpy_payload, cutlass_coop.ThreadData[np.float32])
    assert_type(cutlass_payload[0], int)
    cutlass_payload[1] = 3
    cutlass_loaded = cutlass_coop.load(
        cutlass_group,
        object(),
        cutlass_payload,
    )
    assert_type(cutlass_loaded, cutlass_coop.ThreadData[int])
    assert_type(cutlass_coop.reduce(cutlass_group, cutlass_loaded), int)
    assert_type(
        cutlass_coop.scan(cutlass_group, cutlass_loaded),
        cutlass_coop.ThreadData[int],
    )
    assert_type(cutlass_coop.scan(cutlass_group, common_scalar), int)
    assert_type(cutlass_coop.exclusive_sum(cutlass_group, common_scalar), int)
    assert_type(cutlass_coop.inclusive_sum(cutlass_group, common_scalar), int)
    assert_type(cutlass_coop.exclusive_scan(cutlass_group, common_scalar), int)
    assert_type(cutlass_coop.inclusive_scan(cutlass_group, common_scalar), int)
    assert_type(cutlass_coop.scan(cutlass_warp, common_scalar), int)
    assert_type(
        cutlass_coop.exclusive_sum(cutlass_warp, common_numpy_scalar),
        np.int32,
    )
    assert_type(
        cutlass_coop.inclusive_sum(cutlass_warp, common_numpy_scalar),
        np.int32,
    )
    assert_type(cutlass_coop.exclusive_scan(cutlass_warp, common_scalar), int)
    assert_type(cutlass_coop.inclusive_scan(cutlass_warp, compiler_scalar), Any)
    assert_type(
        cutlass_coop.exchange(
            cutlass_group,
            cutlass_loaded,
            mode="blocked_to_striped",
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.exchange(cutlass_warp, cutlass_loaded),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.radix_rank(cutlass_group, cutlass_payload),
        cutlass_coop.ThreadData[int],
    )
    cutlass_values = cutlass_coop.ThreadData.from_values(1, 2)
    cutlass_generated = cutlass_coop.ThreadData.from_fn(2, lambda index: index)

    class _CutlassRegisterTensor:
        @property
        def shape(self) -> object:
            return (2,)

        @property
        def memspace(self) -> object:
            return object()

        def __getitem__(self, index: int, /) -> int:
            return index

    cutlass_fragment = cutlass_coop.ThreadData.from_register_tensor(
        _CutlassRegisterTensor(), dtype=int
    )

    class _CutlassProducer:
        def __cuda_coop_thread_data_load__(
            self,
        ) -> cutlass_coop.ThreadData[int]:
            return cutlass_coop.ThreadData(1, int)

    cutlass_produced = cutlass_coop.ThreadData.load(_CutlassProducer())
    assert_type(cutlass_generated, cutlass_coop.ThreadData[int])
    assert_type(cutlass_fragment, cutlass_coop.ThreadData[int])
    assert_type(cutlass_produced, cutlass_coop.ThreadData[int])
    assert_type(
        cutlass_coop.histogram(
            cutlass_group,
            cutlass_loaded,
            bins=16,
            counter_dtype=np.int64,
        ),
        cutlass_coop.ThreadData[np.int64],
    )
    cutlass_coop.merge_sort_pairs(
        cutlass_group,
        cutlass_loaded,
        cutlass_values,
        compare_op="less",
    )
    cutlass_coop.reduce(cutlass_group, cutlass_loaded, binary_op=_integer_sum)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_group, cutlass_loaded, scan_op=_integer_sum)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_loaded,
        difference_op=_integer_difference,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_loaded,
        flag_op=_integer_flag,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.merge_sort_pairs(  # pyright: ignore[reportCallIssue]
        cutlass_group,
        cutlass_loaded,
        cutlass_values,
        compare_op=_integer_less,  # pyright: ignore[reportArgumentType]
    )
    cutlass_pairs = cutlass_coop.topk_max_pairs(
        cutlass_group,
        cutlass_loaded,
        cutlass_values,
        1,
    )
    assert_type(
        cutlass_pairs,
        tuple[cutlass_coop.ThreadData[int], cutlass_coop.ThreadData[int]],
    )
    cutlass_coop.ThreadData(2, int, alignas=16)  # pyright: ignore[reportCallIssue]
    cutlass_coop.exchange(cutlass_group, common_scalar)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.exchange(cutlass_group, cutlass_loaded, cutlass_loaded)  # pyright: ignore[reportCallIssue]
    cutlass_coop.exchange(cutlass_group, cutlass_loaded, invented=True)  # pyright: ignore[reportCallIssue]
    cutlass_coop.exchange(
        cutlass_group,
        cutlass_loaded,
        mode="scatter_to_striped",
        ranks=cutlass_loaded,
    )

    numba_group = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_payload = numba_coop.ThreadData(2, int, alignas=16)
    assert_type(numba_payload, coop.ThreadDataLike[int])
    numba_loaded = numba_coop.load(numba_group, object(), numba_payload)
    assert_type(numba_loaded, coop.ThreadDataLike[int])
    assert_type(numba_coop.reduce(numba_group, numba_loaded), int)
    assert_type(
        numba_coop.scan(numba_group, numba_loaded),
        coop.ThreadDataLike[int],
    )
    assert_type(numba_coop.scan(numba_group, common_scalar), int)
    assert_type(numba_coop.exclusive_sum(numba_group, common_scalar), int)
    assert_type(numba_coop.inclusive_sum(numba_group, common_scalar), int)
    assert_type(numba_coop.exclusive_scan(numba_group, common_scalar), int)
    assert_type(numba_coop.inclusive_scan(numba_group, common_scalar), int)
    assert_type(numba_coop.scan(numba_warp, common_scalar), int)
    assert_type(
        numba_coop.exclusive_sum(numba_warp, common_numpy_scalar),
        np.int32,
    )
    assert_type(
        numba_coop.inclusive_sum(numba_warp, common_numpy_scalar),
        np.int32,
    )
    assert_type(numba_coop.exclusive_scan(numba_warp, common_scalar), int)
    assert_type(numba_coop.inclusive_scan(numba_warp, compiler_scalar), Any)
    assert_type(
        numba_coop.exchange(
            numba_group,
            numba_loaded,
            mode="blocked_to_striped",
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.exchange(numba_warp, numba_loaded),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.scan(
            numba_group,
            numba_loaded,
            scan_op=_integer_sum,
            initial_value=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.exclusive_scan(
            numba_group,
            numba_loaded,
            scan_op=_integer_sum,
            initial_value=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.inclusive_scan(
            numba_group,
            numba_loaded,
            scan_op=_integer_sum,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.inclusive_scan(
            numba_warp,
            common_scalar,
            scan_op=_integer_sum,
        ),
        int,
    )
    numba_rank_payload = numba_coop.ThreadData(2, np.int32)
    assert_type(
        numba_coop.radix_rank(numba_group, numba_rank_payload),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.histogram(
            numba_group,
            numba_loaded,
            bins=16,
            counter_dtype=np.int64,
        ),
        coop.ThreadDataLike[np.int64],
    )
    numba_coop.merge_sort_pairs(
        numba_group,
        numba_loaded,
        numba_coop.ThreadData(2, int),
        compare_op=lambda a, b: a < b,
    )
    numba_pairs = numba_coop.topk_min_pairs(
        numba_group,
        numba_loaded,
        numba_coop.ThreadData(2, int),
        1,
    )
    assert_type(
        numba_pairs,
        tuple[coop.ThreadDataLike[int], coop.ThreadDataLike[int]],
    )
    numba_coop.ThreadData(2, int, values=[1, 2])  # pyright: ignore[reportCallIssue]
    numba_coop.exchange(numba_group, common_scalar)  # pyright: ignore[reportArgumentType, reportCallIssue]
    numba_coop.exchange(numba_group, numba_loaded, numba_loaded)  # pyright: ignore[reportCallIssue]
    numba_coop.exchange(numba_group, numba_loaded, launch_metadata={})  # pyright: ignore[reportCallIssue]
    numba_coop.exchange(
        numba_group,
        numba_loaded,
        mode="scatter_to_striped",
        ranks=numba_loaded,
    )

    # One module can contain both backend-neutral kernels. Compiler activation
    # changes lowering; it cannot and should not change this shared static view.
    def cutlass_kernel() -> int:
        payload = coop.ThreadData(2, int)
        return coop.sum(coop.this_block(), payload)

    def numba_kernel() -> int:
        payload = coop.ThreadData(2, int)
        return coop.sum(coop.this_block(), payload)

    assert_type(cutlass_kernel(), int)
    assert_type(numba_kernel(), int)
