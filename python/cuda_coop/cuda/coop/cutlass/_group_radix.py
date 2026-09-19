# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable block radix ordering and digit ranks for CuTe kernels."""

from cuda.coop._core.thread_group import ThreadGroup

from ._temp_storage import TempStorage
from ._thread_data import ThreadData, _snapshot_readable_payload


def _validate_group(group):
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass radix group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop.cutlass radix requires a complete physical block"
        )


def _input(value, name, primitive):
    return _snapshot_readable_payload(
        value, name=name, primitive=primitive, allow_scalar=True
    )


def _sort(
    group,
    keys,
    values,
    *,
    begin_bit,
    end_bit,
    descending,
    blocked_to_striped,
    temp_storage,
):
    _validate_group(group)
    primitive = "radix_sort_keys" if values is None else "radix_sort_pairs"
    for name, value in (
        ("descending", descending),
        ("blocked_to_striped", blocked_to_striped),
    ):
        if not isinstance(value, bool):
            raise TypeError(
                f"cuda.coop.cutlass.{primitive} {name} must be a compile-time bool"
            )
    if temp_storage is not None and not isinstance(temp_storage, TempStorage):
        raise TypeError("cuda.coop.cutlass radix temp_storage must be TempStorage")
    keys = _input(keys, "keys", primitive)
    if values is not None:
        values = _input(values, "values", primitive)
        if isinstance(keys, ThreadData) != isinstance(values, ThreadData):
            raise TypeError(
                "radix_sort_pairs keys and values must have matching scalar or array shapes"
            )
        if (
            isinstance(keys, ThreadData)
            and keys.items_per_thread != values.items_per_thread
        ):
            raise ValueError(
                "radix_sort_pairs keys and values must have matching items_per_thread"
            )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._radix import provider_radix_sort

    return provider_radix_sort(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        values=values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_sort_keys(
    group,
    keys,
    /,
    *,
    begin_bit=0,
    end_bit=None,
    descending=False,
    temp_storage=None,
    blocked_to_striped=False,
):
    """Return fresh, stable radix-sorted keys without changing the input.

    All block threads participate. Bounds select bits in CUB's ordered key
    representation and may be block-uniform runtime integers. Omitted end_bit
    selects the full key width. Qualified calls accept scalars, register payloads,
    floating-point keys, and blocked_to_striped output in addition to the common
    integral ThreadData inputs. Equal selected digits retain blocked input order.
    """
    return _sort(
        group,
        keys,
        None,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_sort_pairs(
    group,
    keys,
    values,
    /,
    *,
    begin_bit=0,
    end_bit=None,
    descending=False,
    temp_storage=None,
    blocked_to_striped=False,
):
    """Return fresh sorted keys and associated values, preserving both inputs.

    Keys and values have matching scalar or fixed-array shapes and independent
    dtypes. Ordering, bit bounds, and output layout follow radix_sort_keys.
    """
    if values is None:
        raise TypeError("radix_sort_pairs values must be a numeric scalar or payload")
    return _sort(
        group,
        keys,
        values,
        begin_bit=begin_bit,
        end_bit=end_bit,
        descending=descending,
        blocked_to_striped=blocked_to_striped,
        temp_storage=temp_storage,
    )


def radix_rank(
    group,
    keys,
    /,
    *,
    begin_bit=0,
    end_bit=None,
    radix_bits=None,
    descending=False,
    exclusive_digit_prefix=None,
):
    """Return stable Int32 ranks for a compile-time interval of 1–8 key bits.

    Signed keys invert their sign bit before digit extraction. Rank preserves
    keys and returns a scalar or fresh payload matching the input shape.
    An optional writable Int32 ThreadData receives exclusive bin prefixes;
    thread t owns ascending bin indices t*P+i in either ordering direction.
    P is ceil(number_of_bins/block_threads), with a minimum of one. Slots past
    the final bin are undefined. Prefix output must be distinct from keys.
    """
    _validate_group(group)
    if not isinstance(descending, bool):
        raise TypeError(
            "cuda.coop.cutlass.radix_rank descending must be a compile-time bool"
        )
    if exclusive_digit_prefix is keys:
        raise ValueError("radix_rank exclusive_digit_prefix must be distinct from keys")
    keys = _input(keys, "keys", "radix_rank")
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._radix import provider_radix_rank

    return provider_radix_rank(
        group=group,
        launch=current_kernel_launch_facts(),
        keys=keys,
        begin_bit=begin_bit,
        end_bit=end_bit,
        radix_bits=radix_bits,
        descending=descending,
        exclusive_digit_prefix=exclusive_digit_prefix,
    )


__all__ = ["radix_sort_keys", "radix_sort_pairs", "radix_rank"]
