# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Independent batches reduced across a complete physical or logical warp."""

from enum import Enum

from cuda.coop._core.thread_group import ThreadGroup

from ._compiler._launch import current_kernel_launch_facts
from ._thread_data import _snapshot_readable_payload
from ._thread_group import _require_complete_warp_partition


def reduce_batched(group, value, /, *, binary_op=None, output_layout="striped"):
    """Reduce each payload slot independently across a warp.

    The groups, output ownership, and participation contract are those of
    :func:`cuda.coop.reduce_batched`. The input is preserved. In addition to
    readable ``ThreadData`` payloads, this qualified form accepts CuTe register
    tensors and ``TensorSSA`` values through ``ThreadData.from_payload``.

    ``binary_op`` accepts the built-in strings and aliases documented by
    :func:`cuda.coop.cutlass.reduce`. Custom callbacks are not supported.
    ``output_layout`` is ``"striped"`` or ``"blocked"``. For ``B`` batches and
    ``W`` lanes, the returned ``ThreadData`` contains ``ceil(B / W)`` slots per
    lane. Only slots corresponding to a batch are defined. Guard reads using
    the batch index, as shown in the common API example.

    The CUB provider exchanges register values within the participating warp.
    It requires no shared scratch allocation or storage-reuse barrier. Logical
    warps may call it independently, including from different branches.

    Parameters
    ----------
    group : ThreadGroup
        Complete physical warp or logical warp of 1, 2, 4, 8, 16, or 32 lanes.
        All members of the selected group participate.
    value : ThreadData or CuTe register payload
        One input per independent batch in each lane. All lanes use the same
        positive batch count and numeric dtype. The input remains unchanged.
    binary_op : str or recognized built-in alias, optional
        Reduction operator, default addition. Bitwise operators require
        integer inputs. Custom callbacks are not supported.
    output_layout : {"striped", "blocked"}, optional
        Result ownership, default striped. Striped slot ``i`` in lane ``t``
        holds batch ``t + i * W``; blocked slot ``i`` holds batch
        ``t * ceil(B / W) + i``. Read only indices below ``B``.

    Returns
    -------
    ThreadData
        Fresh payload with the input dtype and ``ceil(B / W)`` slots per lane.
        Slots beyond the batch count are undefined.

    Notes
    -----
    See the :doc:`Batched Warp Reduction visualization
    <coop/visualizations/reduce-batched>` for lane and result ownership.
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass.reduce_batched group must be a ThreadGroup")
    if not isinstance(output_layout, str) or isinstance(output_layout, Enum):
        raise TypeError("reduce_batched output_layout must be a string")
    output_layout = output_layout.strip().lower().replace("-", "_")
    if output_layout not in {"striped", "blocked"}:
        raise ValueError("reduce_batched output_layout must be striped or blocked")
    from ._operators import normalize_operator

    op = normalize_operator(binary_op, primitive="reduce_batched")
    value = _snapshot_readable_payload(value, name="value", primitive="reduce_batched")
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="reduce_batched", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._reduce_batched import provider_reduce_batched

    return provider_reduce_batched(
        group=group, launch=launch, value=value, op=op, output_layout=output_layout
    )


__all__ = ["reduce_batched"]
