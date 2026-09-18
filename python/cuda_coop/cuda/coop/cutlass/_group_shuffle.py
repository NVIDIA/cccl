# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block array shifts and qualified scalar selection for CuTe kernels."""

from enum import Enum

from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import _coerce_thread_payload


def shuffle(group, value, /, *, mode="down", distance=1):
    """Return a shifted payload or another block member's scalar.

    Fixed payloads support unit ``up`` and ``down`` shifts in blocked order;
    the first or last result item, respectively, is undefined. The input is
    preserved and the result is a fresh payload with the same dtype and extent.

    Qualified scalar ``offset`` and ``rotate`` select rank ``r + distance``.
    Offset accepts signed 32-bit distances, including zero and negative values;
    out-of-block results are undefined. Rotate wraps within the block and
    requires ``1 <= distance < block_size``. Runtime distances may differ
    between members. Every block member must call the primitive. Scratch and
    its trailing reuse barrier are managed automatically.
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass.shuffle group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError("cuda.coop.cutlass.shuffle requires a block group")
    if not isinstance(mode, str) or isinstance(mode, Enum):
        raise TypeError("cuda.coop.cutlass.shuffle mode must be a string")
    mode = mode.strip().lower().replace("-", "_")
    if mode not in {"up", "down", "offset", "rotate"}:
        raise ValueError(
            "cuda.coop.cutlass.shuffle mode must be up, down, offset, or rotate"
        )
    value = _coerce_thread_payload(
        value,
        scope="cuda.coop.cutlass",
        primitive_name="shuffle",
        arg_name="value",
        common_root_payload_kind="thread_data",
    )
    from ._compiler._launch import current_kernel_launch_facts
    from ._lowering._shuffle import provider_shuffle

    return provider_shuffle(
        group=group,
        launch=current_kernel_launch_facts(),
        value=value,
        mode=mode,
        distance=distance,
    )


__all__ = ["shuffle"]
