# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate block Shuffle calls and adapt register payloads.

Payload up/down shifts and scalar offset/rotate calls have different result
ABIs. Planning checks the selected form and distance before the
provider emits a call; register inputs are converted to ThreadData.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, TypeVar

from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import IntegerValue

from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    ThreadData,
    _coerce_thread_payload,
)
from ._typing import (
    BlockGroup,
    CommonNumericScalar,
    CommonShuffleMode,
    CommonThreadDataLike,
    ScalarShuffleMode,
)

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


def shuffle(
    group: BlockGroup,
    value: CommonThreadDataLike[_ItemT]
    | CutlassTensorSample
    | CutlassTensorSSASample
    | _ItemT,
    /,
    *,
    mode: CommonShuffleMode | ScalarShuffleMode = "down",
    distance: Literal[1] | IntegerValue = 1,
) -> ThreadData | _ItemT:
    """Shift register payloads or select another block member's scalar.

    Payload shifts follow flattened blocked order: each thread's items are
    consecutive, and a shift crosses thread boundaries as needed. Scalar
    offset and rotate calls select one other member's value.

    Parameters
    ----------
    group : ThreadGroup
        Complete block whose members all call the primitive. Warp, mapped
        groups, clusters, and grids are unsupported.
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Fixed-size payloads may be ``ThreadData``, CuTe register tensors, or
        ``TensorSSA`` values. The latter two are converted with
        :meth:`cuda.coop.cutlass.ThreadData.from_payload`. For scalar offset or
        rotate modes, each thread supplies one numeric scalar. All members
        use the same dtype and extent. Supported dtypes are signed and
        unsigned 8-, 16-, 32-, and 64-bit integers, ``Float32``, and
        ``Float64``.
    mode : str, optional
        Compile-time mode, default ``"down"``. Payloads support only ``"up"``
        and ``"down"``; the first or last item of the block tile, respectively,
        is undefined. Up moves input ``i - 1`` to output ``i``; down moves
        input ``i + 1`` to output ``i``. Scalars require ``"offset"`` or
        ``"rotate"``: rank ``r``
        receives the value from rank ``r + distance``. Offset results are
        undefined when that source is outside the block; rotate wraps the
        source rank modulo the block size.
    distance : int or CuTe integer scalar, optional
        Default ``1``. Payload shifts require a compile-time unit distance.
        Scalar modes accept compile-time or runtime distances, which may
        differ between threads. Offset distances must fit a signed 32-bit
        integer and may be zero or negative. Rotate requires at least two
        threads and ``1 <= distance < block_size``. A runtime distance may
        have a signed integer dtype up to 64 bits or an unsigned integer dtype
        up to 32 bits, but its value must remain in the mode's allowed range.

    Returns
    -------
    CuTe numeric scalar or cuda.coop.cutlass.ThreadData
        Scalar input produces a CuTe scalar with the input dtype. Payload
        input produces a new writable ``ThreadData`` with the input dtype,
        extent, and requested alignment. The input remains unchanged.
        Observe the chosen mode's boundary rules before reading the result.

    Notes
    -----
    Scratch allocation and trailing synchronization are automatic. Every
    block member must call, including threads with out-of-range offset
    sources. Warp and logical-warp groups are not supported.

    See Also
    --------
    :cpp:struct:`cub::BlockShuffle`
        C++ shift, offset, and rotate primitive.

    Examples
    --------
    Rotate one scalar per thread by seven positions in a 64-thread block,
    including wraparound.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_collective_examples.py
        :language: python
        :start-after: # qualified-rotate-example-begin
        :end-before: # qualified-rotate-example-end
        :dedent: 4
    """
    if not isinstance(group, ThreadGroup):
        raise TypeError("cuda.coop.cutlass.shuffle group must be a ThreadGroup")
    if group.kind != "block":
        raise NotImplementedError(
            "cuda.coop.cutlass.shuffle requires a block group"
        )
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
