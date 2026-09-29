# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block array shifts and qualified scalar selection for CuTe kernels."""

from enum import Enum

from cuda.coop._core.thread_group import ThreadGroup

from ._thread_data import _coerce_thread_payload


def shuffle(group, value, /, *, mode="down", distance=1):
    """Shift register payloads or select another block member's scalar.

    See :func:`cuda.coop.shuffle` for the shared block participation rules,
    dtypes, and unit ``"up"``/``"down"`` shifts. The qualified API adds CuTe
    register payloads and scalar offset or rotate calls.

    Parameters
    ----------
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Fixed-size payloads may be ``ThreadData``, CuTe register tensors, or
        ``TensorSSA`` values. The latter two are converted with
        :meth:`cuda.coop.cutlass.ThreadData.from_payload`. For scalar offset or
        rotate modes, each thread supplies one numeric scalar.
    mode : str, optional
        Compile-time mode, default ``"down"``. Payloads support only ``"up"``
        and ``"down"``; the first or last item of the block tile, respectively,
        is undefined. Scalars require ``"offset"`` or ``"rotate"``: rank ``r``
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
    cuda.coop.shuffle
        Shared array shift contract and executable example.
    :cpp:struct:`cub::BlockShuffle`
        C++ shift, offset, and rotate primitive.

    Examples
    --------
    Rotate one scalar per thread by seven positions in a 64-thread block,
    including wraparound.

    .. literalinclude:: ../../python/cuda_coop/tests/backends/cutlass/runtime/test_qualified_collective_examples.py
        :language: python
        :start-after: # qualified-rotate-example-begin
        :end-before: # qualified-rotate-example-end
        :dedent: 4
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
