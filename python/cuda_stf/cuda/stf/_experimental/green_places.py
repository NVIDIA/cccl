# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Green-context execution places built on cuda.core.

This module creates green contexts (SM partitions) through cuda.core and wraps
them as STF execution places via :meth:`exec_place.from_context`. cuda.core is
the single owner of the SM splitting logic; STF only consumes the resulting
``CUcontext`` handles. The same cuda.core ``Context`` objects can also be
mapped into other frameworks (e.g. ``warp.map_cuda_device``), in which case the
STF place and the other framework's device are backed by the *same* SM
partition.

Requires cuda-core >= 1.0 (green-context support) and CUDA >= 12.4.
"""

from __future__ import annotations

import sys

from ._stf_bindings import exec_place


def _cuda_core_green_api():
    """Import the cuda.core green-context API, raising a helpful error if absent."""
    try:
        from cuda.core import Device
    except ImportError as e:
        raise RuntimeError("green_places() requires cuda-core >= 1.0") from e

    try:
        # Not re-exported publicly as of cuda-core 1.0.1; adjust when they are.
        from cuda.core._context import ContextOptions
        from cuda.core._device_resources import SMResourceOptions
    except ImportError as e:
        raise RuntimeError(
            "green_places() requires cuda-core >= 1.0 with green-context support "
            "(SMResourceOptions/ContextOptions not found)"
        ) from e

    return Device, ContextOptions, SMResourceOptions


def green_places(
    sms_per_place: int,
    n_places: int | None = None,
    device_id: int = 0,
    coscheduled_sm_count: int = 0,
) -> list[exec_place]:
    """Partition a device's SMs into green contexts and return one STF place per partition.

    Args:
        sms_per_place: Number of SMs per place. Rounded up to the device's
            minimum partition size by the driver.
        n_places: Number of places to create, or ``None`` to create as many as
            the device's SM count allows.
        device_id: The device to partition.
        coscheduled_sm_count: Optional co-scheduling constraint forwarded to
            ``SMResourceOptions``.

    Returns:
        A list of :class:`exec_place`, each backed by a cuda.core green
        ``Context``. The contexts are kept alive by the places (see
        :meth:`exec_place.from_context`); places also expose them via the
        read-only ``place.backing_context`` property for interop (e.g.
        ``warp.map_cuda_device``).

    Note:
        The split is performed by carving groups off the device's SM resource
        through ``cuda.core``; if fewer than ``n_places`` groups fit, a
        ``RuntimeError`` is raised. The caller's current CUDA context is saved
        on entry and restored on return, so partitioning a device does not
        leave a different context current for the caller.
    """
    Device, ContextOptions, SMResourceOptions = _cuda_core_green_api()

    from cuda.bindings import driver

    # Save the caller's current context: Device.set_current() and
    # create_context() below both mutate the current context, and we must not
    # leak that side effect back to the caller. A NULL prev_ctx (no current
    # context) is a valid state that we faithfully restore.
    err, prev_ctx = driver.cuCtxGetCurrent()
    if int(err) != 0:
        raise RuntimeError(
            f"green_places(): cuCtxGetCurrent failed with error code {int(err)}"
        )

    try:
        dev = Device(device_id)
        dev.set_current()

        sm = dev.resources.sm
        if sms_per_place <= 0:
            raise ValueError(f"sms_per_place must be positive, got {sms_per_place}")
        if n_places is None:
            n_places = sm.sm_count // max(sms_per_place, sm.min_partition_size)
        if n_places <= 0:
            raise ValueError(f"n_places must be positive, got {n_places}")

        # ``count`` drives the number of groups: a Sequence[int] requests one
        # group per entry, each with the given SM count, in a single split call.
        counts = [sms_per_place] * n_places
        # coscheduled_sm_count requires the CUDA 13.1 structured SM split API;
        # only forward it when the caller actually asked for co-scheduling.
        if coscheduled_sm_count:
            options = SMResourceOptions(
                count=counts, coscheduled_sm_count=[coscheduled_sm_count] * n_places
            )
        else:
            options = SMResourceOptions(count=counts)

        groups, _remainder = sm.split(options)
        if len(groups) < n_places:
            raise RuntimeError(
                f"could not partition device {device_id} into {n_places} places of "
                f"{sms_per_place} SMs (driver returned {len(groups)} groups)"
            )

        places = []
        for group in groups[:n_places]:
            ctx = dev.create_context(ContextOptions(resources=[group]))
            places.append(exec_place.from_context(ctx, dev_id=device_id))

        return places
    finally:
        # Restore the caller's context unconditionally. If restoration fails,
        # surface it -- unless a body exception is already propagating, in
        # which case we must not mask the more informative original error.
        (restore_err,) = driver.cuCtxSetCurrent(prev_ctx)
        if int(restore_err) != 0 and sys.exc_info()[0] is None:
            raise RuntimeError(
                "green_places(): failed to restore the caller's CUDA context "
                f"(cuCtxSetCurrent error code {int(restore_err)})"
            )
