# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""PyTorch interop helpers for ``cuda.stf._experimental``.

This module provides:

* :func:`tensor_arg` and :func:`tensor_arguments` -- convert one or all STF
  task arguments to ``torch.Tensor`` views via the CUDA Array Interface.
* :func:`pytorch_task` -- context manager that opens an STF task, makes the
  task stream the current PyTorch CUDA stream, and yields the task arguments
  as ``torch.Tensor`` views.

PyTorch is imported lazily inside each function. Importing this module does
not require PyTorch to be installed; calling a function that uses PyTorch
without it raises :class:`ImportError` with an installation hint.
"""

from __future__ import annotations

_TORCH_INSTALL_HINT = (
    "This functionality requires PyTorch to be installed. "
    "Install PyTorch or use ``ctx.task()`` directly for a raw task."
)


def _import_torch():
    """Import :mod:`torch`, raising a friendly error if unavailable."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_TORCH_INSTALL_HINT) from exc
    return torch


def tensor_arg(task, index):
    """Return one task argument as a ``torch.Tensor``.

    ``task.get_arg_cai(index)`` returns an stf_cai exposing the
    ``__cuda_array_interface__`` protocol.
    """
    torch = _import_torch()
    return torch.as_tensor(task.get_arg_cai(index))


def tensor_arguments(task):
    """Return all task buffer arguments as ``torch.Tensor`` views.

    Same shape as ``task.args_cai()``: ``None``, a single tensor, or a tuple
    of tensors.
    """
    torch = _import_torch()
    out = task.args_cai()
    if out is None:
        return None
    if isinstance(out, tuple):
        return tuple(torch.as_tensor(o) for o in out)
    return torch.as_tensor(out)


def pytorch_task(ctx, *args):
    """Context manager: ``ctx.task(*args)`` with PyTorch stream + tensor conversion.

    Yields the tensor(s) from ``task.args_cai()`` converted to ``torch.Tensor``
    as a tuple. The STF task stream is also made the current PyTorch CUDA
    stream for the duration of the ``with`` block.

    Example
    -------
    >>> from cuda.stf._experimental.interop.pytorch import pytorch_task
    >>> with pytorch_task(ctx, lX.read(), lY.rw()) as (x_tensor, y_tensor):
    ...     y_tensor[:] = x_tensor * 2
    """
    torch = _import_torch()
    tc = torch.cuda

    t = ctx.task(*args)

    class _PyTorchTaskContext:
        _stream_ctx = None

        def __enter__(self):
            t.start()
            try:
                stream_ctx = tc.stream(tc.ExternalStream(t.stream_ptr()))
                stream_ctx.__enter__()
                self._stream_ctx = stream_ctx
                tensors = tensor_arguments(t)
            except Exception as e:
                if self._stream_ctx is not None:
                    try:
                        self._stream_ctx.__exit__(type(e), e, e.__traceback__)
                    except Exception:
                        pass
                try:
                    t.end()
                except Exception:
                    pass
                raise
            if tensors is None:
                return None
            if isinstance(tensors, tuple):
                return tensors
            return (tensors,)

        def __exit__(self, exc_type, exc_val, exc_tb):
            # Always run both cleanups (stream exit and task end), then decide
            # what to raise. Exception precedence: a failure in the body wins,
            # then a stream-cleanup failure, then a task-cleanup failure. This
            # guarantees the task is always ended even if the stream context
            # exit raises, and never lets cleanup mask the user's own error.
            stream_exc = None
            task_exc = None
            if self._stream_ctx is not None:
                try:
                    self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
                except BaseException as e:  # noqa: BLE001
                    stream_exc = e
            try:
                t.end()
            except BaseException as e:  # noqa: BLE001
                task_exc = e

            if exc_type is not None:
                # Preserve the in-flight body exception; do not mask it.
                return False
            if stream_exc is not None:
                raise stream_exc
            if task_exc is not None:
                raise task_exc
            return False

    return _PyTorchTaskContext()


__all__ = ["pytorch_task", "tensor_arg", "tensor_arguments"]


# ---------------------------------------------------------------------------
# Localized allocation: torch tensors backed by composite VMM data places.
#
# Lifted from the two consumer prototypes that predicted this surface
# (vllm localization/phase0/localized_torch.py and pytorch
# torch/cuda/_localized/_alloc.py — same lineage): one ordinary contiguous
# torch.Tensor whose PHYSICAL pages are striped over the grid's places by a
# partition. Checkpoint loaders and kernels see a plain tensor.
#
# Two tiers:
#   * structured (preferred): a cute-partition SPEC — hashable, carries its
#     own extents, drives placement_evaluate and downstream splitting.
#   * callback (escape hatch): a Python mapper over the flat byte domain —
#     opaque to caching and compilation; for assignments the spec grammar
#     cannot express (e.g. permuted expert->place tables).
#
# Two lifetimes (the interchange protocol picks it):
#   * "pinned" — CAI import (torch.as_tensor): CAI describes memory but
#     transfers no ownership, so the registry pins the allocation for the
#     process and release() evicts explicitly.
#   * "gc" — DLPack import (torch.from_dlpack): the tensor's STORAGE owns
#     the allocation (module -> parameter -> storage -> deleter), freed when
#     the last view dies; the registry holds metadata only, evicted by a
#     finalizer that is reachable precisely because nothing pins the buffer.
#     This is the idiomatic lifetime for nn.Parameter weights: unload paths
#     (model swap, sleep mode) free the VMM with the module.
#
# Metadata is keyed by BASE STORAGE POINTER in a module registry so it
# survives views, reshapes, and nn.Parameter wrapping (tensor attributes do
# not) — the enabler for compiler-side detection in consumers.
# ---------------------------------------------------------------------------

import weakref as _weakref
from dataclasses import dataclass as _dataclass, field as _field
from typing import Any as _Any


def _np_dtype_tables():
    import numpy as np  # noqa: PLC0415
    torch = _import_torch()

    direct = {
        torch.float64: np.float64,
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int64: np.int64,
        torch.int32: np.int32,
        torch.int16: np.int16,
        torch.int8: np.int8,
        torch.uint8: np.uint8,
        torch.bool: np.bool_,
    }
    # numpy has no native bfloat16/fp8: allocate same-size storage and
    # view() the torch tensor back to the requested dtype.
    storage = {
        torch.bfloat16: np.uint16,
        getattr(torch, "float8_e4m3fn", None): np.uint8,
        getattr(torch, "float8_e5m2", None): np.uint8,
    }
    storage.pop(None, None)
    return direct, storage


def _np_dtype(dtype):
    import numpy as np  # noqa: PLC0415

    direct, storage = _np_dtype_tables()
    if dtype in direct:
        return np.dtype(direct[dtype])
    if dtype in storage:
        return np.dtype(storage[dtype])
    try:
        return np.dtype(dtype)
    except TypeError:
        supported = sorted(str(k) for k in (*direct, *storage))
        raise TypeError(
            f"unsupported dtype {dtype!r} for localized allocation; "
            f"supported torch dtypes: {supported}"
        ) from None


@_dataclass
class LocalizedMeta:
    """Placement metadata for one localized allocation."""

    shape: tuple
    dtype: _Any
    grid: _Any
    partition: _Any = None  # structured tier: the cute_partition
    mapper: _Any = None  # callback tier: the Python mapper
    lifetime: str = "pinned"  # "pinned" (CAI + registry) | "gc" (DLPack)
    _keepalive: list = _field(default_factory=list, repr=False)


@_dataclass
class ReplicatedMeta:
    """Placement metadata for one replicated allocation.

    One canonical physical copy backs the tensor; the grid names the places
    that hold their own copy when the tensor is READ at
    ``data_place.replicated(grid)`` (the runtime broadcasts on first use and
    reuses the replicas afterwards). The write-once-read-replicated shape of
    model weights.
    """

    shape: tuple
    dtype: _Any
    grid: _Any
    lifetime: str = "pinned"  # "pinned" (CAI + registry) | "gc" (DLPack)
    _keepalive: list = _field(default_factory=list, repr=False)


#: base storage pointer -> LocalizedMeta | ReplicatedMeta (guarded by the
#: GIL; allocation and lookup are host-side).
_REGISTRY: dict = {}


def _evict(base, meta):
    """Registry eviction for gc-lifetime allocations (finalizer target).

    Guarded on identity so a recycled base address can never evict a
    successor's entry.
    """
    if _REGISTRY.get(base) is meta:
        del _REGISTRY[base]


def localized_empty(shape, dtype, grid, *, spec=None, mapper=None, lifetime="pinned"):
    """Allocate a ``torch.Tensor`` whose pages are placed by *grid*.

    Exactly one of ``spec`` (structured tier; defaults to blocked along
    axis 0 when both are ``None``) or ``mapper`` (callback tier over the
    flat byte domain) selects the placement policy.

    ``lifetime`` selects the interchange protocol and with it who owns the
    allocation:

    * ``"pinned"`` (default): CAI import. CAI transfers no ownership, so
      the registry pins the allocation for the process; free explicitly
      with :func:`release`.
    * ``"gc"``: DLPack import. The tensor's storage OWNS the allocation
      (freed when the last view dies — the idiomatic lifetime for
      ``nn.Parameter`` weights, where unloading the module frees the VMM);
      the registry holds metadata only and self-evicts.
    """
    from .. import DeviceArray, cute_partition, data_place  # noqa: PLC0415

    torch = _import_torch()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    if spec is not None and mapper is not None:
        raise ValueError("pass either spec= or mapper=, not both")
    if lifetime not in ("pinned", "gc"):
        raise ValueError(f'lifetime must be "pinned" or "gc", got {lifetime!r}')

    np_dtype = _np_dtype(dtype)
    meta = LocalizedMeta(shape=shape, dtype=dtype, grid=grid, lifetime=lifetime)

    if mapper is not None:
        numel = 1
        for s in shape:
            numel *= s
        nbytes = numel * np_dtype.itemsize
        dplace = data_place.composite(grid, mapper, data_rank=1)
        buf = DeviceArray(numel, np_dtype, dplace, dims=(nbytes,), elemsize=1)
        meta.mapper = mapper
    else:
        if spec is None:
            spec = (("blocked", 0),) + (None,) * (len(shape) - 1)
        gd = grid.dims() if callable(getattr(grid, "dims", None)) else grid.dims
        part = cute_partition.from_spec(shape, spec, tuple(int(e) for e in gd))
        dplace = data_place.composite_cute(grid, part)
        buf = DeviceArray(shape, np_dtype, dplace)
        meta.partition = part

    if lifetime == "gc":
        # DLPack: the tensor's storage takes ownership (the capsule holds
        # the DeviceArray, which holds the data place). Nothing pins the
        # buffer, so a finalizer on it is REACHABLE and evicts the
        # metadata when the storage dies.
        t = torch.from_dlpack(buf)
    else:
        t = torch.as_tensor(buf)
    _, storage_dtypes = _np_dtype_tables()
    if isinstance(dtype, torch.dtype) and dtype in storage_dtypes:
        t = t.view(dtype)
    t = t.view(shape)

    base = t.untyped_storage().data_ptr()
    if lifetime == "gc":
        _weakref.finalize(buf, _evict, base, meta)
    else:
        # CAI carries no ownership: the DeviceArray owns the VMM allocation
        # and the data place's mapper trampoline must outlive the buffer.
        meta._keepalive.extend((buf, dplace))
    _REGISTRY[base] = meta
    return t


def localized_parameter(
    shape, dtype, grid, *, spec=None, mapper=None, requires_grad=False, lifetime="gc"
):
    """:func:`localized_empty` wrapped as ``torch.nn.Parameter``.

    ``requires_grad`` defaults to ``False`` (the inference /
    ``create_weights`` target). ``lifetime`` defaults to ``"gc"`` here —
    a parameter registered on a module IS the idiomatic owner (module ->
    parameter -> storage -> allocation), so unloading the module frees the
    VMM; pass ``lifetime="pinned"`` for the registry-pinned behavior.
    """
    torch = _import_torch()
    return torch.nn.Parameter(
        localized_empty(shape, dtype, grid, spec=spec, mapper=mapper, lifetime=lifetime),
        requires_grad=requires_grad,
    )


def replicated_empty(shape, dtype, grid, *, device=0, canonical=None, lifetime="pinned"):
    """Allocate a ``torch.Tensor`` intended to be replicated over *grid*.

    The sibling of :func:`localized_empty` for the other half of the
    placement vocabulary: instead of striping one allocation over the grid,
    the tensor is a single canonical copy (plain device memory on
    ``device``) that tasks READ at ``data_place.replicated(grid)`` — the
    runtime materializes one replica per grid member on first use. Write the
    tensor at its canonical place (weight loading), read it replicated:
    replicated places are read-only by contract.

    Use :func:`replicated_dplace` to obtain the read-side data place for
    task dependencies. ``lifetime`` follows :func:`localized_empty`:
    ``"pinned"`` (CAI import, freed via :func:`release`) or ``"gc"``
    (DLPack import, storage owns the allocation).

    ``canonical`` optionally names the data place of the canonical copy
    (overriding ``device``). Pass the first member's place (e.g. a
    locality-domain place on a domain grid) so the canonical copy IS
    replica 0: the runtime then materializes only N-1 broadcast copies
    instead of leaving an extra never-read build copy behind (N+1 total).
    """
    from .. import DeviceArray, data_place  # noqa: PLC0415

    torch = _import_torch()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    if lifetime not in ("pinned", "gc"):
        raise ValueError(f'lifetime must be "pinned" or "gc", got {lifetime!r}')

    np_dtype = _np_dtype(dtype)
    meta = ReplicatedMeta(shape=shape, dtype=dtype, grid=grid, lifetime=lifetime)

    dplace = canonical if canonical is not None else data_place.device(int(device))
    buf = DeviceArray(shape, np_dtype, dplace)

    if lifetime == "gc":
        t = torch.from_dlpack(buf)
    else:
        t = torch.as_tensor(buf)
    _, storage_dtypes = _np_dtype_tables()
    if isinstance(dtype, torch.dtype) and dtype in storage_dtypes:
        t = t.view(dtype)
    t = t.view(shape)

    base = t.untyped_storage().data_ptr()
    if lifetime == "gc":
        _weakref.finalize(buf, _evict, base, meta)
    else:
        meta._keepalive.extend((buf, dplace))
    _REGISTRY[base] = meta
    return t


def replicated_dplace(tensor):
    """Read-side data place for a :func:`replicated_empty` tensor.

    Returns ``data_place.replicated(meta.grid)`` for use in read
    dependencies (write access at a replicated place raises at dependency
    construction).
    """
    from .. import data_place  # noqa: PLC0415

    meta = get_meta(tensor)
    if meta is None:
        raise ValueError("tensor is not a registered STF allocation")
    if not isinstance(meta, ReplicatedMeta):
        raise TypeError("tensor is a localized allocation, not a replicated one")
    return data_place.replicated(meta.grid)


def release(tensor):
    """Explicitly release *tensor*'s localized allocation.

    For ``lifetime="pinned"`` allocations the registry owns the keepalive
    (DeviceArray + data place): ``release`` evicts the entry and the VMM
    mapping is freed once no tensor view references the storage either.
    (A garbage-collected lifetime tied to "the last view" is not
    expressible with CAI imports — the consumer prototypes'
    ``weakref.finalize(buf, ...)`` was unreachable for exactly this
    reason: the registry itself kept the buffer alive. That is what
    ``lifetime="gc"`` exists for.)

    For ``lifetime="gc"`` allocations the storage already owns the
    buffer; ``release`` merely drops the metadata early (harmless — it
    would self-evict when the storage dies).
    """
    base = tensor.untyped_storage().data_ptr()
    meta = _REGISTRY.pop(base, None)
    if meta is None:
        raise ValueError("tensor is not a live localized allocation")
    meta._keepalive.clear()


def get_meta(tensor):
    """Metadata for *tensor* if its storage is a localized allocation.

    Survives views, reshapes, and ``nn.Parameter`` wrapping (keyed by base
    storage pointer). Returns ``None`` for ordinary tensors.
    """
    try:
        base = tensor.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return None
    return _REGISTRY.get(base)


def live_metas():
    """Snapshot of metadata for all live localized allocations."""
    return list(_REGISTRY.values())


def placement_report(tensor, probes: int = 4096):
    """Dry-run the block-owner decision for *tensor*'s allocation.

    Returns the ``placement_evaluate`` stats (bytes per grid index and a
    sampling-fidelity ``accuracy``; ~1.0 means page granularity matches
    the partition exactly).
    """
    from .. import placement_evaluate  # noqa: PLC0415

    meta = get_meta(tensor)
    if meta is None:
        raise ValueError("tensor is not a localized allocation")
    np_dtype = _np_dtype(meta.dtype)
    if isinstance(meta, ReplicatedMeta):
        raise ValueError(
            "tensor is a replicated allocation: one full copy per grid "
            "member by construction (no block-owner decision to report)"
        )
    if meta.partition is not None:
        return placement_evaluate(meta.grid, meta.partition, None, np_dtype.itemsize)
    numel = 1
    for s in meta.shape:
        numel *= s
    return placement_evaluate(meta.grid, meta.mapper, (numel * np_dtype.itemsize,), 1)
