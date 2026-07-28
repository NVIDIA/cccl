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
    _keepalive: list = _field(default_factory=list, repr=False)


#: base storage pointer -> LocalizedMeta (guarded by the GIL; allocation
#: and lookup are host-side).
_REGISTRY: dict = {}


def localized_empty(shape, dtype, grid, *, spec=None, mapper=None):
    """Allocate a ``torch.Tensor`` whose pages are placed by *grid*.

    Exactly one of ``spec`` (structured tier; defaults to blocked along
    axis 0 when both are ``None``) or ``mapper`` (callback tier over the
    flat byte domain) selects the placement policy.
    """
    from .. import DeviceArray, cute_partition, data_place  # noqa: PLC0415

    torch = _import_torch()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    if spec is not None and mapper is not None:
        raise ValueError("pass either spec= or mapper=, not both")

    np_dtype = _np_dtype(dtype)
    meta = LocalizedMeta(shape=shape, dtype=dtype, grid=grid)

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

    t = torch.as_tensor(buf)
    _, storage_dtypes = _np_dtype_tables()
    if isinstance(dtype, torch.dtype) and dtype in storage_dtypes:
        t = t.view(dtype)
    t = t.view(shape)

    # CAI carries no ownership: the DeviceArray owns the VMM allocation and
    # the data place's mapper trampoline must outlive the buffer.
    meta._keepalive.extend((buf, dplace))
    base = t.untyped_storage().data_ptr()
    _REGISTRY[base] = meta
    return t


def localized_parameter(shape, dtype, grid, *, spec=None, mapper=None, requires_grad=False):
    """:func:`localized_empty` wrapped as ``torch.nn.Parameter``.

    ``requires_grad`` defaults to ``False`` (the inference /
    ``create_weights`` target).
    """
    torch = _import_torch()
    return torch.nn.Parameter(
        localized_empty(shape, dtype, grid, spec=spec, mapper=mapper),
        requires_grad=requires_grad,
    )


def release(tensor):
    """Explicitly release *tensor*'s localized allocation.

    Localized allocations are REGISTRY-PINNED: the registry owns the
    keepalive (DeviceArray + data place), matching the weights /
    ``create_weights`` lifetime where allocations live for the process.
    ``release`` evicts the entry; the VMM mapping is freed once no tensor
    view references the storage either. (A garbage-collected lifetime tied
    to "the last view" is not expressible with CAI imports — the consumer
    prototypes' ``weakref.finalize(buf, ...)`` was unreachable for exactly
    this reason: the registry itself kept the buffer alive.)
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
    if meta.partition is not None:
        return placement_evaluate(meta.grid, meta.partition, None, np_dtype.itemsize)
    numel = 1
    for s in meta.shape:
        numel *= s
    return placement_evaluate(meta.grid, meta.mapper, (numel * np_dtype.itemsize,), 1)
