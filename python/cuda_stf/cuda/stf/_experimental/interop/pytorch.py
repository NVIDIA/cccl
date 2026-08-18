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

import sys as _sys  # noqa: E402
import weakref as _weakref  # noqa: E402
from dataclasses import dataclass as _dataclass  # noqa: E402
from dataclasses import field as _field  # noqa: E402
from typing import Any as _Any  # noqa: E402


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


#: base storage pointer -> LocalizedMeta (guarded by the
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
        # The buffer below is allocated flat (dims=(nbytes,), elemsize=1), so
        # the mapper's contract on this path is byte space: data_rank=1 with
        # data_dims == (nbytes,) and byte-offset coordinates, matching the
        # documented task-path behavior of data_place.composite.
        dplace = data_place.composite(grid, mapper, data_rank=1)
        buf = DeviceArray(numel, np_dtype, dplace, dims=(nbytes,), elemsize=1)
        meta.mapper = mapper
    else:
        if spec is None:
            spec = (("blocked", 0),) + (None,) * (len(shape) - 1)
        if isinstance(spec, cute_partition):
            # Prebuilt partition (e.g. reused from another allocation's meta
            # by the *_like factories): its extents are the contract.
            if tuple(spec.true_dims) != shape:
                raise ValueError(
                    f"partition true_dims {tuple(spec.true_dims)} do not match shape {shape}"
                )
            part = spec
        else:
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
        localized_empty(
            shape, dtype, grid, spec=spec, mapper=mapper, lifetime=lifetime
        ),
        requires_grad=requires_grad,
    )


def localized_zeros(shape, dtype, grid, *, spec=None, mapper=None, lifetime="pinned"):
    """:func:`localized_empty` filled with zeros.

    The fill is an ordinary in-place torch write through the tensor's single
    base pointer. Unlike NUMA first-touch, VMM placement is fixed at
    allocation by the partition -- the fill has no effect on locality (this
    holds for any in-place initializer: ``normal_()``, ``nn.init.*``, ...).
    """
    t = localized_empty(shape, dtype, grid, spec=spec, mapper=mapper, lifetime=lifetime)
    t.zero_()
    return t


def localized_ones(shape, dtype, grid, *, spec=None, mapper=None, lifetime="pinned"):
    """:func:`localized_empty` filled with ones (see :func:`localized_zeros`)."""
    t = localized_empty(shape, dtype, grid, spec=spec, mapper=mapper, lifetime=lifetime)
    t.fill_(1)
    return t


def localized_full(
    shape, fill_value, dtype, grid, *, spec=None, mapper=None, lifetime="pinned"
):
    """:func:`localized_empty` filled with ``fill_value`` (see :func:`localized_zeros`)."""
    t = localized_empty(shape, dtype, grid, spec=spec, mapper=mapper, lifetime=lifetime)
    t.fill_(fill_value)
    return t


def _localized_like(tensor, dtype, lifetime):
    meta = get_meta(tensor)
    if meta is None or not isinstance(meta, LocalizedMeta):
        raise ValueError("tensor is not a localized allocation")
    return localized_empty(
        meta.shape,
        meta.dtype if dtype is None else dtype,
        meta.grid,
        spec=meta.partition if meta.partition is not None else None,
        mapper=meta.mapper,
        lifetime=meta.lifetime if lifetime is None else lifetime,
    )


def localized_empty_like(tensor, *, dtype=None, lifetime=None):
    """New localized allocation with *tensor*'s placement (same grid and
    partition object -- or mapper -- same shape; ``dtype``/``lifetime``
    overridable)."""
    return _localized_like(tensor, dtype, lifetime)


def localized_zeros_like(tensor, *, dtype=None, lifetime=None):
    """:func:`localized_empty_like` filled with zeros."""
    t = _localized_like(tensor, dtype, lifetime)
    t.zero_()
    return t


def localized_ones_like(tensor, *, dtype=None, lifetime=None):
    """:func:`localized_empty_like` filled with ones."""
    t = _localized_like(tensor, dtype, lifetime)
    t.fill_(1)
    return t


def localized_full_like(tensor, fill_value, *, dtype=None, lifetime=None):
    """:func:`localized_empty_like` filled with ``fill_value``."""
    t = _localized_like(tensor, dtype, lifetime)
    t.fill_(fill_value)
    return t


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


def spec_of(tensor):
    """Placement spec of a registered allocation: the ``cute_partition``
    (structured tier) or the mapper (callback tier). Looked up by base
    storage pointer, so views, reshapes and ``nn.Parameter`` wrapping all
    resolve to their root allocation (tensor attributes would not survive
    those)."""
    meta = get_meta(tensor)
    if meta is None:
        raise ValueError("tensor is not a registered STF allocation")
    return meta.partition if meta.partition is not None else meta.mapper


def grid_of(tensor):
    """Execution grid of a registered allocation (see :func:`spec_of`)."""
    meta = get_meta(tensor)
    if meta is None:
        raise ValueError("tensor is not a registered STF allocation")
    return meta.grid


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


# ---------------------------------------------------------------------------
# map: per-die execution of a map expression over localized operands.
# ---------------------------------------------------------------------------


def _partitions_equal(a, b):
    if a is b:
        return True
    return (
        tuple(a.true_dims) == tuple(b.true_dims)
        and tuple(a.grid_dims) == tuple(b.grid_dims)
        and a.place_leaves == b.place_leaves
        and a.local_leaves == b.local_leaves
    )


#: per-grid-size stream pools for the fork/join (created once, reused)
_MAP_STREAMS: dict = {}


def _map_streams(nplaces):
    torch = _import_torch()
    pool = _MAP_STREAMS.get(nplaces)
    if pool is None:
        pool = [torch.cuda.Stream() for _ in range(nplaces)]
        _MAP_STREAMS[nplaces] = pool
    return pool


def _die_view(torch, tensor, part, die):
    """Strided view selecting exactly die's owned elements (padded space).

    The local leaves are the die-local iteration shape; the grid place
    offset rebases it. Die identity lives entirely in the storage offset,
    so all dies' views are shape/stride identical -- one torch.compile
    artifact (guards on shape/stride) serves every die.
    """
    leaves = part.local_leaves
    sizes = tuple(int(e) for e, _ in leaves)
    strides = tuple(int(st) for _, st in leaves)
    offset = int(part.grid_place_offset(die)) + tensor.storage_offset()
    return torch.as_strided(tensor, sizes, strides, offset)


def views(tensor, spec=None):
    """The per-die strided views of a localized tensor (one per grid
    position, exactly the die's owned elements, padded space).

    The escape hatch for constructs beyond :func:`map` -- e.g. reductions
    over a SPLIT dim, done as per-die partials over these views followed by
    a fold of the P partials (the write-dual pattern).
    """
    torch = _import_torch()
    part = spec if spec is not None else spec_of(tensor)
    if part is None or callable(part):
        raise ValueError("views requires the structured (spec) tier")
    gd = 1
    for e in tuple(part.grid_dims):
        gd *= int(e)
    return [_die_view(torch, tensor, part, d) for d in range(gd)]


def map(fn, *tensors, spec=None, streams=None):  # noqa: A001 - namespace attribute
    """Apply a MAP expression per die, each die over its owned elements.

    ``fn`` is any callable -- eager, or a (stock) ``torch.compile`` artifact
    -- whose dataflow respects the split axes: pointwise always; dim-wise
    ops along UNSPLIT dims (softmax/LayerNorm over an unsplit hidden dim
    with batch-blocked operands) are valid; reductions or stencils touching
    a split dim are not (those need per-die partials + a fold). ``fn`` must
    write IN-PLACE (or into localized operands passed to it): out-of-place
    results would come from the ordinary torch allocator, unlocalized.

    The iteration spec is inferred from the operands: all localized
    operands must share one partition (validated eagerly from the
    registry); ordinary broadcast scalars pass through whole. ``spec=``
    overrides only when no localized operand carries one.

    Execution forks one launch per die on a cached per-die stream (the
    event-based fork/join idiom, which stream capture follows), each over
    a strided view of exactly the die's elements -- restriction by
    re-indexing, not predication. Confinement to SM partitions can be
    layered by passing explicit ``streams=`` (e.g. green-context streams).

    Views cover the PADDED space: split dims are padded to divisibility,
    so ``fn`` may compute on padding elements; they are never observed
    through the tensor's true extents.
    """
    torch = _import_torch()

    part = spec
    view_args = []  # per operand: partition or None (pass-through)
    for t in tensors:
        meta = get_meta(t) if isinstance(t, torch.Tensor) else None
        if meta is None:
            view_args.append(None)  # scalars and plain tensors: whole
            continue
        if meta.partition is None:
            raise ValueError(
                "map requires the structured (spec) tier; a mapper-tier "
                "allocation has no leaves to build per-die views from"
            )
        if part is None:
            part = meta.partition
        elif not _partitions_equal(part, meta.partition):
            raise ValueError(
                "misaligned operands: all localized operands of map must "
                "share one partition"
            )
        view_args.append(meta.partition)
    if part is None:
        raise ValueError(
            "no localized operand carries a partition; pass spec= explicitly"
        )

    gd = 1
    for e in tuple(part.grid_dims):
        gd *= int(e)

    if streams is None:
        streams = _map_streams(gd)
    if len(streams) < gd:
        raise ValueError(f"need {gd} streams, got {len(streams)}")

    current = torch.cuda.current_stream()
    fork = torch.cuda.Event()
    fork.record(current)
    join_events = []
    for die in range(gd):
        s = streams[die]
        s.wait_event(fork)
        with torch.cuda.stream(s):
            args = tuple(
                _die_view(torch, t, p, die) if p is not None else t
                for t, p in zip(tensors, view_args)
            )
            fn(*args)
        e = torch.cuda.Event()
        e.record(s)
        join_events.append(e)
    for e in join_events:
        current.wait_event(e)


# ---------------------------------------------------------------------------
# Optional convenience: a `torch.localized` namespace.
#
# Purely additive sugar -- an attribute (and sys.modules entry) on the torch
# module, nothing about torch's own behavior changes. The names mirror the
# torch factory family; placement arguments are ours.
# ---------------------------------------------------------------------------


def _build_namespace(qualname):
    import types  # noqa: PLC0415

    ns = types.ModuleType(qualname)
    ns.__doc__ = (
        "Localized tensor factories attached by cuda.stf "
        "(see cuda.stf._experimental.interop.pytorch.install)."
    )
    ns.empty = localized_empty
    ns.zeros = localized_zeros
    ns.ones = localized_ones
    ns.full = localized_full
    ns.parameter = localized_parameter
    ns.empty_like = localized_empty_like
    ns.zeros_like = localized_zeros_like
    ns.ones_like = localized_ones_like
    ns.full_like = localized_full_like
    ns.release = release
    ns.get_meta = get_meta
    ns.spec_of = spec_of
    ns.grid_of = grid_of
    ns.map = map
    ns.views = views
    ns.live_metas = live_metas
    ns.placement_report = placement_report
    ns._cuda_stf_localized = True
    return ns


def install(name="localized"):
    """Attach the localized factory namespace to torch as ``torch.<name>``.

    After ``install()``, pytorch-style code reads naturally::

        import torch
        import cuda.stf._experimental as stf

        stf.interop.pytorch.install()
        stf.machine_init()
        grid = stf.exec_place_grid.from_devices([0, 1])

        w = torch.localized.parameter((4096, 4096), torch.bfloat16, grid,
                                      spec=(("blocked", 0), None))
        x = torch.localized.zeros((8, 4096), torch.float32, grid)
        torch.localized.placement_report(w)

    ``from torch.localized import zeros`` also works (a ``sys.modules``
    entry is registered). Idempotent; refuses to clobber a ``torch.<name>``
    attribute that is not ours; reversible with :func:`uninstall`. Returns
    the namespace. For codebases that prefer no patching, :func:`namespace`
    returns the same object without touching torch.
    """
    torch = _import_torch()
    existing = getattr(torch, name, None)
    if existing is not None and not getattr(existing, "_cuda_stf_localized", False):
        raise RuntimeError(
            f"torch.{name} already exists and does not belong to cuda.stf; "
            f"pick another name: install(name=...)"
        )
    ns = _build_namespace(f"torch.{name}")
    setattr(torch, name, ns)
    _sys.modules[f"torch.{name}"] = ns
    return ns


def namespace():
    """The localized factory namespace WITHOUT patching torch (for users or
    codebases that prefer explicit imports over convenience patching)."""
    return _build_namespace("cuda.stf.localized")


def uninstall(name="localized"):
    """Remove a namespace previously attached by :func:`install`."""
    torch = _import_torch()
    existing = getattr(torch, name, None)
    if existing is None:
        return
    if not getattr(existing, "_cuda_stf_localized", False):
        raise RuntimeError(f"torch.{name} does not belong to cuda.stf; not removing it")
    delattr(torch, name)
    _sys.modules.pop(f"torch.{name}", None)
