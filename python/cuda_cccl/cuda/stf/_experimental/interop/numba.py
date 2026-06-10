# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba interop helpers for ``cuda.stf._experimental``.

This module provides:

* :func:`get_arg_numba` and :func:`numba_arguments` -- low-level converters
  from STF CAI objects to Numba CUDA device arrays.
* :func:`numba_task` -- context manager that opens an STF task and yields its
  arguments as Numba device arrays plus the task stream pointer.
* :func:`jit` -- an ergonomic ``@jit`` decorator that lets a Numba kernel be
  invoked directly with STF ``dep`` arguments. The first call compiles the
  underlying Numba kernel; subsequent calls reuse the cached compilation.

Numba is imported lazily inside each function. Importing this module does not
require Numba to be installed; calling a function that uses Numba without
``numba-cuda`` available raises :class:`ImportError` with an installation hint.
"""

from __future__ import annotations

from cuda.stf._experimental import context, dep, exec_place

_NUMBA_INSTALL_HINT = (
    "This functionality requires ``numba-cuda`` to be installed. "
    "Install it with e.g. ``pip install cuda-cccl[cu13]``."
)


def _import_numba_cuda():
    """Import :mod:`numba.cuda`, raising a friendly error if unavailable."""
    try:
        from numba import cuda  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_NUMBA_INSTALL_HINT) from exc
    return cuda


def get_arg_numba(task, index):
    """Return one task argument as a Numba device array.

    ``task.get_arg_cai(index)`` returns an stf_cai exposing the
    ``__cuda_array_interface__`` protocol.
    """
    cuda = _import_numba_cuda()
    return cuda.from_cuda_array_interface(
        task.get_arg_cai(index), owner=None, sync=False
    )


def numba_arguments(task):
    """Return all task buffer arguments as Numba device arrays.

    Same shape as ``task.args_cai()``: ``None``, a single array, or a tuple of
    arrays.
    """
    cuda = _import_numba_cuda()
    out = task.args_cai()
    if out is None:
        return None
    if isinstance(out, tuple):
        return tuple(
            cuda.from_cuda_array_interface(o, owner=None, sync=False) for o in out
        )
    return cuda.from_cuda_array_interface(out, owner=None, sync=False)


def numba_task(ctx, *args, symbol=None):
    """Context manager: ``ctx.task(*args)`` yielding ``(numba_arrays, stream)``.

    ``numba_arrays`` is a tuple of Numba CUDA device arrays (one per non-token
    dep), converted from each ``stf_cai`` via the CUDA Array Interface.

    ``stream`` is the STF task's stream pointer and implements the
    ``__cuda_stream__`` protocol, so it can be passed as ``stream=`` to
    ``cuda.compute`` algorithms.

    Example
    -------
    >>> from cuda.stf._experimental.interop.numba import numba_task
    >>> with numba_task(ctx, lA.read(), lB.read(), lC.rw()) as (args, stream):
    ...     cuda.compute.binary_transform(
    ...         args[0], args[1], args[2], OpKind.PLUS, N, stream=stream
    ...     )
    """
    cuda = _import_numba_cuda()

    def _to_numba(cai):
        return cuda.from_cuda_array_interface(cai, owner=None, sync=False)

    t = ctx.task(*args, symbol=symbol)

    class _NumbaTaskContext:
        def __enter__(self):
            t.start()
            try:
                cais = t.args_cai()
                stream = t.stream_ptr()
                if cais is None:
                    numba_args = ()
                elif isinstance(cais, tuple):
                    numba_args = tuple(_to_numba(c) for c in cais)
                else:
                    numba_args = (_to_numba(cais),)
            except Exception:
                t.end()
                raise
            return (numba_args, stream)

        def __exit__(self, exc_type, exc_val, exc_tb):
            t.end()
            return False

    return _NumbaTaskContext()


class stf_kernel_decorator:
    """Decorator-class wrapper around a Numba CUDA kernel for STF.

    Created by :func:`jit`; not intended for direct instantiation.
    """

    def __init__(self, pyfunc, jit_args, jit_kwargs):
        self._pyfunc = pyfunc
        self._jit_args = jit_args
        self._jit_kwargs = jit_kwargs
        self._compiled_kernel = None
        self._launch_cfg = None

    def __getitem__(self, cfg):
        if not isinstance(cfg, (tuple, list)):
            raise TypeError("use kernel[grid, block ([, exec_place, ctx])]")
        n = len(cfg)
        if n not in (2, 3, 4):
            raise TypeError(
                "use kernel[grid, block], kernel[grid, block, exec_place], "
                "or kernel[grid, block, exec_place, ctx]"
            )

        grid_dim = cfg[0]
        block_dim = cfg[1]
        ctx = None
        exec_pl = None

        if n >= 3:
            exec_pl = cfg[2]

        if n == 4:
            ctx = cfg[3]

        if exec_pl is not None and not isinstance(exec_pl, exec_place):
            raise TypeError("3rd item must be an exec_place")

        if ctx is not None and not isinstance(ctx, context):
            raise TypeError("4th item must be an STF context (or None to infer)")

        self._launch_cfg = (grid_dim, block_dim, ctx, exec_pl)
        return self

    def __call__(self, *args, **kwargs):
        if self._launch_cfg is None:
            raise RuntimeError(
                "launch configuration missing -- use kernel[grid, block], "
                "kernel[grid, block, exec_place], or "
                "kernel[grid, block, exec_place, ctx](...)"
            )

        gridDim, blockDim, ctx, exec_pl = self._launch_cfg

        dep_items = []
        for i, a in enumerate(args):
            if isinstance(a, dep):
                if ctx is None:
                    ld = a.get_ld()
                    ctx = ld.borrow_ctx_handle()
                dep_items.append((i, a))

        if ctx is None:
            raise TypeError(
                "No STF context could be inferred. Provide at least one dep argument "
                "or pass an explicit context via kernel[grid, block, exec_place, ctx]."
            )

        cuda = _import_numba_cuda()
        task_args = [exec_pl] if exec_pl else []
        task_args.extend(a for _, a in dep_items)

        with ctx.task(*task_args) as t:
            dev_args = list(args)
            for dep_index, (pos, _) in enumerate(dep_items):
                cai = t.get_arg_cai(dep_index)
                dev_args[pos] = cuda.from_cuda_array_interface(
                    cai.__cuda_array_interface__, owner=None, sync=False
                )

            if self._compiled_kernel is None:
                self._compiled_kernel = cuda.jit(*self._jit_args, **self._jit_kwargs)(
                    self._pyfunc
                )

            nb_stream = cuda.external_stream(t.stream_ptr())
            self._compiled_kernel[gridDim, blockDim, nb_stream](*dev_args, **kwargs)

        return None


def jit(*jit_args, **jit_kwargs):
    """STF-aware ``@jit`` decorator wrapping :func:`numba.cuda.jit`.

    A decorated function can be invoked as ``kernel[grid, block](*args)`` where
    arguments that are STF ``dep`` objects are transparently converted into
    Numba device arrays inside an STF task. The Numba compilation happens at
    first call.

    Examples
    --------
    Bare decorator::

        @jit
        def axpy(a, x, y):
            ...

    With Numba ``cuda.jit`` arguments::

        @jit(fastmath=True)
        def kernel(...):
            ...

    Then::

        axpy[grid, block](2.0, lX.read(), lY.rw())
    """
    if jit_args and callable(jit_args[0]):
        pyfunc = jit_args[0]
        return _build_kernel(pyfunc, (), jit_kwargs)

    def _decorator(fn):
        return _build_kernel(fn, jit_args, jit_kwargs)

    return _decorator


def _build_kernel(pyfunc, jit_args, jit_kwargs=None):
    if jit_kwargs is None:
        jit_kwargs = {}
    return stf_kernel_decorator(pyfunc, jit_args, jit_kwargs)


__all__ = [
    "get_arg_numba",
    "jit",
    "numba_arguments",
    "numba_task",
    "stf_kernel_decorator",
]
