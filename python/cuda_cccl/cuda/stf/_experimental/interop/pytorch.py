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
                    self._stream_ctx.__exit__(type(e), e, e.__traceback__)
                t.end()
                raise
            if tensors is None:
                return None
            if isinstance(tensors, tuple):
                return tensors
            return (tensors,)

        def __exit__(self, exc_type, exc_val, exc_tb):
            try:
                if self._stream_ctx is not None:
                    self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
            finally:
                t.end()
            return False

    return _PyTorchTaskContext()


__all__ = ["pytorch_task", "tensor_arg", "tensor_arguments"]
