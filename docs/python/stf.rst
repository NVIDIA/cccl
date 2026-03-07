.. _cccl-python-stf:

``cuda.stf``: Sequential Task Flow
==================================

The ``cuda.stf`` module provides a Python binding to the **Sequential Task Flow (STF)**
model for CUDA: you define logical data and submit tasks that read or write that data;
STF infers dependencies and orchestrates execution and data movement. For the full
description of the model, see the :ref:`C++ CUDASTF documentation <stf>`.

Example
-------

The following example creates a context, registers three arrays as logical data, and
submits four tasks with different read/write annotations. STF orders the tasks so that
dependencies are respected (e.g. the task that writes ``Y`` runs after the one that
reads ``X`` and writes ``Y``).

.. literalinclude:: ../../python/cuda_cccl/tests/stf/test_context.py
   :language: python
   :pyobject: test_ctx3
   :caption: Context, logical data, and tasks. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/stf/test_context.py>`__

Context and logical data
-------------------------

Create a **context** with :func:`context() <cuda.stf.context>` (optionally
``use_graph=True`` for CUDA graph execution). All logical data and tasks belong to
one context. When you are done submitting tasks, call :meth:`finalize() <context.finalize>`
to run the graph and synchronize (or let the context be destroyed; ``finalize`` is
called automatically).

**Logical data** represents a buffer that tasks access. Create it from existing
buffers or allocate new ones:

* :meth:`logical_data(buf, ...) <context.logical_data>` — from a NumPy array or any
  object implementing the **CUDA Array Interface** (CuPy, PyTorch, Numba device arrays)
  or the Python buffer protocol. For GPU arrays, pass :ref:`data_place <stf-data-place>`
  (e.g. ``data_place.device(0)``).
* :meth:`logical_data_empty(shape, dtype, ...) <context.logical_data_empty>` —
  uninitialized allocation.
* :meth:`logical_data_full(shape, fill_value, ...) <context.logical_data_full>` —
  allocated and filled with a constant (like ``numpy.full()``).
* :meth:`logical_data_zeros` / :meth:`logical_data_ones` — convenience wrappers.

Pass each logical data into a task with an access mode: :meth:`read()`, :meth:`write()`,
or :meth:`rw()`. Example: ``ctx.task(lX.read(), lY.rw())``.

Tasks and interop
-----------------

Use ``with ctx.task(...) as t:`` to get a task handle. Inside the block:

* **Stream** — :meth:`t.stream_ptr() <task.stream_ptr>` returns the task’s CUDA stream
  (as an integer). Wrap it for your framework (e.g. ``numba.cuda.external_stream(t.stream_ptr())``
  or ``torch.cuda.ExternalStream(t.stream_ptr())``).
* **Buffer views** — :meth:`t.get_arg_cai(index) <task.get_arg_cai>` and
  :meth:`t.args_cai() <task.args_cai>` return object(s) that implement the
  **CUDA Array Interface**, so you can pass them to Numba (``cuda.from_cuda_array_interface(...)``),
  PyTorch (``torch.as_tensor(...)``), or CuPy (``cp.asarray(...)``).

The ``cuda.stf`` package does not ship Numba/PyTorch helpers; see
`tests/stf/numba_helpers.py <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/stf/numba_helpers.py>`_,
`numba_decorator.py <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/stf/numba_decorator.py>`_,
and `pytorch_task.py <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/stf/pytorch_task.py>`_
for examples.

Places
------

.. _stf-exec-place:

* **Execution place** (``exec_place``) — where the task runs. Pass as the first
  argument to ``ctx.task(...)``: ``exec_place.device(device_id)`` or ``exec_place.host()``.
  Example: ``ctx.task(exec_place.device(0), lX.read(), lY.rw())``.

.. _stf-data-place:

* **Data place** (``data_place``) — where logical data lives: ``data_place.host()``,
  ``data_place.device(device_id)``, ``data_place.managed()``. Use when creating
  logical data or in a dependency, e.g. ``lZ.rw(data_place.device(1))``.

Tokens
------

:meth:`ctx.token() <context.token>` creates a **token** (logical data with no buffer)
for ordering tasks without data transfer. Use ``token.read()`` or ``token.rw()`` in
task dependencies.

Example collections
-------------------

For runnable examples (Numba kernels, PyTorch, tokens, multi-GPU, FDTD), see the
`STF tests and examples <https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/stf>`_.

For the full STF programming model, graph visualization, and C++ API, see
:ref:`CUDASTF (C++) <stf>`.
