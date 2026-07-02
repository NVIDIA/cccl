.. _cccl-python-stf:

``cuda.stf._experimental``: Sequential Task Flow
================================================

The ``cuda.stf._experimental`` module provides a Python binding to the **Sequential
Task Flow (STF)** model for CUDA: you define logical data and submit tasks that read
or write that data; STF infers dependencies and orchestrates execution and data
movement. For the full description of the model, see the
:ref:`C++ CUDASTF documentation <stf>`.

The module ships inside the ``cuda-cccl`` wheel; install it with
``pip install cuda-cccl[cu13]`` (or ``[cu12]``). It is exposed under the
``_experimental`` subpackage because the Python API is still evolving and may change
without notice.

Example
-------

The following example registers three arrays as logical data and submits four GPU tasks
that scale and combine them. Each task only declares how it accesses its data
(``read()``, ``write()``, ``rw()``); from those annotations STF infers the dependency
graph, orders the tasks accordingly without any explicit synchronization, moves the data
to and from the device, and copies the results back into ``X``, ``Y``, and ``Z`` when the
context is finalized. ``scale`` and ``axpy`` are ordinary Numba CUDA kernels;
``t.stream_ptr()`` and ``numba_arguments(t)`` (described in *Tasks and interop* below)
bridge each task to its kernel launch.

.. literalinclude:: ../../python/cuda_cccl/tests/stf/interop/test_numba.py
   :language: python
   :pyobject: axpy_chain_example
   :caption: Real tasks with dependencies inferred from data accesses. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/stf/interop/test_numba.py>`__

Context and logical data
-------------------------

Create a **context** with ``context()`` (optionally ``use_graph=True`` for CUDA graph
execution). All logical data and tasks belong to one context.

When you are done submitting tasks, call ``finalize()`` to run the graph and
synchronize. A context does **not** finalize automatically when it is garbage
collected: it abandons its STF/CUDA resources and emits a ``ResourceWarning`` instead.
Always finalize explicitly, or -- preferably -- use the context as a context manager,
which finalizes on exit::

    with stf.context() as ctx:
        lX = ctx.logical_data(X)
        with ctx.task(lX.rw()) as t:
            ...
    # ctx.finalize() runs automatically here

**Logical data** represents a buffer that tasks access. Create it from existing
buffers or allocate new ones:

* ``logical_data(buf, ...)`` -- from a NumPy array or any object implementing the
  **CUDA Array Interface** (CuPy, PyTorch, Numba device arrays) or the Python buffer
  protocol. For GPU arrays, pass a :ref:`data_place <stf-data-place>`
  (e.g. ``data_place.device(0)``).
* ``logical_data_empty(shape, dtype, ...)`` -- uninitialized allocation.
* ``logical_data_full(shape, fill_value, ...)`` -- allocated and filled with a constant
  (like ``numpy.full()``).
* ``logical_data_zeros(...)`` / ``logical_data_ones(...)`` -- convenience wrappers.

Pass each logical data into a task with an access mode: ``read()``, ``write()``,
or ``rw()``. Example: ``ctx.task(lX.read(), lY.rw())``.

Tasks and interop
-----------------

Use ``with ctx.task(...) as t:`` to get a task handle. Inside the block:

* **Stream** -- ``t.stream_ptr()`` returns a ``CudaStream`` object for the task's CUDA
  stream. It implements the ``__cuda_stream__`` protocol (and also behaves like an
  integer raw pointer), so you can pass it straight to ``cuda.compute`` algorithms or
  wrap it for your framework (e.g. ``numba.cuda.external_stream(t.stream_ptr())`` or
  ``torch.cuda.ExternalStream(t.stream_ptr())``).
* **Buffer views** -- ``t.get_arg_cai(index)`` and ``t.args_cai()`` return object(s)
  that implement the **CUDA Array Interface**, so you can pass them to Numba
  (``cuda.from_cuda_array_interface(...)``), PyTorch (``torch.as_tensor(...)``), or
  CuPy (``cp.asarray(...)``).
* **Host callbacks** -- ``ctx.host_launch(...)`` schedules a Python callback with
  dependency tracking; the dependencies are unpacked as NumPy arrays and passed to the
  callback (e.g. ``ctx.host_launch(lX.read(), fn=lambda x: print(x.sum()))``).

For kernels that should become native CUDA graph nodes (instead of being captured from
a stream), use ``ctx.cuda_kernel(...)``. It accepts the same dependency and
``exec_place`` arguments as ``ctx.task(...)``, and the resulting object exposes a
``launch()`` method that describes the kernel to STF directly::

    with ctx.cuda_kernel(lX.read(), lY.rw(), symbol="axpy") as k:
        dX, dY = k.get_arg(0), k.get_arg(1)
        k.launch(kernel, grid=(4,), block=(256,),
                 args=[ctypes.c_int(N), ctypes.c_double(alpha), dX, dY])

Interop adapters
----------------

On top of the raw CUDA Array Interface, ``cuda.stf._experimental`` ships small,
**opt-in** adapters for Numba and PyTorch under ``cuda.stf._experimental.interop``.
Importing ``cuda.stf._experimental`` does **not** import Numba or PyTorch; the optional
runtime is imported lazily inside each adapter, and a missing dependency raises a clear
``ImportError`` at first use. You import only the adapter you need.

**PyTorch** (``cuda.stf._experimental.interop.pytorch``) -- ``pytorch_task`` opens a
task, makes the task's CUDA stream the current PyTorch stream for the duration of the
block, and yields the task arguments as ``torch.Tensor`` views::

    from cuda.stf._experimental.interop.pytorch import pytorch_task

    with pytorch_task(ctx, lX.read(), lY.rw()) as (x_tensor, y_tensor):
        y_tensor[:] = x_tensor * 2

``tensor_arg(task, index)`` and ``tensor_arguments(task)`` convert one or all
task arguments to tensors if you manage the task block yourself.

**Numba** (``cuda.stf._experimental.interop.numba``) -- ``numba_task`` opens a task and
yields ``(numba_arrays, stream)``, where ``stream`` can be passed straight to
``cuda.compute`` algorithms::

    from cuda.stf._experimental.interop.numba import numba_task

    with numba_task(ctx, lA.read(), lB.read(), lC.rw()) as (args, stream):
        cuda.compute.binary_transform(args[0], args[1], args[2], OpKind.PLUS, N, stream=stream)

The ``jit`` decorator wraps ``numba.cuda.jit`` so a kernel can be launched
directly with STF ``dep`` arguments; the conversion into device arrays (and the
task that scopes them) happens automatically::

    from numba import cuda

    from cuda.stf._experimental.interop.numba import jit

    @jit
    def axpy(a, x, y):
        i = cuda.grid(1)
        if i < x.size:
            y[i] = a * x[i] + y[i]

    axpy[grid, block](2.0, lX.read(), lY.rw())

``get_arg_numba(task, index)`` and ``numba_arguments(task)`` are the lower-level
converters used by these helpers.

Record-once task graphs
-----------------------

For repeated work, use ``task_graph()`` to record an STF task DAG once and launch it
many times. The graph owns a ``stackable_context`` exposed as ``graph.context``. Declare
logical data before recording, enter ``with graph:`` exactly once to submit tasks, then
call ``graph.launch()`` whenever the recorded graph should replay.

.. literalinclude:: ../../python/cuda_cccl/tests/stf/test_task_graph.py
   :language: python
   :pyobject: _record_add_graph
   :caption: Record a task graph once. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/stf/test_task_graph.py>`__

.. literalinclude:: ../../python/cuda_cccl/tests/stf/test_task_graph.py
   :language: python
   :pyobject: test_task_graph_relaunch
   :caption: Replay the recorded graph many times.

``graph.context.task(...)`` is intentionally valid only while ``with graph:``
is active. This catches the common mistake of submitting a task to the owned
context outside the recorded graph. Data declarations such as
``graph.context.logical_data(...)`` and ``graph.context.token()`` remain valid
outside the recording block so buffers and ordering tokens can be prepared
first.

Keep the Python objects backing recorded logical data alive for as long as the
graph may launch. ``task_graph()`` records work against the memory referenced by
objects such as NumPy arrays, Warp arrays, PyTorch tensors, or CuPy arrays; it
does not take ownership of those Python objects or extend their lifetime.

The lower-level ``stackable_context.pop_prologue_shared()`` API remains
available for advanced code that manages stackable scopes manually, but most
Python examples should prefer ``task_graph()``.

Places
------

.. _stf-exec-place:

* **Execution place** (``exec_place``) -- where the task runs. Pass as the first
  argument to ``ctx.task(...)``: ``exec_place.device(device_id)`` or ``exec_place.host()``.
  Example: ``ctx.task(exec_place.device(0), lX.read(), lY.rw())``.

.. _stf-data-place:

* **Data place** (``data_place``) -- where logical data lives: ``data_place.affine()``
  (the default -- lets the runtime place data near the task's execution place),
  ``data_place.host()``, ``data_place.device(device_id)``, ``data_place.managed()``.
  Use when creating logical data or in a dependency, e.g. ``lZ.rw(data_place.device(1))``.

Tokens
------

``ctx.token()`` creates a **token** (logical data with no buffer) for ordering tasks
without data transfer. Use ``token.read()`` or ``token.rw()`` in task dependencies.

Example collections
-------------------

For runnable examples, see the
`STF tests and examples <https://github.com/NVIDIA/cccl/tree/main/python/cuda_cccl/tests/stf>`_.
The ``interop/`` subdirectory exercises the Numba and PyTorch adapters (kernels,
tokens, multi-GPU, FDTD), and ``examples/`` holds larger end-to-end programs
(conjugate gradient, Cholesky, Burger, neural ODE).

For the full STF programming model, graph visualization, and C++ API, see
:ref:`CUDASTF (C++) <stf>`.

API Reference
-------------

- :ref:`cuda_stf_experimental-module`
