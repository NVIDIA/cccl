.. _cccl-python-stf:

``cuda.stf._experimental``: Sequential Task Flow
================================================

The ``cuda.stf._experimental`` module provides a Python binding to the **Sequential
Task Flow (STF)** model for CUDA: you define logical data and submit tasks that read
or write that data; STF infers dependencies and orchestrates execution and data
movement. For the full description of the model, see the
:ref:`C++ CUDASTF documentation <stf>`.

Install the module with ``pip install cuda-stf[cu13]`` (or ``[cu12]``). Install
``cuda-cccl`` as well when using ``cuda.compute`` or compiling external C++ code
that needs the libcudacxx, CUB, or Thrust headers.

The module is exposed under the ``_experimental`` subpackage because the Python
API is still evolving and may change without notice. CUDASTF is currently Linux-only.

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

.. literalinclude:: ../../python/cuda_stf/tests/stf/interop/test_numba.py
   :language: python
   :pyobject: axpy_chain_example
   :caption: Real tasks with dependencies inferred from data accesses. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_stf/tests/stf/interop/test_numba.py>`__

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

For a default context, ``finalize()`` blocks until all work completes, matching the
C++ ``ctx.finalize()`` contract. If you create a context bound to a caller-owned CUDA
stream (``context(stream=my_stream)``), ``finalize()`` is **asynchronous**: it returns
without synchronizing your stream, exactly like the C/C++ API, and you are responsible
for synchronizing that stream before relying on results or reusing the resources.

Host callbacks scheduled with ``host_launch()`` run on a CUDA host thread and cannot
raise directly into your code. Any exception they raise is captured and re-raised by
the next blocking ``wait()`` or blocking ``finalize()``. For caller-stream contexts
(whose ``finalize()`` is asynchronous and therefore cannot observe a callback that has
not run yet), call ``ctx.check_errors()`` after synchronizing your stream to surface a
captured callback exception.

**Logical data** represents a buffer that tasks access. Create it from existing
buffers or allocate new ones:

* ``logical_data(buf, ...)`` -- from a NumPy array or any object implementing the
  **CUDA Array Interface** (CuPy, PyTorch, Numba device arrays) or the Python buffer
  protocol. The **registration placement** defaults to ``data_place.host()``; pass a
  :ref:`data_place <stf-data-place>` (e.g. ``data_place.device(0)``) for data that
  already lives on a device. The source must be C-contiguous, and a read-only source
  (a const CUDA Array Interface export or a non-writable buffer) may only be used with
  ``read()`` -- requesting ``write()``/``rw()`` on it raises ``ValueError``. The Python
  object backing the buffer is retained for the lifetime of the logical data, so do not
  resize or free it while the logical data is in use.
* ``logical_data_empty(shape, dtype, ...)`` -- uninitialized allocation.
* ``logical_data_full(shape, fill_value, ...)`` -- allocated and filled with a constant
  (like ``numpy.full()``). Any 1/2/4/8-byte element type is supported with no optional
  third-party dependency.
* ``logical_data_zeros(...)`` / ``logical_data_ones(...)`` -- convenience wrappers.

Pass each logical data into a task with an access mode: ``read()``, ``write()``,
or ``rw()``. Example: ``ctx.task(lX.read(), lY.rw())``. Unlike the registration
placement, the **dependency placement** defaults to ``data_place.affine()`` (the
runtime places the working instance near the task's execution place); override it
per dependency, e.g. ``lZ.rw(data_place.device(1))``.

CUDA Array Interface views obtained inside a task (``t.get_arg_cai()`` / ``t.args_cai()``)
are valid **only while the task is active** (until ``stf_task_end()`` / the end of the
``with ctx.task(...)`` block). The views advertise **no stream** (the CUDA Array
Interface ``stream`` is ``None``). This is intentional: inside a task you must launch
your own work on the task stream(s) -- ``stream_ptr()`` for a scalar task, or
``get_stream_at_index()`` / ``get_stream_ptrs()`` for a grid -- and STF has already
ordered those streams behind the data's producers, so no extra synchronization is
required. Advertising a concrete stream would instead trigger a host-side synchronize
in consumers such as Numba (illegal during graph capture) for no benefit. The same
rule makes the grid case correct with a single view: STF enforces the per-place
dependencies, and each place's work runs on its own place stream.

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
``launch()`` method that describes the kernel to STF directly. The following excerpt
launches a precompiled AXPY ``kernel`` (see ``tests/stf/test_cuda_kernel.py`` for the
complete program)::

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
block, and yields the task arguments as ``torch.Tensor`` views. The following excerpt
stores ``2 * lX`` into ``lY`` (see ``tests/stf/interop/test_pytorch.py`` for the
complete program)::

    from cuda.stf._experimental.interop.pytorch import pytorch_task

    with pytorch_task(ctx, lX.read(), lY.rw()) as (x_tensor, y_tensor):
        y_tensor[:] = x_tensor * 2

``tensor_arg(task, index)`` and ``tensor_arguments(task)`` convert one or all
task arguments to tensors if you manage the task block yourself.

**Numba** (``cuda.stf._experimental.interop.numba``) -- ``numba_task`` opens a task and
yields ``(numba_arrays, stream)``, where ``stream`` can be passed straight to
``cuda.compute`` algorithms. The following excerpt sums two logical data with
``cuda.compute`` (see ``tests/stf/interop/test_cuda_compute.py`` for the complete
program)::

    from cuda.stf._experimental.interop.numba import numba_task

    with numba_task(ctx, lA.read(), lB.read(), lC.rw()) as (args, stream):
        cuda.compute.binary_transform(args[0], args[1], args[2], OpKind.PLUS, N, stream=stream)

The ``jit`` decorator wraps ``numba.cuda.jit`` so a kernel can be launched
directly with STF ``dep`` arguments; the conversion into device arrays (and the
task that scopes them) happens automatically. The following excerpt declares an
AXPY kernel and launches it on two logical data (see
``tests/stf/interop/test_decorator.py`` for the complete program)::

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

.. literalinclude:: ../../python/cuda_stf/tests/stf/test_task_graph.py
   :language: python
   :pyobject: _record_add_graph
   :caption: Record a task graph once. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_stf/tests/stf/test_task_graph.py>`__

.. literalinclude:: ../../python/cuda_stf/tests/stf/test_task_graph.py
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

Device-side loops
-----------------

A ``stackable_context`` (created directly or owned by a ``task_graph()``) can
nest scopes that become CUDA conditional graph nodes (CUDA 12.4+), so iteration
runs entirely on the device with no host round-trip per step:

* ``with ctx.repeat(count):`` -- run the body a fixed number of times.
* ``with ctx.while_loop() as loop:`` -- run the body while a condition holds.

A while body executes **at least once**. Call ``loop.continue_while(...)``
exactly once per body, after the body tasks; it schedules an internal task
that reads 1-element logical data and sets the continuation predicate for the
**next** iteration (tasks submitted after it still run in the current one).
Conditions compare device scalars against host constants and combine with
``&`` (continue while all hold) or ``|`` (continue while any holds), with
``~`` for negation -- the usual way to pair a convergence test with an
iteration cap so a non-converging solve cannot replay forever. The following
excerpt caps a solver loop on both the residual and an iteration counter (see
``tests/stf/examples/cg.py`` for the complete program)::

    liter = ctx.logical_data_zeros((1,), np.float64, name="iter")

    with ctx.while_loop() as loop:
        # ... solver step updating lresidual, and a task incrementing liter ...
        loop.continue_while((lresidual > tol) & (liter < max_iter))

``(lresidual > tol)`` is sugar for the canonical leaf constructor
``stf.cond(lresidual, ">", tol)``, which also suits generated code; both
forms lower onto a single condition task and one tiny kernel regardless of
the number of terms (at most 8, one ``&``-chain or one ``|``-chain per
condition). Use ``and`` / ``or`` and these expressions raise ``TypeError``.

See ``tests/stf/examples/cg.py`` and ``tests/stf/examples/bicgstab.py`` for
complete solvers built on device-side while loops.

Places
------

.. _stf-exec-place:

* **Execution place** (``exec_place``) -- where the task runs. Pass as the first
  argument to ``ctx.task(...)``: ``exec_place.device(device_id)`` or ``exec_place.host()``.
  Example: ``ctx.task(exec_place.device(0), lX.read(), lY.rw())``. Multi-device work can
  use ``exec_place_grid`` (via ``exec_place_grid.from_devices([...])`` or
  ``exec_place_grid.create(places, grid_dims=..., mapper=...)``); a grid retains its
  sub-places, and a ``composite`` data place retains its grid and partition-function
  closure, so those Python objects need not be kept alive separately.
  Places created from an external CUDA context (``exec_place.from_context(...)``, e.g.
  the green contexts produced by ``green_places()``) expose the backing object through
  the read-only ``place.backing_context`` property and keep it alive for the place's
  lifetime.

.. _stf-data-place:

* **Data place** (``data_place``) -- where logical data lives:
  ``data_place.host()`` (the default for **registration**, i.e. ``logical_data(...)``),
  ``data_place.affine()`` (the default for a **dependency**, letting the runtime place
  data near the task's execution place), ``data_place.device(device_id)``, and
  ``data_place.managed()``. Use when creating logical data or in a dependency, e.g.
  ``lZ.rw(data_place.device(1))``.

Localizing tensor allocations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A structured partition describes how tensor coordinates map onto a grid of execution
places. The same partition can back a composite data place, whose geometry-aware
allocation localizes each allocation block on the device owning most of its elements.

The Python API is C/row-major throughout: shapes, per-dimension specifications,
callback coordinates, and grid axes all use axis 0 as the outermost dimension,
exactly like a NumPy shape -- no reversal is ever needed. (Internally, the C and
C++ layers use a dimension-0-fastest convention; the conversion is private to the
Python binding.)

For example, on a system with two CUDA devices, distribute the rows of a
NumPy-shaped tensor between them::

    import math

    import numpy as np

    import cuda.stf._experimental as stf

    stf.machine_init()

    shape = (4096, 8192)  # (rows, columns), last dimension contiguous
    dtype = np.dtype(np.float32)

    grid = stf.exec_place_grid.from_devices([0, 1])

    # Distribute row bands over grid axis 0; each row stays intact.
    partition = stf.cute_partition.from_spec(
        shape,
        (("blocked", 0), None),
        grid.dims,
    )

    place = stf.data_place.composite_cute(grid, partition)
    ptr = place.allocate(shape, elemsize=dtype.itemsize)

    try:
        # Use ptr through CUDA Python, Numba, CuPy, or another CUDA
        # interoperability layer.
        ...
    finally:
        place.deallocate(ptr, math.prod(shape) * dtype.itemsize)

``allocate()`` returns a raw CUDA pointer rather than a NumPy array; an
interoperability layer must wrap that pointer before a Python array library can
use it. Partitioning the outermost dimension gives each device contiguous row
bands and avoids interleaving owners within the allocation granularity.

Physical placement is page-granular: memory is localized in blocks of the
device's allocation granularity (typically 2 MiB), and a block landing on the
boundary between two owners is placed with the majority owner. Placement can
therefore only approximate element ownership when ownership runs are smaller
than a page. Use ``placement_evaluate(grid, partition, elemsize)`` to score a
candidate mapping -- its ``accuracy`` is the fraction of bytes local to their
owner -- before committing memory.

**Tensor of tiles.** Multidimensional distributions match the page granularity
best when storage is reorganized into tiles: a ``(tiles_y, tiles_x, tile_y,
tile_x)`` tensor keeps each tile's payload contiguous, so every ownership run
spans a whole tile regardless of the distribution policy. The data partition is
the tile partition's specification with the payload dimensions left
undistributed (``None``)::

    tiles = (16, 16)          # tile grid, distributed
    tile = (512, 1024)        # per-tile payload, 2 MiB of float32: page-exact
    shape = tiles + tile

    grid = stf.exec_place_grid.create(places, grid_dims=(2, 2))

    partition = stf.cute_partition.from_spec(
        shape,
        (("blocked", 0), ("blocked", 1), None, None),
        grid.dims,
    )

    place = stf.data_place.composite_cute(grid, partition)
    ptr = place.allocate(shape, elemsize=4)

Ownership of element ``(i, j, y, x)`` depends only on the tile coordinates
``(i, j)``, so the same specification drives both tiled execution and data
placement. ``owner()`` answers the ownership question in closed form (C-order
coordinates in, C-order grid coordinates out), which makes the property easy
to check -- and is the primitive an adapter can use to reason about element
placement without re-implementing any policy::

    tile_partition = stf.cute_partition.from_spec(tiles, spec, grid.dims)
    assert partition.owner((i, j, y, x)) == tile_partition.owner((i, j))

Note that ``owner()`` is exact element-level ownership; the *physical*
placement of an allocation is page-granular and may only approximate it
(``placement_evaluate`` quantifies the difference). Note that tile-major storage is a real storage format: viewing it as
a conventional ``(rows, columns)`` spatial tensor requires a permutation, not a
reshape.

Tokens
------

``ctx.token()`` creates a **token** (logical data with no buffer) for ordering tasks
without data transfer. Use ``token.read()`` or ``token.rw()`` in task dependencies.

Example collections
-------------------

For runnable examples, see the
`STF tests and examples <https://github.com/NVIDIA/cccl/tree/main/python/cuda_stf/tests/stf>`_.
The ``interop/`` subdirectory exercises the Numba and PyTorch adapters (kernels,
tokens, multi-GPU, FDTD), and ``examples/`` holds larger end-to-end programs
(conjugate gradient, Cholesky, Burger, neural ODE).

For the full STF programming model, graph visualization, and C++ API, see
:ref:`CUDASTF (C++) <stf>`.

API Reference
-------------

- :ref:`cuda_stf_experimental-module`
