.. _cuda_stf_experimental-module:

``cuda.stf._experimental`` API Reference
==========================================

.. warning::
  ``cuda.stf._experimental`` is experimental. The API is subject to change
  without notice.

The core context, logical-data, task, and place types are implemented as a compiled
extension; they are covered in the :ref:`narrative guide <cccl-python-stf>` and the
:ref:`C++ CUDASTF documentation <stf>`. Because the extension is compiled (and mocked
during documentation builds), autodoc cannot introspect these types, so their public
surface is documented explicitly below. The pure-Python helper layers follow.

Contexts
--------

.. py:currentmodule:: cuda.stf._experimental

.. py:class:: context(use_graph=False, *, stream=None, handle=None)

   Owns a Stream Task Flow graph. All logical data and tasks belong to one context.
   Pass ``use_graph=True`` for the CUDA-graph backend, ``stream=`` a caller-owned CUDA
   stream (any ``__cuda_stream__`` object or raw pointer) to emit work on top of it, and
   ``handle=`` an :py:class:`async_resources` to share stream pools / cached graphs
   across contexts. Prefer ``with context() as ctx:`` so :py:meth:`finalize` runs on exit.

   .. py:method:: logical_data(buf, dplace=None, name=None)

      Register an existing buffer (NumPy array, CUDA Array Interface object, or Python
      buffer) as logical data. Registration placement defaults to ``data_place.host()``.
      The source must be C-contiguous; a read-only source rejects ``write()``/``rw()``.

   .. py:method:: logical_data_empty(shape, dtype=None, name=None, *, no_export=False)

      Allocate uninitialized logical data of the given shape/dtype.

   .. py:method:: logical_data_full(shape, fill_value, dtype=None, where=None, exec_place=None, name=None)

      Allocate and fill logical data with a constant. Supports any 1/2/4/8-byte element
      type without optional third-party packages.

   .. py:method:: logical_data_zeros(shape, dtype=None, **kwargs)
   .. py:method:: logical_data_ones(shape, dtype=None, **kwargs)

      Convenience wrappers over :py:meth:`logical_data_full`.

   .. py:method:: token()

      Create a token (buffer-less logical data) for pure ordering dependencies.

   .. py:method:: task(*args)

      Open a task. Positional args are dependencies (``ld.read()`` / ``write()`` /
      ``rw()``) and at most one :py:class:`exec_place`. Use as a context manager.

   .. py:method:: cuda_kernel(*args)

      Like :py:meth:`task` but for kernels described directly to STF as CUDA-graph nodes.

   .. py:method:: host_launch(*deps, fn, args=None, symbol=None)

      Schedule a Python callback with dependency tracking. Non-token dependencies are
      materialized as NumPy arrays and passed positionally to ``fn``; token dependencies
      are ordering-only and are not materialized or passed. Exceptions raised by ``fn``
      are captured and re-raised by a blocking :py:meth:`wait`/:py:meth:`finalize` or by
      :py:meth:`check_errors`.

   .. py:method:: wait(ld)

      Block until ``ld`` is available and return a host NumPy copy. Re-raises any pending
      host-callback exception.

   .. py:method:: fence()

      Return a raw ``CUstream`` (as ``int``) that completes when all pending tasks finish,
      without destroying the context.

   .. py:method:: check_errors()

      Re-raise the first pending host-callback exception, if any (and clear it). Use with
      caller-stream contexts after synchronizing the stream, since their :py:meth:`finalize`
      is asynchronous.

   .. py:method:: finalize()

      Run the graph and release resources. Blocking for a default context; asynchronous
      (non-blocking on the caller stream) for a context created with ``stream=``.

   .. py:attribute:: place_resources

      Borrowed :py:class:`exec_place_resources` owned by this context. Do not use past
      :py:meth:`finalize`.

.. py:class:: stackable_context()

   Nestable context supporting ``graph_scope()`` / ``while_loop()`` / ``repeat(count)``
   scopes and record-once graphs. Mirrors :py:class:`context` for ``logical_data*``,
   ``task``, ``host_launch``, ``token``, ``fence`` and ``check_errors``. ``finalize()``
   is only legal at root: every ``graph_scope`` / ``while_loop`` / ``repeat`` /
   :py:class:`LaunchableGraph` must be closed first, or ``finalize()`` raises.

   .. py:method:: graph_scope()
   .. py:method:: while_loop()
   .. py:method:: repeat(count)

      Return context managers for a nested graph, a conditional while loop, or a
      fixed-count repeat scope.

   .. py:method:: launchable_graph_scope()

      Return a context manager that instantiates the nested graph into a reusable
      ``cudaGraphExec_t`` launchable multiple times within the scope.

   .. py:method:: pop_prologue_shared()

      Return a storable :py:class:`LaunchableGraph` for a graph built after ``push()``.

Logical data and dependencies
-----------------------------

.. py:class:: logical_data

   A registered or allocated buffer tracked by a context. Created through the ``context``
   ``logical_data*`` / ``token`` factories, not directly.

   .. py:method:: read(dplace=None)
   .. py:method:: write(dplace=None)
   .. py:method:: rw(dplace=None)

      Build a :py:class:`dep` for a task/host_launch. Dependency placement defaults to
      ``data_place.affine()``. ``write()``/``rw()`` raise on read-only sources.

   .. py:attribute:: dtype
   .. py:attribute:: shape
   .. py:attribute:: symbol
   .. py:attribute:: readonly

      Metadata; ``readonly`` is ``True`` when the backing source forbids writes.

   .. py:method:: empty_like()

      Create a new logical data with the same shape/dtype metadata.

.. py:class:: dep

   The result of ``ld.read()``/``write()``/``rw()``. Also produced by the module-level
   :py:func:`read` / :py:func:`write` / :py:func:`rw` helpers.

.. py:function:: read(ld, dplace=None)
.. py:function:: write(ld, dplace=None)
.. py:function:: rw(ld, dplace=None)

   Functional forms of the dependency builders.

.. py:class:: AccessMode

   ``IntFlag`` of access modes: ``NONE``, ``READ``, ``WRITE``, ``RW``.

Tasks, kernels, and streams
---------------------------

.. py:class:: task

   Returned by ``context.task(...)``; used as a context manager. Buffer/stream accessors
   are valid only while the task is active.

   .. py:method:: get_arg(index)

      Raw device pointer (``int``) for dependency ``index``. Raises for token arguments.

   .. py:method:: get_arg_cai(index)
   .. py:method:: args_cai()

      CUDA Array Interface view(s) for the non-token arguments. The views advertise no
      stream (CAI ``stream`` is ``None``); launch your own work on the task stream(s)
      (:py:meth:`stream_ptr` for scalar tasks, :py:meth:`get_stream_at_index` /
      :py:meth:`get_stream_ptrs` for grids), which STF has already ordered behind the
      data's producers. Tokens are skipped.

   .. py:method:: stream_ptr()

      The task's :py:class:`CudaStream`.

   .. py:method:: get_grid_dims()
   .. py:method:: get_stream_at_index(place_index)
   .. py:method:: get_stream_ptrs()

      Grid-task helpers: grid shape ``(x, y, z, t)`` or ``None``, the per-place stream at a
      linear index, and the list of all place streams.

   .. py:method:: set_exec_place(exec_place)
   .. py:method:: set_symbol(name)

.. py:class:: cuda_kernel

   Returned by ``context.cuda_kernel(...)``. Adds ``launch(kernel, grid, block, args, shmem=0)``
   on top of the task accessors, describing a kernel as a native CUDA-graph node.

.. py:class:: CudaStream

   ``int`` subclass wrapping a raw ``CUstream``; implements ``__cuda_stream__`` and exposes
   ``.ptr``.

Places, grids, and resources
----------------------------

.. py:class:: exec_place

   Where a task runs. Construct with :py:meth:`device`, :py:meth:`host`,
   :py:meth:`current_device`, :py:meth:`green_ctx`, or :py:meth:`from_context`.

   .. py:staticmethod:: device(dev_id)
   .. py:staticmethod:: host()
   .. py:staticmethod:: current_device()
   .. py:staticmethod:: green_ctx(view, use_green_ctx_data_place=False)
   .. py:staticmethod:: from_context(ctx, dev_id=-1)

      Build a place from a device, the host, the current device, a green-context view, or
      an external ``CUcontext``. Green-context and external-context places retain the
      objects they reference.

   .. py:attribute:: kind
   .. py:attribute:: dims
   .. py:attribute:: size
   .. py:attribute:: backing_context

      ``kind`` is ``"host"``/``"device"``; ``dims``/``size`` describe grids;
      ``backing_context`` is the external object backing a ``from_context`` place (else
      ``None``), retained for the place's lifetime.

   .. py:method:: set_affine_data_place(dplace)
   .. py:attribute:: affine_data_place
   .. py:method:: pick_stream(resources, for_computation=True)
   .. py:method:: get_place(idx)

.. py:class:: exec_place_grid

   A grid of execution places (subclass of :py:class:`exec_place`).

   .. py:staticmethod:: from_devices(device_ids)
   .. py:staticmethod:: create(places, grid_dims=None, mapper=None)

      Build a grid from device ordinals, or from explicit places with an optional
      ``grid_dims`` shape (validated for rank, positivity, and product) and a ``mapper``
      partition function. The grid retains its sub-places.

.. py:class:: data_place

   Where logical data lives. Construct with :py:meth:`device`, :py:meth:`host`,
   :py:meth:`managed`, :py:meth:`affine`, :py:meth:`current_device`, :py:meth:`green_ctx`,
   or :py:meth:`composite`.

   .. py:staticmethod:: device(dev_id)
   .. py:staticmethod:: host()
   .. py:staticmethod:: managed()
   .. py:staticmethod:: affine()
   .. py:staticmethod:: current_device()
   .. py:staticmethod:: green_ctx(view)
   .. py:staticmethod:: composite(grid, mapper)

      ``composite`` retains both the grid and the ctypes mapper closure it references.

   .. py:attribute:: kind
   .. py:attribute:: device_id
   .. py:method:: allocate(nbytes, stream=None)
   .. py:method:: deallocate(ptr, nbytes, stream=None)

.. py:class:: exec_place_resources

   Per-place stream-pool registry. Construct standalone (``exec_place_resources()``) or
   borrow ``context.place_resources``.

.. py:class:: async_resources

   Shareable ``async_resources_handle``. Reuse one across contexts to amortize
   graph-instantiation and share stream pools; it must outlive every context it is passed to.

.. py:class:: LaunchableGraph

   Storable, shared-ownership handle for a re-launchable stackable graph, returned by
   ``stackable_context.pop_prologue_shared()``.

   .. py:method:: launch()
   .. py:method:: reset()
   .. py:attribute:: valid
   .. py:attribute:: exec_graph
   .. py:attribute:: stream
   .. py:attribute:: graph

      ``launch()`` replays the graph; ``reset()`` drops the shared reference (running
      ``pop_epilogue`` when it was the last one) and is idempotent; ``valid`` reports
      whether the handle still refers to a live graph. Assigning the handle to another
      variable aliases the same object -- there is no handle-duplication API -- so
      resetting one resets all aliases.

Green-context places
--------------------

.. autofunction:: cuda.stf._experimental.green_places.green_places

Record-once task graphs
-----------------------

Use ``task_graph()`` to create a record-once task graph. It returns a
``TaskGraph`` object, which is the context manager and launch handle for the
recorded graph.

.. autofunction:: cuda.stf._experimental.task_graph.task_graph

.. autoclass:: cuda.stf._experimental.task_graph.TaskGraph
  :members:
  :exclude-members: __init__

Device allocations
------------------

.. automodule:: cuda.stf._experimental.device_array
  :members:
  :undoc-members:

Path discovery
--------------

.. automodule:: cuda.stf._experimental.paths
  :members:
  :undoc-members:

Numba interop
-------------

.. automodule:: cuda.stf._experimental.interop.numba
  :members:
  :undoc-members:

PyTorch interop
---------------

.. automodule:: cuda.stf._experimental.interop.pytorch
  :members:
  :undoc-members:
