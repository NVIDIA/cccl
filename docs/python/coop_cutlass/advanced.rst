.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-advanced-topics:

Advanced topics
===============

This page is for kernels that run native CuTe code next to the primitives.
It covers three places where the two meet: a tile that native code stages in
shared memory, shared memory that both use in turn, and registers that
another object loads.

A walkthrough follows one complete kernel, in which a Tensor Core
accumulator moves from tensor memory into registers, through a top-k
selection and back, with the synchronization that dependent matrix
multiply-accumulate (MMA) iterations need:

.. toctree::
   :maxdepth: 1

   tensor_memory_topk

.. _coop-shared-load-store:

Staging a tile in shared memory
-------------------------------

This kernel stages each group's :term:`tile`: a native CuTe copy brings the
tile from global memory into shared memory, and ``load`` reads it from
there.

`Full example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_staging.py>`__

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_staging.py
   :language: python
   :start-after: docs: start cutlass-native-shared-staging
   :end-before: docs: end cutlass-native-shared-staging

The block has 64 threads, the value of ``_THREADS``. ``_OFFSET`` is 5 and
``_DEFAULT`` is -7. A ``width`` of 64 selects the block as the group, and a
``width`` of 1, 2, 4, 8, 16 or 32 selects groups of that many lanes. Each
step of the kernel has its rule:

The copy
   Each thread copies its own elements of its group's tile. ``source_tile``
   and ``shared_tile`` have the same ``layout``, and ``source_row`` and
   ``shared_row`` are row ``rank`` of each, so a thread writes the shared
   elements that match the global elements it reads. ``group_origin`` puts
   each group's tile where ``load`` looks for it: on a group of lanes,
   ``load`` adds the tiles of the block's earlier groups to its ``offset``.

The predicates
   ``cute.basic_copy_if`` copies the elements whose predicate is true and
   leaves the other elements of the destination as they were. This kernel
   wants ``_DEFAULT`` there, so it writes ``_DEFAULT`` to every element
   before the copy. A kernel that writes no default leaves those elements
   with whatever shared memory held, and must read no item loaded from
   them.

The barrier
   ``load`` can give a thread elements that other threads copied, so every
   thread must finish its copy before ``load`` runs.
   ``cute.arch.sync_threads()`` is the block barrier that waits for them.
   The package adds no barrier between your writes to shared memory and a
   primitive that reads them.

The load
   ``load`` takes the shared tensor as it takes any other source. It takes
   the element type from the tensor and returns a new payload in each
   thread. ``algorithm`` decides which elements a thread receives. The copy
   above is :term:`striped`: the thread with rank ``r`` copies elements
   ``r``, ``r + width``, ``r + 2 * width`` and so on. With
   ``algorithm="striped"`` the payload is striped too, and a thread
   receives the elements it copied. With the default algorithm the payload
   is :term:`blocked`, and a thread receives neighboring elements,
   whichever thread copied them. With one item per thread, or one thread in
   a group, the two layouts are the same.

   Load the layout that the next primitive reads. The ``store`` at the end
   of this kernel has the default algorithm, which reads a blocked payload.
   When the two layouts differ, ``striped=True`` therefore makes it write
   the tile to ``destination`` in another order.

The copy is native CuTe code, and CuTe sets its rules.
``cute.basic_copy_if`` takes a source, a destination and predicates with the
same number of elements, and source and destination element types of the
same width. The source and the destination must not overlap. Give the shared
tensor room for the tiles of every group of the block, which is
``_THREADS * items_per_thread`` elements here. Other native copies have
rules of their own for the copy atom and for the alignment of the views. An
asynchronous copy must also have completed before the barrier, so wait for
it first, in the way CuTe gives for that copy.

``store`` can stage a tile too. Pass a shared tensor as its destination, and
put a barrier between that call and the first read of the tile by another
thread.

A tensor from CuTe's ``SmemAllocator`` has bytes of its own, apart from the
scratch of every primitive. A tensor from ``TempStorage.reserve`` can share
its bytes with scratch, which the next section covers.

.. _coop-cutlass-typed-workspaces:

Typed workspaces
----------------

:meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>` returns
a one-dimensional CuTe shared tensor inside a ``TempStorage`` region, so
native CuTe code and primitives can take turns with the same bytes:

`Full example on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_workspace.py>`__

.. literalinclude:: ../../../python/cuda_coop_cutlass/examples/shared_workspace.py
   :language: python
   :start-after: docs: start cutlass-shared-workspace
   :end-before: docs: end cutlass-shared-workspace
   :dedent: 4

The block has 64 threads. ``workspace`` is 64 ``Float32`` elements in the
region of ``storage``, and ``prepare`` is native CuTe code that writes one
of them in each thread. Each thread then reads into a register the element
that thread ``63 - thread`` wrote. ``coop.sum`` takes its scratch from the
same ``storage``, which has ``sharing="shared"``, so the sum overwrites the
workspace. After the sum the kernel fills the workspace again.
``persistent`` comes from CuTe's ``SmemAllocator`` and keeps its values
across the sum, which the host checks.

Each ``storage.sync()`` is a block barrier between one use of the bytes and
the next:

#. After ``prepare``, so that every element is written before another
   thread reads it.
#. After the read, so that every thread holds its value in a register
   before the sum overwrites the workspace.
#. After the sum, so that the sum is finished with its scratch before the
   kernel writes the workspace again.
#. After the second filling, so that every element is written before
   another thread reads it.

The entry for
:meth:`TempStorage.reserve <cuda.coop.cutlass.TempStorage.reserve>` has the
rules for element types, ``sharing`` and barriers.

A reserved tensor can also be the source of ``load`` or the destination of
``store``, like the shared tensor of the previous section. The exceptions
are ``Float16`` and ``BFloat16`` tensors, which ``load`` and ``store``
reject when the kernel is compiled. The algorithms ``"direct"``,
``"striped"`` and ``"vectorize"`` use no scratch. A transpose algorithm uses
scratch on warps and groups of lanes, or on a block when ``items_per_thread``
is greater than 1. Single-item block transfers use no scratch. With
``sharing="shared"``, a transpose ``load`` or ``store`` that uses scratch and
takes the tensor's own ``TempStorage`` puts that scratch where the tensor
starts, and nothing checks for the overlap. Give that call scratch elsewhere:
leave ``temp_storage`` out, or pass another ``TempStorage``.

.. _coop-cutlass-producer-interop:

Producer-owned register loads
-----------------------------

A producer is an object that owns a register load: its
``__cuda_coop_thread_data_load__`` method, the hook, copies the producer's
data into each thread's registers and returns them. This producer owns a
CuTe register tensor, and ``calls`` counts how often its hook runs:

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_thread_data_producer.py>`__

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_thread_data_producer.py
   :language: python
   :start-after: # cutlass-producer-class-begin
   :end-before: # cutlass-producer-class-end

:meth:`ThreadData.load <cuda.coop.cutlass.ThreadData.load>` calls the hook
and makes a payload of what it returns. Here the kernel loads the producer's
registers, and the check on ``calls`` runs while the kernel is compiled. The
payload is a copy, so the write to ``registers[0]`` leaves ``values`` as it
was. The runtime loop adds 1 to item 0 on each trip, and
``to_tensor_ssa(like=source)`` returns the payload's items as a
``TensorSSA`` with the producer's ``shape`` and ``dtype``:

.. literalinclude:: ../../../python/cuda_coop_cutlass/tests/runtime/test_thread_data_producer.py
   :language: python
   :start-after: # cutlass-producer-load-begin
   :end-before: # cutlass-producer-load-end
   :dedent: 8

The producer:

* Chooses the copy: its copy atom, and the elements each thread takes.
* Makes sure its source is ready to be read, with whatever barrier or wait
  that takes.
* Owns its source for as long as the kernel reads it.
* Rejects a second call itself if its source can be read only once.

``ThreadData.load``:

* Looks for the hook on the object and its class. A hook that only
  ``__getattr__`` would supply is ignored, and an object without a hook is
  rejected when the kernel is compiled.
* Calls the hook once for each ``ThreadData.load`` in the kernel, while the
  kernel is compiled. The CuTe operations the hook performs become part of
  the kernel, so it can run a native copy there, such as one from tensor
  memory (TMEM) into registers.
* Passes on an error that the hook raises, unchanged.
* Converts the hook's result with
  :meth:`~cuda.coop.cutlass.ThreadData.from_payload`.

The hook returns a ``ThreadData``, or register values that ``from_payload``
takes, such as a CuTe register tensor or a ``TensorSSA``. The item count
must be known when the kernel is compiled, from the result's shape or from
``items_per_thread``. The optional ``items_per_thread`` and ``dtype`` of
``ThreadData.load`` work as they do for ``from_payload``, and an
``items_per_thread`` that is not a positive Python integer is rejected
before the hook runs. A result of another kind, such as a tensor in global,
shared or tensor memory, is rejected with an error that names the producer's
class and the class of the result.

A tensor in tensor memory that is passed to ``ThreadData.load`` by itself
is rejected as well. The tensor names no copy atom and does not say which
elements each thread takes, when it is ready or how long it lives. Write a
producer that supplies those, and test its copy and its synchronization on
a GPU that has tensor memory.

``values.to_tensor_ssa(like=source)`` reads ``source.shape`` and
``source.dtype`` for the shape and the element type of its result. A
``shape`` or ``dtype`` argument takes precedence over ``like``. The shape is
made of positive Python integers and holds exactly the payload's item count.
With neither ``shape`` nor ``like`` the result has the shape
``(items_per_thread,)``. The items keep their flat order. A payload keeps no
record of the shape of the tensor it came from. The kernel keeps track of
which thread owns which element of an MMA fragment.

Each conversion between a payload and a CuTe register tensor or
``TensorSSA`` copies, and the compiler can keep register values in local
memory. To see what a kernel does with them, read its linked cubin, as
:ref:`Checking and tuning a kernel <coop-cutlass-checking>` describes.

Three protocol classes describe a producer to a type checker. A producer
works without them, because ``ThreadData.load`` finds the hook by its name.

.. currentmodule:: cuda.coop.cutlass

.. autoclass:: ThreadDataLoadSource()
   :no-members:
   :no-special-members:

.. autoclass:: ThreadDataTensorMetadata()
   :no-members:
   :no-special-members:

.. autoclass:: ThreadDataSource()
   :no-members:
   :no-special-members:
