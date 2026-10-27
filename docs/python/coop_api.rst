.. _cuda_coop-module:

``cuda.coop`` API Reference
===========================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

.. _coop-portable-api:
.. _portable-api:
.. _coop-common-api:

Common API
----------

The primitive functions below are compiler markers; ``register`` is a
host-side configuration function. The installed ``.pyi`` files are
authoritative for overload and result typing.

.. currentmodule:: cuda.coop

Backend registration
^^^^^^^^^^^^^^^^^^^^

See :ref:`registering a backend <coop-backend-registration>`.

.. autofunction:: register

Thread groups
^^^^^^^^^^^^^

See :ref:`thread groups <coop-thread-groups>` and
:ref:`participation and synchronization <coop-participation>` for the shared
execution model. ``ThreadGroup`` and ``ThreadHierarchy`` describe which
threads cooperate. A descriptor's availability does not imply that every
primitive supports that group.

.. autofunction:: this_thread
.. autofunction:: this_warp
.. autofunction:: this_block
.. autofunction:: this_cluster
.. autofunction:: this_grid

.. autoclass:: ThreadGroup
   :no-members:
   :no-special-members:

   .. automethod:: group_by

.. autoclass:: ThreadHierarchy
   :no-members:
   :no-special-members:

.. py:class:: Hierarchy

   Alias for :class:`ThreadHierarchy`.

Payloads and temporary storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``load(..., items_per_thread=N)`` returns a per-thread ``ThreadData`` payload.
Both returning Load and direct construction accept ``alignment``. Construct
``ThreadData`` directly for computed values or an existing output.
Thread-group descriptors describe the participating threads.

.. autofunction:: ThreadData

.. autoclass:: ThreadDataLike
   :no-members:
   :no-special-members:

.. autofunction:: TempStorage

.. autoclass:: TempStorageLike
   :no-members:
   :no-special-members:

Memory operations
^^^^^^^^^^^^^^^^^

Prefer ``values = load(group, source, items_per_thread=N)`` for fresh
per-thread values. ``load(group, source, output)`` fills an existing
per-thread payload and returns ``None``. To fill a shared array directly,
use ``load(group, source, shared, items_per_thread=N)``. This also returns
``None``; physical block thread ``r`` writes ``shared[r*N:(r+1)*N]``.
Global and shared arrays can be Load sources or Store destinations.
See :ref:`memory directions and staging <coop-shared-load-store>`.

.. autofunction:: load
.. autofunction:: store


.. _coop-numba-extensions:

Numba-CUDA-MLIR-qualified API
-----------------------------

.. py:module:: cuda.coop.numba_mlir

Use this module for the extensions below. Shared parameters and behavior
follow the :ref:`Common API <coop-common-api>`.

.. code-block:: python

   import cuda.coop.numba_mlir as coop

Qualified calls also accept fixed-size, one-dimensional local arrays where
the operation accepts per-thread payloads. ``local`` and ``shared`` expose
Numba-CUDA-MLIR's memory namespaces.
For Load, such a local array is an existing per-thread output. Shared arrays
instead serve as Load sources and Store destinations in both the common
and qualified APIs. Prefer returning Load unless existing per-thread storage
must be filled; see :ref:`choosing a Load form <coop-faq-load-forms>`.

.. currentmodule:: cuda.coop.numba_mlir

Payloads and temporary storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: ThreadData

.. autoclass:: TempStorage
   :no-members:
   :no-special-members:

Memory operations
^^^^^^^^^^^^^^^^^

.. autofunction:: load
.. autofunction:: store
