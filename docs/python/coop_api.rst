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
host-side configuration function. The installed ``.pyi``
files are authoritative for overload and result typing.

.. currentmodule:: cuda.coop

Backend registration
^^^^^^^^^^^^^^^^^^^^

See :ref:`registering a backend <coop-backend-registration>`.

.. autofunction:: register

Thread groups
^^^^^^^^^^^^^

See :ref:`thread groups <coop-thread-groups>` and
:ref:`participation and synchronization <coop-participation>` for the shared
execution model. A descriptor's availability does not imply that every
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

.. autofunction:: load
.. autofunction:: store
