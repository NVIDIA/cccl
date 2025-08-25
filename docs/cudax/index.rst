.. _cudax-module:

CUDA Experimental
=================

.. toctree::
   :hidden:
   :maxdepth: 1

   Overview <self>
   container
   memory_resource
   graph
   stf
   API reference <api/index>

``CUDA Experimental`` (``cudax``) provides experimental new features that are still in development and subject to change.
However, any feature within this library has important use cases and we encourage users to experiment with them.

Specifically, ``cudax`` provides:
   - :ref:`uninitialized storage <cudax-containers-uninitialized-buffer>`
   - :ref:`an owning type erased memory resource <cudax-memory-resource-any-async-resource>`
   - :ref:`stream-ordered memory resources <cudax-memory-resource-async>`
   - :ref:`graph functionality <cudax-graph>`
   - dimensions description functionality
   - :ref:`an implementation of the STF (Sequential Task Flow) programming model <stf>`

Stability Guarantees
---------------------

There are no stability guarantees whatsoever. We reserve the right to change both the ABI and the API of any feature
within ``cudax`` at any time without notice.

Availability
-------------

Due to its experimental nature and the lack of stability guarantees, ``cudax`` is not shipped with the CUDA toolkit but
is solely available through github.
