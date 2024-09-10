.. _cudax-module:

CUDA Experimental
=================

.. toctree::
   :hidden:
   :maxdepth: 3

   container
   memory_resource
   ${repo_docs_api_path}/cudax_api

``CUDA Experimental`` (``cudax``) provides experimental new features that are still in development and subject to change.
However, any feature within this library has important use cases and we encourage users to experiment with them.

Specifically, ``cudax`` provides:
   - :ref:`uninitialized storage <cudax-containers-uninitialized-buffer>`
   - :ref:`an owning type erased memory resource <cudax-memory-resource-async-any-resource>`
   - :ref:`stream-ordered memory resources <cudax-memory-resource-async>`
   - dimensions description functionality

Stability Guarantees
---------------------

There are no stability guarantees whatsoever. We reserve the right to change both the ABI and the API of any feature
within ``cudax`` at any time without notice.

Availability
-------------

Due to its experimental nature and the lack of stability guarantees, ``cudax`` is not shipped with the CUDA toolkit but
is solely available through github.
