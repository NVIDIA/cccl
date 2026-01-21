.. _libcudacxx-extended-api-asynchronous-operations:

Asynchronous Operations
-----------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   asynchronous_operations/memcpy_async_tx
   asynchronous_operations/memcpy_async

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>`
     - Asynchronously copies one range to another
     - libcu++ 1.1.0 / CCCL 2.0.0
     - CUDA 11.0

   * - :ref:`cuda::memcpy_async_tx <libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx>`
     - Asynchronously copies one range to another with manual transaction accounting
     - libcu++ 1.2.0 / CCCL 2.0.0
     - CUDA 11.1

.. note::

  **Asynchronous operations** like `memcpy_async <libcudacxx-extended-api-asynchronous-operations-memcpy-async>`
  are non-blocking operations performed as-if by a new thread of execution.
