.. _libcudacxx-extended-api-memory:

Memory
======

.. toctree::
   :hidden:
   :maxdepth: 1

   memory/get_device_address
   memory/aligned_size
   memory/is_aligned
   memory/align_up
   memory/align_down
   memory/ptr_rebind

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`aligned_size_t <libcudacxx-extended-api-memory-aligned-size>`
     - Defines an extent of bytes with a statically defined alignment.
     - libcu++ 1.2.0 / CCCL 2.0.0
     - CUDA 11.1

   * - :ref:`get_device_address <libcudacxx-extended-api-memory-get-device-address>`
     - Returns a valid address to a device object
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`is_aligned <libcudacxx-extended-api-memory-is_aligned>`
     - Check if a pointer is aligned
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`align_up <libcudacxx-extended-api-memory-align_up>`
     - Align up a pointer to the specified alignment
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`align_down <libcudacxx-extended-api-memory-align_down>`
     - Align down a pointer to the specified alignment
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`ptr_rebind <libcudacxx-extended-api-memory-ptr_rebind>`
     - Rebind a pointer to a different type
     - CCCL 3.1.0
     - CUDA 13.1
