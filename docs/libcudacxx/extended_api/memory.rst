.. _libcudacxx-extended-api-memory:

Memory
======

.. toctree::
   :hidden:
   :maxdepth: 1

   memory/align_down
   memory/align_up
   memory/aligned_size
   memory/discard_memory
   memory/get_device_address
   memory/is_address_from
   memory/is_aligned
   memory/ptr_rebind
   memory/ptr_in_range
   memory/ranges_overlap

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`aligned_size_t <libcudacxx-extended-api-memory-aligned-size>`
     - Defines an extent of bytes with a statically defined alignment.
     - libcu++ 1.2.0 / CCCL 2.0.0 (in ``<cuda/memory>`` since CCCL 3.1.0)
     - CUDA 11.1

   * - :ref:`discard_memory <libcudacxx-extended-api-memory-discard-memory>`
     - Writes indeterminate values to memory
     - libcu++ 1.6.0 / CCCL 2.0.0 (in ``<cuda/memory>`` since CCCL 3.1.0)
     - CUDA 11.5

   * - :ref:`get_device_address <libcudacxx-extended-api-memory-get-device-address>`
     - Returns a valid address to a device object
     - CCCL 2.8.0 (in ``<cuda/memory>`` since CCCL 3.1.0)
     - CUDA 12.9

   * - :ref:`is_address_from and is_object_from <libcudacxx-extended-api-memory-is_address_from>`
     - Check if a pointer or object is from a specific address space
     - CCCL 3.1.0
     - CUDA 13.1

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

   * - :ref:`ptr_in_range <libcudacxx-extended-api-memory-ptr_in_range>`
     - Check if a pointer is in a range
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`ranges_overlap <libcudacxx-extended-api-memory-ranges_overlap>`
     - Check if two ranges overlap
     - CCCL 3.2.0
     - CUDA 13.2
