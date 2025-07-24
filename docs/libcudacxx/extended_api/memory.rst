.. _libcudacxx-extended-api-memory:

Memory
======

.. toctree::
   :hidden:
   :maxdepth: 1

   memory/is_aligned
   memory/align_up
   memory/align_down
   memory/ptr_rebind
   memory/is_address_from

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

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

   * - :ref:`is_address_from and is_object_from <libcudacxx-extended-api-memory-is_address_from>`
     - Check if a pointer or an object is from a specific address space
     - CCCL 3.1.0
     - CUDA 13.1
