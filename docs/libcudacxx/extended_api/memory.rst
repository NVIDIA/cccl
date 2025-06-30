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

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`is_aligned <libcudacxx-extended-api-memory-is_aligned>`
     - Generate a bitmask
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`align_up <libcudacxx-extended-api-memory-align_up>`
     - Reverse the order of bits
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`align_down <libcudacxx-extended-api-memory-align_down>`
     - Insert a bitfield
     - CCCL 3.1.0
     - CUDA 13.1

   * - :ref:`ptr_rebind <libcudacxx-extended-api-memory-ptr_rebind>`
     - Extract a bitfield
     - CCCL 3.1.0
     - CUDA 13.1
