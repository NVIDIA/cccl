.. _libcudacxx-extended-api-mdspan:

Mdspan
======

.. toctree::
   :hidden:
   :maxdepth: 1

   mdspan/host_device_accessor
   mdspan/restrict_accessor
   mdspan/shared_memory_accessor
   mdspan/mdspan_to_dlpack
   mdspan/dlpack_to_mdspan

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`host/device/managed mdspan and accessor <libcudacxx-extended-api-mdspan-host-device-accessor>`
     - CUDA memory space ``mdspan`` and accessors
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`restrict mdspan and accessor <libcudacxx-extended-api-mdspan-restrict-accessor>`
     - ``mdspan`` and accessor with the *restrict* aliasing policy
     - CCCL 3.0.0
     - CUDA 13.0

   * - :ref:`shared_memory mdspan and accessor <libcudacxx-extended-api-mdspan-shared-memory-accessor>`
     - ``mdspan`` and accessor for CUDA shared memory
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`mdspan to dlpack <libcudacxx-extended-api-mdspan-mdspan-to-dlpack>`
     - Convert a ``mdspan`` to a ``DLTensor``
     - CCCL 3.2.0
     - CUDA 13.2

   * - :ref:`dlpack to mdspan <libcudacxx-extended-api-mdspan-dlpack-to-mdspan>`
     - Convert a ``DLTensor`` to a ``mdspan``
     - CCCL 3.2.0
     - CUDA 13.2
