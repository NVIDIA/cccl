.. _libcudacxx-extended-api-mdspan:

Mdspan
======

.. toctree::
   :hidden:
   :maxdepth: 1

   mdspan/host_device_accessor
   mdspan/restrict_accessor

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
