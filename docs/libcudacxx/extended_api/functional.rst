.. _libcudacxx-extended-api-functional:

Function wrapper
-----------------

.. toctree::
   :hidden:
   :maxdepth: 1

   functional/proclaim_return_type
   functional/get_device_address

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`cuda::proclaim_return_type <libcudacxx-extended-api-functional-proclaim-return-type>`
     - Creates a forwarding call wrapper that proclaims return type
     - libcu++ 1.9.0 / CCCL 2.0.0 / CUDA 11.8

   * - ``cuda::proclaim_copyable_arguments``
     - Creates a forwarding call wrapper that proclaims that arguments can be freely copied before an invocation of the wrapped callable
     - CCCL 2.8.0

   * - :ref:`cuda::get_device_address <libcudacxx-extended-api-functional-get-device-address>`
     - Returns a valid address to a device object
     - CCCL 2.8.0
