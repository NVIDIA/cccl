.. _libcudacxx-extended-api-functional:

Functional
----------

.. toctree::
   :hidden:
   :maxdepth: 1

   functional/proclaim_return_type
   functional/maximum_minimum
   functional/operator_properties
   memory/get_device_address

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::maximum <libcudacxx-extended-api-functional-maximum-minimum>`
     - Returns the maximum of two values
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::minimum <libcudacxx-extended-api-functional-maximum-minimum>`
     - Returns the minimum of two values
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::proclaim_return_type <libcudacxx-extended-api-functional-proclaim-return-type>`
     - Creates a forwarding call wrapper that proclaims return type
     - libcu++ 1.9.0 / CCCL 2.0.0
     - CUDA 11.8

   * - ``cuda::proclaim_copyable_arguments``
     - Creates a forwarding call wrapper that proclaims that arguments can be freely copied before an invocation of the wrapped callable
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::get_device_address <libcudacxx-extended-api-memory-get-device-address>`
     - Returns a valid address to a device object
     - CCCL 2.8.0
     - CUDA 12.9

   * - :ref:`cuda::is_associative_v <libcudacxx-extended-api-functional-operator-properties>`, :ref:`cuda::is_commutative_v <libcudacxx-extended-api-functional-operator-properties>`, :ref:`cuda::identity_element() <libcudacxx-extended-api-functional-operator-properties>`, :ref:`cuda::absorbing_element() <libcudacxx-extended-api-functional-operator-properties>`
     - Determines if an operator is associative for a type
     - CCCL 3.3.0
     - CUDA 13.3
