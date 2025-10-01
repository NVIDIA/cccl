.. _libcudacxx-extended-api-memory-access-properties:

Memory access properties
------------------------

.. toctree::
   :hidden:
   :maxdepth: 1

   memory_access_properties/access_property
   memory_access_properties/annotated_ptr
   memory_access_properties/apply_access_property
   memory_access_properties/associate_access_property

.. list-table::
   :widths: 25 45 30 30
   :header-rows: 1

   * - **Header**
     - **Content**
     - **CCCL Availability**
     - **CUDA Toolkit Availability**

   * - :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`
     - Represents a memory access property
     - libcu++ 1.6.0 / CCCL 2.0.0 /
     - CUDA 11.5

   * - :ref:`cuda::annotated_ptr <libcudacxx-extended-api-memory-access-properties-annotated-ptr>`
     - Binds an access property to a pointer
     - libcu++ 1.6.0 / CCCL 2.0.0
     - CUDA 11.5
   * - :ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`
     - Applies access property to memory
     - libcu++ 1.6.0 / CCCL 2.0.0
     - CUDA 11.5

   * - :ref:`cuda::associate_access_property <libcudacxx-extended-api-memory-access-properties-associate-access-property>`
     - Associates access property with raw pointer
     - libcu++ 1.6.0 / CCCL 2.0.0
     - CUDA 11.5
