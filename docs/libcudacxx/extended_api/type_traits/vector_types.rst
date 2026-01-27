.. _libcudacxx-extended-api-type_traits-vector_types:

Vector Type Traits
==================

Defined in the ``<cuda/type_traits>`` header.

``cuda::__is_vector_type_v``
----------------------------

.. code:: cuda

   namespace cuda {

   template <class T>
   inline constexpr bool __is_vector_type_v = /* see below */;

   } // namespace cuda

``cuda::__is_vector_type_v<T>`` is ``true`` if ``T`` is a CUDA vector type, such as ``int2``, ``float4``, or ``dim3``.

**Supported vector types**

- ``char1``, ``char2``, ``char3``, ``char4``
- ``uchar1``, ``uchar2``, ``uchar3``, ``uchar4``
- ``short1``, ``short2``, ``short3``, ``short4``
- ``ushort1``, ``ushort2``, ``ushort3``, ``ushort4``
- ``int1``, ``int2``, ``int3``, ``int4``
- ``uint1``, ``uint2``, ``uint3``, ``uint4``
- ``long1``, ``long2``, ``long3``, ``long4`` (and alignment variants in CTK 13.0+)
- ``ulong1``, ``ulong2``, ``ulong3``, ``ulong4`` (and alignment variants in CTK 13.0+)
- ``longlong1``, ``longlong2``, ``longlong3``, ``longlong4`` (and alignment variants in CTK 13.0+)
- ``ulonglong1``, ``ulonglong2``, ``ulonglong3``, ``ulonglong4`` (and alignment variants in CTK 13.0+)
- ``float1``, ``float2``, ``float3``, ``float4``
- ``double1``, ``double2``, ``double3``, ``double4`` (and alignment variants in CTK 13.0+)
- ``dim3``

``cuda::__is_extended_fp_vector_type_v``
----------------------------------------

.. code:: cuda

   namespace cuda {

   template <class T>
   inline constexpr bool __is_extended_fp_vector_type_v = /* see below */;

   } // namespace cuda

``cuda::__is_extended_fp_vector_type_v<T>`` is ``true`` if ``T`` is an extended floating-point vector type.

**Supported extended floating-point vector types**

- ``__half2`` (requires ``_CCCL_HAS_NVFP16()``)
- ``__nv_bfloat162`` (requires ``_CCCL_HAS_NVBF16()``)
- ``__nv_fp8x2_e4m3``, ``__nv_fp8x2_e5m2``, ``__nv_fp8x4_e4m3``, ``__nv_fp8x4_e5m2`` (requires ``_CCCL_HAS_NVFP8()``)
- ``__nv_fp8x2_e8m0``, ``__nv_fp8x4_e8m0`` (requires CTK 12.8+)
- ``__nv_fp6x2_e2m3``, ``__nv_fp6x2_e3m2``, ``__nv_fp6x4_e2m3``, ``__nv_fp6x4_e3m2`` (requires ``_CCCL_HAS_NVFP6()``)
- ``__nv_fp4x2_e2m1``, ``__nv_fp4x4_e2m1`` (requires ``_CCCL_HAS_NVFP4()``)

``cuda::__vector_type_t``
-------------------------

.. code:: cuda

   namespace cuda {

   template <class T, cuda::std::size_t Size>
   using __vector_type_t = /* see below */;

   } // namespace cuda

``cuda::__vector_type_t<T, Size>`` is a type alias that maps a scalar type ``T`` and a vector size ``Size`` to the corresponding CUDA vector type. If no valid vector type exists for the given combination, the type is ``void``.

**Supported scalar types and sizes**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - **Scalar Type**
     - **Valid Sizes**

   * - ``signed char``
     - 1, 2, 3, 4

   * - ``unsigned char``
     - 1, 2, 3, 4

   * - ``short``
     - 1, 2, 3, 4

   * - ``unsigned short``
     - 1, 2, 3, 4

   * - ``int``
     - 1, 2, 3, 4

   * - ``unsigned int``
     - 1, 2, 3, 4

   * - ``long``
     - 1, 2, 3, 4

   * - ``unsigned long``
     - 1, 2, 3, 4

   * - ``long long``
     - 1, 2, 3, 4

   * - ``unsigned long long``
     - 1, 2, 3, 4

   * - ``float``
     - 1, 2, 3, 4

   * - ``double``
     - 1, 2, 3, 4

   * - ``__half``
     - 2

   * - ``__nv_bfloat16``
     - 2

   * - ``__nv_fp8_e4m3``, ``__nv_fp8_e5m2``
     - 2, 4

   * - ``__nv_fp8_e8m0`` (CTK 12.8+)
     - 2, 4

   * - ``__nv_fp6_e2m3``, ``__nv_fp6_e3m2``
     - 2, 4

   * - ``__nv_fp4_e2m1``
     - 2, 4

``cuda::__has_vector_type_v``
-----------------------------

.. code:: cuda

   namespace cuda {

   template <class T, cuda::std::size_t Size>
   inline constexpr bool __has_vector_type_v = /* see below */;

   } // namespace cuda

``cuda::__has_vector_type_v<T, Size>`` is ``true`` if a valid CUDA vector type exists for the scalar type ``T`` and size ``Size``. It is equivalent to ``!cuda::std::is_same_v<cuda::__vector_type_t<T, Size>, void>``.

``cuda::__scalar_type_t``
-------------------------

.. code:: cuda

   namespace cuda {

   template <class VectorType>
   using __scalar_type_t = /* see below */;

   } // namespace cuda

``cuda::__scalar_type_t<VectorType>`` extracts the scalar element type from a CUDA vector type. For example, ``cuda::__scalar_type_t<int2>`` is ``int`` and ``cuda::__scalar_type_t<float4>`` is ``float``.

Example
-------

.. code:: cuda

    #include <cuda/type_traits>

    // Check if a type is a vector type
    static_assert(cuda::__is_vector_type_v<int2>);
    static_assert(cuda::__is_vector_type_v<float4>);
    static_assert(cuda::__is_vector_type_v<dim3>);
    static_assert(!cuda::__is_vector_type_v<int>);

    // Get the vector type for a scalar type and size
    static_assert(cuda::std::is_same_v<cuda::__vector_type_t<int, 2>, int2>);
    static_assert(cuda::std::is_same_v<cuda::__vector_type_t<float, 4>, float4>);

    // Check if a vector type exists
    static_assert(cuda::__has_vector_type_v<int, 4>);
    static_assert(!cuda::__has_vector_type_v<int, 5>);  // size 5 not supported

    // Extract the scalar type from a vector type
    static_assert(cuda::std::is_same_v<cuda::__scalar_type_t<int2>, int>);
    static_assert(cuda::std::is_same_v<cuda::__scalar_type_t<float4>, float>);

    // Extended floating-point vector types
    #if _CCCL_HAS_NVFP16()
    static_assert(cuda::__is_extended_fp_vector_type_v<__half2>);
    static_assert(cuda::std::is_same_v<cuda::__vector_type_t<__half, 2>, __half2>);
    #endif
