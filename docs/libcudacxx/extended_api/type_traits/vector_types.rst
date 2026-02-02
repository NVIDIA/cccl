.. _libcudacxx-extended-api-type_traits-vector_types:

Vector Type Traits
==================

Defined in the ``<cuda/type_traits>`` header.

``cuda::is_vector_type_v``
--------------------------

.. code:: cuda

   namespace cuda {

   template <class T>
   inline constexpr bool is_vector_type_v = /* see below */;

   } // namespace cuda

``cuda::is_vector_type_v<T>`` is ``true`` if ``T`` is a CUDA vector type, such as ``int2``, ``float4``, or ``dim3``.

**Supported vector types**

- ``char1``, ``char2``, ``char3``, ``char4``.
- ``uchar1``, ``uchar2``, ``uchar3``, ``uchar4``.
- ``short1``, ``short2``, ``short3``, ``short4``.
- ``ushort1``, ``ushort2``, ``ushort3``, ``ushort4``.
- ``int1``, ``int2``, ``int3``, ``int4``.
- ``uint1``, ``uint2``, ``uint3``, ``uint4``.
- ``long1``, ``long2``, ``long3``, ``long4``, and alignment variants in CUDA Toolkit 13.0+.
- ``ulong1``, ``ulong2``, ``ulong3``, ``ulong4``, and alignment variants in CUDA Toolkit 13.0+.
- ``longlong1``, ``longlong2``, ``longlong3``, ``longlong4``, and alignment variants in CUDA Toolkit 13.0+.
- ``ulonglong1``, ``ulonglong2``, ``ulonglong3``, ``ulonglong4``, and alignment variants in CUDA Toolkit 13.0+.
- ``float1``, ``float2``, ``float3``, ``float4``.
- ``double1``, ``double2``, ``double3``, ``double4``, and alignment variants in CUDA Toolkit 13.0+.
- ``dim3``.

``cuda::is_extended_fp_vector_type_v``
--------------------------------------

.. code:: cuda

   namespace cuda {

   template <class T>
   inline constexpr bool is_extended_fp_vector_type_v = /* see below */;

   } // namespace cuda

``cuda::is_extended_fp_vector_type_v<T>`` is ``true`` if ``T`` is an extended floating-point vector type.

**Supported extended floating-point vector types**

- ``__half2``.
- ``__nv_bfloat162``.
- ``__nv_fp8x2_e4m3``, ``__nv_fp8x2_e5m2``, ``__nv_fp8x4_e4m3``, ``__nv_fp8x4_e5m2``.
- ``__nv_fp8x2_e8m0``, ``__nv_fp8x4_e8m0``.
- ``__nv_fp6x2_e2m3``, ``__nv_fp6x2_e3m2``, ``__nv_fp6x4_e2m3``, ``__nv_fp6x4_e3m2``.
- ``__nv_fp4x2_e2m1``, ``__nv_fp4x4_e2m1``.

``cuda::vector_type_t``
-----------------------

.. code:: cuda

   namespace cuda {

   template <class T, cuda::std::size_t Size>
   using vector_type_t = /* see below */;

   } // namespace cuda

``cuda::vector_type_t<T, Size>`` is a type alias that maps a scalar type ``T`` and a vector size ``Size`` to the corresponding CUDA vector type. If no valid vector type exists for the given combination, the type is ``void``. It supports both integral and floating-point vector types.

``cuda::scalar_type_t``
-----------------------

.. code:: cuda

   namespace cuda {

   template <class VectorType>
   using scalar_type_t = /* see below */;

   } // namespace cuda

``cuda::scalar_type_t<VectorType>`` extracts the scalar element type from a CUDA vector type. For example, ``cuda::scalar_type_t<int2>`` is ``int`` and ``cuda::scalar_type_t<float4>`` is ``float``. . It supports both integral and floating-point vector types.

Example
-------

.. code:: cuda

    #include <cuda/type_traits>

    // Check if a type is a vector type
    static_assert(cuda::is_vector_type_v<int2>);
    static_assert(cuda::is_vector_type_v<float4>);
    static_assert(cuda::is_vector_type_v<dim3>);
    static_assert(!cuda::is_vector_type_v<int>);

    // Get the vector type for a scalar type and size
    static_assert(cuda::std::is_same_v<cuda::vector_type_t<int, 2>, int2>);
    static_assert(cuda::std::is_same_v<cuda::vector_type_t<float, 4>, float4>);

    // Extract the scalar type from a vector type
    static_assert(cuda::std::is_same_v<cuda::scalar_type_t<int2>, int>);
    static_assert(cuda::std::is_same_v<cuda::scalar_type_t<float4>, float>);

    // Extended floating-point vector types
    static_assert(cuda::is_extended_fp_vector_type_v<__half2>);
    static_assert(cuda::std::is_same_v<cuda::vector_type_t<__half, 2>, __half2>);

See Also
--------

- :ref:`Tuple Protocol for Vector Types <libcudacxx-extended-api-vector-tuple-protocol>` - Tuple protocol support for CUDA vector types.
