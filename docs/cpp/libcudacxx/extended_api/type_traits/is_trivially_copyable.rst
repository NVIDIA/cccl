.. _libcudacxx-extended-api-type_traits-is_trivially_copyable:

``cuda::is_trivially_copyable``
=======================================

Defined in the ``<cuda/type_traits>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   constexpr bool is_trivially_copyable_v = /* see below */;

   template <typename T>
   using is_trivially_copyable = cuda::std::bool_constant<cuda::std::is_trivially_copyable_v<T>>;

   } // namespace cuda

``cuda::is_trivially_copyable_v`` trait evaluates if a type can be copied by copying its underlying bytes.
It extends ``cuda::std::is_trivially_copyable`` to also recognize CUDA extended floating-point vector types as trivially copyable.

The trait is true when ``T`` is any of the following:

- A type for which ``cuda::std::is_trivially_copyable_v<T>`` is true.
- An extended floating-point vector type, for example ``__half2``, ``__nv_bfloat162``.

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are trivially copyable when ``T`` is.
- ``cuda::std::array<T, N>``: trivially copyable when ``T`` is.
- ``cuda::std::pair<T1, T2>``: trivially copyable when both ``T1`` and ``T2`` are.
- ``cuda::std::tuple<Ts...>``: trivially copyable when all ``Ts...`` are.
- ``cuda::std::complex<T>``: trivially copyable when ``T`` is.
- ``cuda::complex<T>``: trivially copyable when ``T`` is.
- `Aggregates <https://en.cppreference.com/cpp/language/aggregate_initialization>`__: trivially copyable when all their members are.

  - On MSVC, recursive data-member inspection is not supported beyond the first level.

``const`` qualification is handled transparently, while ``volatile`` is compiler dependent.

Examples
--------

.. code:: cuda

   #include <cuda/type_traits>
   #include <cuda/std/array>
   #include <cuda/std/tuple>
   #include <cuda/std/utility>

   #include <cuda_fp16.h>

   // Standard trivially copyable types
   static_assert(cuda::is_trivially_copyable_v<int>);
   static_assert(cuda::is_trivially_copyable_v<float>);

   // Extended floating-point types
   static_assert(cuda::is_trivially_copyable_v<__half>);
   static_assert(cuda::is_trivially_copyable_v<__nv_bfloat16>);
   static_assert(cuda::is_trivially_copyable_v<__half2>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::complex<__half>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::complex<__half>>);

   // Composite types containing extended floating-point types
   static_assert(cuda::is_trivially_copyable_v<__half[4]>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::array<__half2, 4>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::pair<__half2, __half>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::tuple<__half, __half2>>);
   static_assert(cuda::is_trivially_copyable_v<cuda::std::pair<__half2, int>>);


`See it on Godbolt 🔗 <https://godbolt.org/z/PqccjfEv6>`__
