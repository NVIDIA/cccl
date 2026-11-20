.. _libcudacxx-extended-api-type_traits-is_bitwise_comparable:

``cuda::is_bitwise_comparable``
===============================

Defined in the ``<cuda/type_traits>`` header.

.. code:: cuda

   namespace cuda {

   template <typename T>
   constexpr bool is_bitwise_comparable_v = /* see below */;

   template <typename T>
   using is_bitwise_comparable = cuda::std::bool_constant<is_bitwise_comparable_v<T>>;

   } // namespace cuda

``cuda::is_bitwise_comparable_v`` trait evaluates if a type is comparable at bit-level, meaning that two instances of the same type can be compared as a raw sequence of bytes, independently from their semantics.

.. note::

  The following conditions generally prevent a type from being bitwise comparable:

  - The type is a floating-point type. ``NaN`` and ``+/-0`` values are not bitwise comparable.
  - The type has internal padding, for example a structure with ``char`` and ``int`` members.
  - The type has special comparison semantics, such as a user-defined ``operator==``.

``cuda::is_bitwise_comparable_v<T>`` is a user-specializable variable template that relies on ``cuda::std::has_unique_object_representations`` but excludes extended floating-point scalar and vector types.

The trait also propagates through composite types:

- C-style arrays: ``T[N]`` and ``T[]`` are bitwise comparable when ``T`` is.
- ``cuda::std::array<T, N>``: bitwise comparable when ``T`` is.
- ``cuda::std::pair<T1, T2>``: bitwise comparable when both ``T1`` and ``T2`` are and the object has no padding.
- ``cuda::std::tuple<Ts...>``: bitwise comparable when all ``Ts...`` are and the object has no padding.
- ``cuda::complex<T>``, ``cuda::std::complex<T>``: not bitwise comparable.
- Aggregates (types that can be initialized with a braced initializer list ``{}``): bitwise comparable when all their members are.

  - On MSVC, recursive data-member inspection is not supported beyond the first level.

``const``, ``volatile``, and ``const volatile`` qualifications are handled transparently.

Custom Specialization
---------------------

Users may specialize ``cuda::is_bitwise_comparable_v`` for their own types to indicate that two object representations can be compared bitwise, even when the implementation cannot determine this automatically.
The specialization must be provided for the unqualified type; cv-qualified forms are handled automatically.

.. code:: cuda

  struct MyType {
    double value;
  };

  template <>
  constexpr bool cuda::is_bitwise_comparable_v<MyType> = true;

  static_assert(cuda::is_bitwise_comparable_v<MyType>);
  static_assert(cuda::is_bitwise_comparable_v<const MyType>);

.. warning::

  Users are responsible for ensuring that the type is actually bitwise comparable when specializing this variable template. Otherwise, the behavior of functions that rely on this trait is undefined. For the reasons described above, bitwise comparison is especially problematic for floating-point types and types with internal padding.

Examples
--------

.. code:: cuda

   #include <cuda/type_traits>

   // Integer types have unique object representations
   static_assert(cuda::is_bitwise_comparable_v<int>);
   static_assert(cuda::is_bitwise_comparable_v<unsigned>);
   static_assert(cuda::is_bitwise_comparable_v<char>);
   static_assert(cuda::is_bitwise_comparable_v<int[4]>);

   // Floating-point types do not
   static_assert(!cuda::is_bitwise_comparable_v<float>);
   static_assert(!cuda::is_bitwise_comparable_v<double>);
