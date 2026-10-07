.. _libcudacxx-extended-api-type_traits-is_arithmetic:

``cuda::is_arithmetic``
=======================

.. code:: cuda

   namespace cuda {

   template <class T>
   inline constexpr bool is_arithmetic_v = __ implementation defined __;

   template <class T>
   using is_arithmetic = cuda::std::bool_constant<is_arithmetic_v<T>>;

   } // namespace cuda

Tells whether a type is an arithmetic type, including implementation defined extended floating point types.
Users are allowed to specialize the variable template for their own types, but CCCL does not provide support for any issues arising from that.
