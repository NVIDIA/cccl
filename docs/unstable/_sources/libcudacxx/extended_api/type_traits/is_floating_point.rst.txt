.. _libcudacxx-extended-api-type_traits-is_floating_point:

``cuda::is_floating_point``
===========================

.. code:: cuda

   namespace cuda {

   template <class T>
   inline constexpr bool is_floating_point_v = __ implementation defined __;

   template <class T>
   using is_floating_point = cuda::std::bool_constant<is_floating_point_v<T>>;

   } // namespace cuda

Tells whether a type is a floating point type, including implementation defined extended floating point types.
Users are allowed to specialize the variable template for their own types, but CCCL does not provide support for any issues arising from that.
