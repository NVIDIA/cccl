.. _libcudacxx-extended-api-type_traits-is_floating_point:

cuda::is_floating_point
==============================


.. code:: cuda

   template <class T>
   inline constexpr bool is_floating_point_v;

   template <class T>
   using is_floating_point = cuda::std::bool_constant<is_floating_point_v<T>>;


Tells whether a type is a floating point type, including extended floating point types.
Users are allowed to specialize the variable template for their own types.
