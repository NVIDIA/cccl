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

Tells whether a type is an arithmetic type, including implementation defined extended
floating point types. This includes types such as ``__half``, ``__nv_bfloat16_t`` and
other CUDA builtin floating point types not normally considered to be arithmetic or
floating point types by the standard traits.

A type ``T`` is considered arithmetic if and only if it satisfies the following
conditions:

#. It is either an integral or floating point type (in the case of floating point types,
   ``cuda::is_floating_point_v<T>`` must evaluate to ``true``).
#. All of the arithmetic operators (``*``, ``+``, ``-``, ``/``) are defined (possibly in
   combination with the usual arithmetic conversions) for the type.

If all of the above conditions are satisfied, then users are allowed to specialize the
variable template (``cuda::is_arithmetic_v``) for their own types, but CCCL does NOT
provide support for any issues arising from this.
