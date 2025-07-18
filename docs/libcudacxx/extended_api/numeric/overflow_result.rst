.. _libcudacxx-extended-api-numeric-overflow_result:

``cuda::overflow_result``
==========================

.. code:: cpp

   template <class T>
   struct overflow_result
   {
       T value;
       bool overflow;

       __host__ __device__ constexpr explicit operator bool() const noexcept;
   };

The ``overflow_result`` struct is used to represent the result of arithmetic operations that may overflow. It contains the following members:
- ``value``: The result of the operation of type ``T``.
- ``overflow``: A boolean indicating whether an overflow occurred during the operation.

**Constraints**

- ``T`` must be cv-unqualified integer type.
