.. _libcudacxx-extended-api-numeric-overflow_result:

``cuda::overflow_result``
=========================

.. code:: cpp

   template <class T>
   struct overflow_result
   {
        T    value;
        bool overflow;

        __host__ __device__
        constexpr explicit operator bool() const noexcept;
   };

The ``overflow_result`` struct is used to represent the result of arithmetic operations that may overflow. It contains the following members:

- ``value``: The result of the operation of type ``T``.
- ``overflow``: A boolean indicating whether an overflow occurred during the operation.

The ``operator bool()`` returns ``true`` if an overflow occurred, and ``false`` otherwise.
It can be used in conditional expressions to check whether an overflow occurred.

Example:

.. code:: cpp

    auto result = /* overflow operation */;
    if (result)
    {
        // Overflow occurred
    }

**Constraints**

- ``T`` must be an integer type.
