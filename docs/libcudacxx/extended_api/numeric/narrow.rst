.. _libcudacxx-extended-api-numeric-narrow:

``cuda::narrow``
=====================

.. code:: cpp

   template <typename To, typename From>
   [[nodiscard]] constexpr
   To narrow(From from)

Casts the value ``from`` to type ``To`` and checks whether the value has changed.
Throws in host code and traps in device code.
Modelled after ``gsl::narrow``.

Example
-------

.. code:: cpp

    #include <cuda/narrow>
    #include <cuda/std/cassert>

    int main() {
        unsigned char r1 = narrow<unsigned char>(      200); // ok
        unsigned char r2 = narrow<unsigned char>(      300); // throws
        unsigned int  r3 = narrow<unsigned int >(2LL << 35); // throws
        unsigned int  r4 = narrow<unsigned int >(     -100); // throws
    }
