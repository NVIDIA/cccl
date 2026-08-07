.. _libcudacxx-extended-api-random-pcg64:

``pcg64``
=========

Defined in the ``<cuda/random>`` header.

``cuda::pcg64`` is a 128-bit state PCG XSL RR 128/64 engine that produces 64-bit unsigned integer outputs. It has a
period of ``2^128`` and supports logarithmic-time ``discard``. ``cuda::pcg64`` models the
`UniformRandomBitGenerator <https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator>`_ named requirement.

Example
-------

.. code:: cuda

    #include <cuda/random>

    __global__ void sample_kernel() {
        cuda::pcg64 rng(42);
        auto value = rng();
        rng.discard(10);
    }
