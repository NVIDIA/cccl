.. _libcudacxx-extended-api-random-mwc:

``mwc64x32`` and ``mwc128x32``
==============================

Defined in the ``<cuda/random>`` header as **experimental** feature in the namespace ``cuda::experimental``.

These scrambled *Multiply-With-Carry (MWC)* engines implement the `RandomNumberEngine <https://en.cppreference.com/w/cpp/named_req/RandomNumberEngine>`_ interface and produce unsigned 32-bit values over the entire range ``[0, 2^32 - 1]``.
They use the **MWC64X/32** and **MWC128X/32** algorithms from `Scrambled Multiply-with-Carry Pseudorandom Number Generators <https://arxiv.org/abs/2610.27313>`_.

.. note::

    These engines are not cryptographic generators and are not recommended for simulation or scientific computing requiring independently validated streams.

.. list-table::
   :header-rows: 1

   * - Engine
     - State size
     - Multiplier
     - Period
   * - ``cuda::experimental::mwc64x32``
     - 64 bits
     - ``0xFFF145F8``
     - ``(0xFFF145F8 * 2^32 - 2) / 2`` (approximately ``2^63``)
   * - ``cuda::experimental::mwc128x32``
     - 128 bits
     - ``0xFFCCCC66``
     - ``(0xFFCCCC66 * 2^96 - 2) / 2`` (approximately ``2^127``)

Both engines support host and device execution, constant evaluation, and logarithmic-time ``discard(unsigned long long)``.

Seeding
-------

Both engines can be initialized or reseeded with a 64-bit unsigned integer, including zero, or a seed sequence. The default seed is ``1``.

Parallel streams
----------------

Randomly seeding many ``mwc64x32`` engines can lead to overlapping streams because its period is relatively short. The user should use ``discard`` instead to partition a sequence into non-overlapping bounded streams. See the example below:

.. code:: cuda

    #include <cuda/random>

    __global__ void sample(unsigned* output) {
        unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
        cuda::experimental::mwc128x32 engine(42);
        engine.discard(tid * 1024); // Reserve 1024 outputs per thread
        output[tid] = engine();
    }

For performance reasons, each stream should be initialized once, then reuse the engine multiple times. Jumping ahead is more expensive than generating a value.
