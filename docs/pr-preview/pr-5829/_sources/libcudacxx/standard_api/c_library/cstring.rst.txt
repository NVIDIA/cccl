.. _libcudacxx-standard-api-cstring:

``<cuda/std/cstring>``
======================

``cuda::std::memset``
---------------------

.. code:: cuda

   __host__ __device__
   inline void* memset(void* dest, int ch, size_t count) noexcept;

See `std::memset <https://en.cppreference.com/w/cpp/string/byte/memset.html>`_ for the full documentation.

**Preconditions**

The following preconditions are only enabled with CCCL 3.2 or later:

    - ``dest`` is a valid pointer.
    - ``dest + count`` is a valid pointer.

A valid pointer is one that is not NULL and within the correct range if it belongs to the shared memory address space.

----

``cuda::std::memcpy``
---------------------

.. code:: cuda

   __host__ __device__
   inline void* memcpy(void* dest, const void* src, size_t count) noexcept;

See `std::memcpy <https://en.cppreference.com/w/cpp/string/byte/memcpy.html>`_  for the full documentation.

**Preconditions**

The following preconditions are only enabled with the CUDA Toolkit 13.2 and CCCL 3.2 or later:

    - ``src`` is a valid pointer.
    - ``src + count`` is a valid pointer.
    - ``dest`` is a valid pointer.
    - ``dest + count`` is a valid pointer.
    - ``src`` and ``dest`` don't overlap.

A valid pointer is one that is not NULL and within the correct range if it belongs to the shared memory address space.
