.. _libcudacxx-extended-api-memory-access-properties-apply-access-property:

cuda::apply_access_property
===============================

.. code:: cuda

   template <class ShapeT>
   __host__ __device__
   void apply_access_property(void const volatile* ptr, ShapeT shape, cuda::access_property::persisting) noexcept;
   template <class ShapeT>
   __host__ __device__
   void apply_access_property(void const volatile* ptr, ShapeT shape, cuda::access_property::normal) noexcept;

**Mandates**: :ref:`ShapeT <libcudacxx-extended-api-memory-access-shapes>` is either `std::size_t <https://en.cppreference.com/w/cpp/types/size_t>`_ or
:ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-access-shapes-aligned-size>`.

**Preconditions**: ``ptr`` points to a valid allocation for ``shape`` in the global memory address space.

**Effects**: no effects.

**Hint**: to prefetch ``shape`` bytes of memory starting at ``ptr`` while applying a property. Two properties are supported:

-  :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`
-  :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-normal>`

**Note**: in **Preconditions** "valid allocation for ``shape`` means
that:

-  if ``ShapeT`` is ``aligned_size_t<N>(sz)`` then ``ptr`` is aligned to an ``N``-bytes alignment boundary, and
-  for all offsets ``i`` in the extent of ``shape``, i.e., ``i`` in ``[0, shape)`` then the expression ``*(ptr + i)``
   does not exhibit undefined behavior.

**Note**: currently ``apply_access_property`` is ignored by nvcc and nvc++ on the host.

Example
-------

Given three input and output vectors ``x``, ``y``, and ``z``, and two arrays of coefficients ``a`` and ``b``,
all of length ``N``:

.. code:: cuda

   size_t N;
   int* x, *y, *z;
   int* a, *b;

the grid-strided kernel:

.. code:: cuda

   __global__ void update(int* const x, int const* const a, int const* const b, size_t N) {
       auto g = cooperative_groups::this_grid();
       for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
           x[idx] = a[idx] * x[idx] + b[idx];
       }
   }

updates ``x``, ``y``, and ``z`` as follows:

.. code:: cuda

   update<<<grid, block>>>(x, a, b, N);
   update<<<grid, block>>>(y, a, b, N);
   update<<<grid, block>>>(z, a, b, N);

The elements of ``a`` and ``b`` are used in all kernels. For certain values of ``N``, this may prevent parts of ``a``
and ``b`` from being evicted from the L2 cache, avoiding reloading these from memory in the subsequent ``update`` kernel.

With :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>` and
:ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`, we can
write kernels that specify that ``a`` and ``b`` are accessed more often than (``pin``) and as often as (``unpin``) other data:

.. code:: cuda

   __global__ void pin(int* a, int* b, size_t N) {
       auto g = cooperative_groups::this_grid();
       for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
           cuda::apply_access_property(a + idx, sizeof(int), cuda::access_property::persisting{});
           cuda::apply_access_property(b + idx, sizeof(int), cuda::access_property::persisting{});
       }
   }
   __global__ void unpin(int* a, int* b, size_t N) {
       auto g = cooperative_groups::this_grid();
       for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
           cuda::apply_access_property(a + idx, sizeof(int), cuda::access_property::normal{});
           cuda::apply_access_property(b + idx, sizeof(int), cuda::access_property::normal{});
       }
   }

which we can launch before and after the ``update`` kernels:

.. code:: cuda

   pin<<<grid, block>>>(a, b, N);
   update<<<grid, block>>>(x, a, b, N);
   update<<<grid, block>>>(y, a, b, N);
   update<<<grid, block>>>(z, a, b, N);
   unpin<<<grid, block>>>(a, b, N);

This does not require modifying the ``update`` kernel, and for certain values of ``N`` prevents ``a`` and ``b``
from having to be re-loaded from memory.

The ``pin`` and ``unpin`` kernels can be fused into the kernels for the ``x`` and ``z`` updates by modifying these kernels.
