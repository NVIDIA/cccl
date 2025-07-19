.. _libcudacxx-extended-api-memory-access-properties-apply-access-property:

``cuda::apply_access_property``
===============================

Defined in header ``<cuda/annotated_ptr>``.

.. code:: cuda

   template <typename ShapeT>
   [[nodiscard]] __host__ __device__
   void apply_access_property(const volatile void* ptr, ShapeT shape, cuda::access_property::persisting) noexcept;

   template <typename ShapeT>
   [[nodiscard]] __host__ __device__
   void apply_access_property(const volatile void* ptr, ShapeT shape, cuda::access_property::normal) noexcept;

Prefetch memory in the L2 cache starting at ``ptr`` applying a residence control property.

**Constraints**

- ``ShapeT`` is either ``size_t`` or :ref:`cuda::aligned_size_t <libcudacxx-extended-api-memory-aligned-size>`.
- Two properties are supported:

    -  :ref:`cuda::access_property::persisting <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`
    -  :ref:`cuda::access_property::normal <libcudacxx-extended-api-memory-access-properties-access-property-normal>`

**Preconditions**

- ``ptr`` points to a valid allocation for ``shape`` in the *global memory* address space.

    -  if ``ShapeT`` is ``aligned_size_t<N>(sz)``, then ``ptr`` is aligned to an ``N``-bytes alignment boundary, and
    -  for all offsets ``i`` in the extent of ``shape``, namely ``i`` in ``[0, shape)``, then the expression ``*(ptr + i)`` does not exhibit undefined behavior.

*Note*:  currently ``apply_access_property`` is ignored on the host.

Example
-------

Given three input and output vectors ``x``, ``y``, and ``z``, and two arrays of coefficients ``a`` and ``b``, all of length ``N``:

.. code:: cuda

   size_t N;
   int* x, *y, *z;
   int* a, *b;

the grid-strided kernel:

.. code:: cuda

    __global__ void update(const int* x, const int* a, const int* b, size_t N) {
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

The elements of ``a`` and ``b`` are used in all kernels. For certain values of ``N``, this may prevent parts of ``a`` and ``b`` from being evicted from the L2 cache, avoiding reloading these from memory in the subsequent ``update`` kernel.

With :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>` and :ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>`, we can write kernels that specify that ``a`` and ``b`` are accessed more often in the ``pin`` kernel and with normal access in the ``unpin`` kernel:

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

This does not require modifying the ``update`` kernel, and for certain values of ``N`` prevents ``a`` and ``b`` from having to be re-loaded from memory.

The ``pin`` and ``unpin`` kernels can be fused into the kernels for the ``x`` and ``z`` updates by modifying these kernels.
