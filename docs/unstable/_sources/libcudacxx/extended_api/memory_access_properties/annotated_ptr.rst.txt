.. _libcudacxx-extended-api-memory-access-properties-annotated-ptr:

``cuda::annotated_ptr``
=======================

Defined in header ``<cuda/annotated_ptr>``.

``cuda::annotated_ptr`` is a pointer annotated with an access property that *may* be applied to its memory operations.

.. code:: cuda

   namespace cuda {

   template<typename Type, typename Property>
   class annotated_ptr {
   public:
     using value_type      = Type;
     using size_type       = size_t;
     using reference       = value_type&;
     using pointer         = value_type*;
     using const_pointer   = const value_type*;
     using difference_type = ptrdiff_t;

     annotated_ptr() noexcept = default;

     __host__ __device__ explicit constexpr annotated_ptr(pointer) noexcept;

     template <typename RuntimeProperty>
     __host__ __device__ annotated_ptr(pointer, RuntimeProperty) noexcept;

     template <typename T, typename P>
     __host__ __device__ annotated_ptr(const annotated_ptr<T,P>&) noexcept;

     __host__ __device__ constexpr explicit operator bool() const noexcept;

     [[nodiscard]] __host__ __device__ pointer   get() const noexcept;
     [[nodiscard]] __host__ __device__ reference operator*() const noexcept;
     [[nodiscard]] __host__ __device__ pointer   operator->() const noexcept;
     [[nodiscard]] __host__ __device__ reference operator[](ptrdiff_t) const noexcept;
     [[nodiscard]] __host__ __device__ constexpr difference_type operator-(annotated_ptr) const noexcept;

   private:
     pointer  ptr;  // exposition only
     Property prop; // exposition only
   };

   } // namespace cuda

.. note::
  If ``Property`` is :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`,
  namely a dynamic property with a runtime value,
  then ``sizeof(cuda::annotated_ptr<Type, cuda::access_property>) == 2 * sizeof(Type*)``. Otherwise, its size is ``sizeof(Type*)``.

In contrast to :ref:`cuda::associate_access_property <libcudacxx-extended-api-memory-access-properties-associate-access-property>`, ``cuda::annotated_ptr`` maintains the association between the pointer and the property when passed across translation units.

**Constraints**

``Property`` is one of:

-  :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`,
-  :ref:`cuda::access_property::global <libcudacxx-extended-api-memory-access-properties-access-property-global>`,
-  :ref:`cuda::access_property::persisting <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`,
-  :ref:`cuda::access_property::normal <libcudacxx-extended-api-memory-access-properties-access-property-normal>`,
-  :ref:`cuda::access_property::streaming <libcudacxx-extended-api-memory-access-properties-access-property-streaming>` or,
-  :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>`:
   a type-erased specification that allows ``annotated_ptr`` to set the access property at runtime value.

**Semantics**

.. list-table::
   :widths: 25 30 40
   :header-rows: 1

   * - Pointer Expression
     - ``cuda::annotated_ptr<T, P>``
     - Description

   * - ``T* a``
     - ``cuda::annotated_ptr<T, P> a``
     - Non-``const`` pointer to non-``const`` memory

   * - ``T const * a``
     - ``cuda::annotated_ptr<T const, P> a``
     - Non-``const`` pointer to ``const`` memory

   * - ``T* const a``
     - ``const cuda::annotated_ptr<T, P> a``
     - ``const`` pointer to non-``const`` memory

   * - ``T const* const a``
     - ``const cuda::annotated_ptr<T const, P> a``
     - ``const`` pointer to ``const`` memory

   * - ``val = *a;``
     - ``val = *a;``
     - Dereference operator to load an element

   * - ``*a = val;``
     - ``*a = val;``
     - Dereference operator to store an element

   * - ``val = a[n];``
     - ``val = a[n];``
     - Subscript operator to load an element

   * - ``a[n] = val;``
     - ``a[n] = val;``
     - Subscript operator to store an element

   * - ``T* a = nullptr;``
     - ``annotated_ptr<T, P> a = nullptr;``
     - ``nullptr`` initialization

   * - ``n = a - b;``
     - ``n = a - b;``
     - Difference operator

   * - ``if (a) { ... }``
     - ``if (a) { ... }``
     - Bool conversion

*Note*: It is not a drop-in replacement for pointers since, among others, it does not:

-  model any `Iterator <https://en.cppreference.com/w/cpp/iterator>`_ concept,
-  implement `cuda::std::pointer_traits <https://en.cppreference.com/w/cpp/memory/pointer_traits>`_,    `cuda::std::iterator_traits <https://en.cppreference.com/w/cpp/iterator/iterator_traits>`_, etc.
-  have the same variance as pointer.

----

Constructors and Assignment
---------------------------

Default constructor
~~~~~~~~~~~~~~~~~~~

.. code:: cuda

   annotated_ptr() noexcept = default;

**Effects**:  as if constructed by ``annotated_ptr(nullptr)``;

Constructor from pointer
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cuda

   constexpr explicit annotated_ptr(pointer ptr);

Constructs an ``annotated_ptr`` requesting associating ``ptr`` with ``Property``.

**Constraints**:

- If ``Property`` is :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`, :ref:`cuda::access_property::global <libcudacxx-extended-api-memory-access-properties-access-property-global>`,  :ref:`cuda::access_property::normal <libcudacxx-extended-api-memory-access-properties-access-property-normal>`, :ref:`cuda::access_property::streaming <libcudacxx-extended-api-memory-access-properties-access-property-streaming>`, :ref:`cuda::access_property::persisting <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`, or `cuda::access_property` (dynamic).

**Preconditions**:

- If ``Property`` is :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`, then ``ptr`` must be a generic pointer that is a valid pointer to the *shared memory* address space.
- If ``Property`` is  not :ref:`cuda::access_property::shared <libcudacxx-extended-api-memory-access-properties-access-property-shared>`, then ``ptr`` must be a generic pointer    that is a valid pointer to the *global memory* address space.

Constructor from pointer and access property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: cuda

   template <typename RuntimeProperty>
   annotated_ptr(pointer ptr, RuntimeProperty prop);

Constructs an ``annotated_ptr`` requesting the association of ``ptr`` with the property ``prop``.

**Constraints**:

-  ``RuntimeProperty`` is any of :ref:`cuda::access_property::global <libcudacxx-extended-api-memory-access-properties-access-property-global>`,
   :ref:`cuda::access_property::normal <libcudacxx-extended-api-memory-access-properties-access-property-normal>`,
   :ref:`cuda::access_property::streaming <libcudacxx-extended-api-memory-access-properties-access-property-streaming>`,
   :ref:`cuda::access_property::persisting <libcudacxx-extended-api-memory-access-properties-access-property-persisting>`, or
   :ref:`cuda::access_property <libcudacxx-extended-api-memory-access-properties-access-property>` (same as *global*).

**Preconditions**:

- ``ptr`` is a pointer to a valid allocation in the *global memory* address space.

Copy Constructor from a different ``annotated_ptr``
----------------------------------------------------

.. code:: cuda

   template <typename T, typename P>
   constexpr annotated_ptr(const annotated_ptr<T, P>& a);

Constructs an ``annotated_ptr`` for the same pointer as the input ``annotated_ptr``.

**Constraints**

-  ``annotated_ptr<Type, Property>::pointer`` is assignable from ``annotated_ptr<T, P>::pointer``.
-  ``Property`` is either ``cuda::access_property`` (*dynamic*) or ``P``.
-  ``Property`` and ``P`` specify the same memory space.

**Preconditions**

- ``pointer`` is compatible with ``Property``.

Explicit conversion operator to ``bool``
----------------------------------------

.. code:: cuda

   constexpr operator bool() const noexcept;

**Returns**: ``false`` if the pointer is a ``nullptr``, ``true`` otherwise.

Raw pointer access
------------------

.. code:: cuda

   pointer get() const noexcept;

**Returns**: A pointer derived from the ``annotated_ptr``.

Operators
---------

Dereference
~~~~~~~~~~~

.. code:: cuda

   reference operator*() const noexcept;

**Returns**: value pointed by ``annotated_ptr``.

**Preconditions**

The underlying pointer is not null.

Pointer-to-member
~~~~~~~~~~~~~~~~~

.. code:: cuda

   pointer operator->() const noexcept;

**Preconditions**

- The underlying pointer is not null.

**Returns**: underlying pointer.

Subscript
~~~~~~~~~

.. code:: cuda

   reference operator[](ptrdiff_t i) const noexcept;

**Returns**: reference to element ``i``.

**Preconditions**

- The underlying pointer plus the offset ``i`` is not null.

Pointer distance
~~~~~~~~~~~~~~~~

.. code:: cuda

   constexpr difference_type operator-(annotated_ptr p) const;

**Returns**: Difference of pointers, as-if ``get() - p.get()``.

**Preconditions**

- ``ptr >= p``.

----

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
        for (int i = g.thread_rank(); idx < N; idx += g.size()) {
            x[i] = a[i] * x[i] + b[i];
        }
    }

updates ``x``, ``y``, and ``z`` as follows:

.. code:: cuda

   update<<<grid, block>>>(x, a, b, N);
   update<<<grid, block>>>(y, a, b, N);
   update<<<grid, block>>>(z, a, b, N);

The elements of ``a`` and ``b`` are used in all kernels. If ``N`` is large enough, elements of ``a`` and ``b`` might be evicted from the L2 cache, requiring these to be re-loaded from memory in the next ``update``.

We can make the ``update`` kernel generic to allow the caller to pass ``cuda::annotated_ptr`` objects that hint at how memory will be accessed:

.. code:: cuda

    template <typename PointerX, typename PointerA, typename PointerB>
    __global__ void update_template(PointerX x, PointerA a, PointerB b, size_t N) {
        auto g = cooperative_groups::this_grid();
        for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
            x[idx] = a[idx] * x[idx] + b[idx];
        }
    }

With ``cuda::annotated_ptr``, the caller can then specify the temporal locality of the memory accesses:

.. code:: cuda

   // Frequent accesses to "a" and "b"; infrequent accesses to "x" and "y":
   cuda::annotated_ptr<const int, cuda::access_property::persisting> a_persistent{a}, b_persistent{b};
   cuda::annotated_ptr<int, cuda::access_property::streaming>        x_streaming{x}, y_streaming{y};
   update_template<<<grid, block>>>(x_streaming, a_persistent, b_persistent, N);
   update_template<<<grid, block>>>(y_streaming, a_persistent, b_persistent, N);

   // Infrequent accesses to "a" and "b"; frequent accesses to "z":
   cuda::annotated_ptr<const int, cuda::access_property::streaming> a_streaming{a}, b_streaming{b};
   cuda::annotated_ptr<int, cuda::access_property::persisting>      z_persistent{z};
   update_template<<<grid, block>>>(z_persistent, a_streaming, b_streaming, N);

   // Different kernel, "update_z", uses "z" again one last time.
   // Since "z" was accessed as "persisting" by the previous kernel,
   // parts of it are more likely to have previously survived in the L2 cache.
   update_z<<<grid, block>>>(z, ...);

Notice how the raw pointers to ``a`` and ``b`` can be wrapped by both ``annotated_ptr<T, persistent>`` and ``annotated_ptr<T, streaming>``, and accesses through each pointer applies the corresponding access property.
