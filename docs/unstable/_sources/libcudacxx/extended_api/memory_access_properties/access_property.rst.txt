.. _libcudacxx-extended-api-memory-access-properties-access-property:

``cuda::access_property``
=========================

Defined in header ``<cuda/annotated_ptr>``.

The class ``cuda::access_property`` provides an opaque encoding for *L2 cache memory residence* control and *memory space* properties. It is used in combination with :ref:`cuda::annotated_ptr <libcudacxx-extended-api-memory-access-properties-annotated-ptr>`, :ref:`cuda::associate_access_property <libcudacxx-extended-api-memory-access-properties-associate-access-property>` and :ref:`cuda::apply_access_property <libcudacxx-extended-api-memory-access-properties-apply-access-property>` to *request* the application of properties to memory operations.

.. code:: cuda

   namespace cuda {

   class access_property {
     public:
       // Static memory space property:
       struct shared {};
       struct global {};

       // Static global memory residence control property:
       struct normal {
           __host__ __device__ constexpr operator cudaAccessProperty() const noexcept;
       };
       struct persisting {
           __host__ __device__ constexpr operator cudaAccessProperty() const noexcept;
       };
       struct streaming {
           __host__ __device__ constexpr operator cudaAccessProperty() const noexcept;
       };

       access_property() noexcept = default;

       // Constructors from static global memory residence control properties:
       __host__ __device__ constexpr access_property(global)     noexcept;
       __host__ __device__ constexpr access_property(normal)     noexcept;
       __host__ __device__ constexpr access_property(streaming)  noexcept;
       __host__ __device__ constexpr access_property(persisting) noexcept;

       // Dynamic interleaved global memory residence control property constructors:
       __host__ __device__ constexpr access_property(normal,     float probability) noexcept;
       __host__ __device__ constexpr access_property(streaming,  float probability) noexcept;
       __host__ __device__ constexpr access_property(persisting, float probability) noexcept;
       __host__ __device__ constexpr access_property(normal,     float probability, streaming) noexcept;
       __host__ __device__ constexpr access_property(persisting, float probability, streaming) noexcept;

       // Dynamic range global memory residence control property constructors:
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, normal) noexcept;
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, streaming) noexcept;
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, persisting) noexcept;
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, global,     streaming) noexcept;
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, normal,     streaming) noexcept;
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, persisting, streaming) noexcept;
       __host__ __device__ access_property(void* ptr, size_t primary_bytes, size_t total_bytes, streaming,  streaming) noexcept;
   };

   } // namespace cuda

Kinds of Access Properties
--------------------------

Access properties are either *static* compile-time values or *dynamic* runtime values. The following properties
of a memory access are provided:

*Shared Memory property*:

   .. _libcudacxx-extended-api-memory-access-properties-access-property-shared:

- ``cuda::access_property::shared``: memory access to the shared memory space.

*Global Memory properties*:

   .. _libcudacxx-extended-api-memory-access-properties-access-property-global:

- ``cuda::access_property::global``: memory access to the global memory space *without* indicating an expected frequency of access to that memory, namely the access behavior is not modified.

   .. _libcudacxx-extended-api-memory-access-properties-access-property-normal:

- ``cuda::access_property::normal``: memory access to the global memory space expecting the memory to be accessed as frequent as other memory.

   .. _libcudacxx-extended-api-memory-access-properties-access-property-persisting:

- ``cuda::access_property::persisting``: memory access to the global memory space expecting the memory to be accessed more frequently than other memory; this priority is suitable for data that should remain persistent in cache.

   .. _libcudacxx-extended-api-memory-access-properties-access-property-streaming:

- ``cuda::access_property::streaming``: memory access to the global memory space expecting the memory to be accessed infrequently; this priority is suitable for streaming data.

**Note**: The difference between ``cuda::access_property::global`` and ``cuda::access_property::normal`` is subtle.
The ``cuda::access_property::normal`` hints that the pointer points to the global address space *and* the memory will
be accessed with "normal frequency", while ``cuda::access_property::global`` only hints that the pointer points to
the global address-space, it does not hint about how frequent the accesses will be.

.. warning::

   The behavior of *requesting* the application of ``cuda::access_property`` to memory accesses, or their association
   with memory addresses, outside of the corresponding address space is *undefined*
   (note: even if that address is not used). The correctness of the input pointer and memory properties are verified in debug mode.

----

Global Memory Property Definition
---------------------------------

The L2 residence control can be specified in two ways:

- **Interleaved**: A memory address is accessed with a property with a given ``probability``, while the remaining ``1 - probability`` accesses are performed with a second one.
- **Range**: The first ``primary_bytes`` of a memory address is accessed with one property and the remaining ``total_bytes - primary_bytes`` addresses with a second one.

Default constructor
-------------------

.. code:: cuda

   access_property() noexcept = default;

**Effects**: as if ``access_property(global)`` (unchanged).

Static global memory residence control property constructors
------------------------------------------------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::access_property(global) noexcept;
   __host__ __device__ constexpr access_property::access_property(normal) noexcept;
   __host__ __device__ constexpr access_property::access_property(streaming) noexcept;
   __host__ __device__ constexpr access_property::access_property(persisting) noexcept;

**Effects**: as if ``access_property(PROPERTY, 1.0)`` where ``PROPERTY`` is one of ``global`` (unchanged), ``normal``, ``streaming``, or ``persisting``.

Dynamic interleaved global memory residence control property constructors
-------------------------------------------------------------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::access_property(normal,     float probability) noexcept;
   __host__ __device__ constexpr access_property::access_property(streaming,  float probability) noexcept;
   __host__ __device__ constexpr access_property::access_property(persisting, float probability) noexcept;
   __host__ __device__ constexpr access_property::access_property(normal,     float probability, streaming) noexcept;
   __host__ __device__ constexpr access_property::access_property(persisting, float probability, streaming) noexcept;

**Preconditions**: ``0 < probability <= 1.0``.

**Effects**: constructs an *interleaved* access property that *requests* the first and third arguments - access properties - to be applied with ``probability`` and ``1 - probability`` to memory accesses. The overloads without a third argument request applying ``global`` (unchanged) with ``1 - probability``.

Dynamic range global memory residence control property constructors
-------------------------------------------------------------------

.. code:: cuda

   __host__ __device__ access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, normal) noexcept;
   __host__ __device__ access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, streaming) noexcept;
   __host__ __device__ access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, persisting) noexcept;
   __host__ __device__ access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, normal,     streaming) noexcept;
   __host__ __device__ access_property::access_property(void* ptr, size_t leading_bytes, size_t total_bytes, persisting, streaming) noexcept;

..

   note: pointer arithmetic below performed ``char* ptr`` instead of
   ``void* ptr``

**Preconditions**:

   - ``ptr`` is a generic pointer that is *valid* to cast to a pointer to the global memory address space.
   - ``0 < leading_bytes <= total_bytes <= 4GB``.

**Postconditions**: memory accesses requesting the application of this property must be in range ``[ptr, ptr + total_bytes)``.

**Effects**: the fourth and fifth arguments, access properties, are called *primary* and *secondary* properties. The overloads without a fifth argument use ``global`` as the *secondary* property. Constructs a *range* access property *requesting* the properties to be **approximately** applied to memory accesses as follows:

-  *primary property* to accesses in address-range:   ``[ptr, ptr + leading_bytes)``
-  *secondary property* to accesses in address-range: ``[ptr + leading_bytes, ptr + total_bytes)``

**Note**: This property enables two main use cases:

1. Unary range ``[ptr, ptr + total_bytes)`` with *primary property* by using ``leading_bytes == total_bytes``.

2. Binary range ``[ptr, ptr + leading_bytes)`` and ``[ptr + leading_bytes, ptr + total_bytes)`` with *primary* and
   *secondary properties* respectively.

Conversion operators
--------------------

.. code:: cuda

   __host__ __device__ constexpr access_property::normal::operator     cudaAccessProperty() const noexcept;
   __host__ __device__ constexpr access_property::streaming::operator  cudaAccessProperty() const noexcept;
   __host__ __device__ constexpr access_property::persisting::operator cudaAccessProperty() const noexcept;

Allows ``constexpr cuda::access_property::normal{}``, ``cuda::access_property::streaming{}``, and ``cuda::access_property::persisting{}`` to be used in lieu of the corresponding CUDA Runtime `cudaAccessProperty <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g4991a8bc9c2356a8da28d093a1da6758>`_. See also `L2 Policy for Persisting Accesses <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#l2-policy-for-persisting-accesses>`_.

Example
-------

.. code:: cuda

   #include <cuda/access_property>

   __global__ void kernel(int* global_ptr, size_t num_bytes) {
       __shared__ int smem;
       cuda::access_property shared_prop{&smem, cuda::access_property::shared{}};
       cuda::access_property streaming_prop{global_ptr, sizeof(int), sizeof(int), cuda::access_property::streaming{}};
       cuda::access_property streaming_interleaved_prop{cuda::access_property::streaming{}, 1.0};
       cuda::access_property persisting_prop{global_ptr, num_bytes, num_bytes, cuda::access_property::persisting{});
   }

   __global__ void undefined_behavior(int* global_ptr) { // verified in debug mode
       __shared__ int smem;
       // Associating pointers with mismatching address spaces is undefined:
       cuda::access_property{global_ptr, cuda::access_property::shared{}}; // undefined behavior
       cuda::access_property{&smem, cuda::access_property::normal{}};      // undefined behavior
       cuda::access_property{&smem, cuda::access_property::streaming{}};   // undefined behavior
       cuda::access_property{&smem, cuda::access_property::persisting{}};  // undefined behavior

       // Using a zero probability or probability out-of-range (0, 1] is undefined:
       cuda::access_property{cuda::access_property::streaming{}, 0.0f};    // undefined behavior
       cuda::access_property{cuda::access_property::streaming{}, 2.0f};    // undefined behavior

       // Providing size values out-of-range is undefined:
       cuda::access_property{global_ptr, 0, 0, cuda::access_property::streaming{}, 0.0f}; // undefined behavior
       cuda::access_property{global_ptr, 8, 4, cuda::access_property::streaming{}, 2.0f}; // undefined behavior
   }
