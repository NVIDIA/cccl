.. _cccl-runtime-buffer:

Buffer
======

The buffer API provides a typed container allocated from memory resources. It handles stream-ordered allocation, initialization, and deallocation of memory.

``cuda::buffer``
----------------
.. _cccl-runtime-buffer-buffer:

``cuda::buffer`` is a container that provides typed storage allocated from a given :ref:`memory resource <libcudacxx-extended-api-memory-resources-resource>`. It handles alignment and release of allocations. The elements are initialized during construction, which may require a kernel launch.

In addition to being typed, ``buffer`` also takes a set of :ref:`properties <libcudacxx-extended-api-memory-resources-properties>` to ensure that memory accessibility and other constraints are checked at compile time.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/buffer>
   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_buffer(cuda::stream_ref stream) {
     // Create a device buffer
     auto mr = cuda::device_default_memory_pool(cuda::devices[0]);
     auto buf = cuda::make_buffer<float>(
       stream,
       mr,
       1024,  // size
       0.0f   // value
     );

     // Use buffer...

     // Buffer is automatically deallocated when destroyed
   }

Type Aliases
------------
.. _cccl-runtime-buffer-type-aliases:

Convenience type aliases are provided for common buffer types:

- ``cuda::device_buffer<T>`` - Buffer with ``device_accessible`` property
- ``cuda::host_buffer<T>`` - Buffer with ``host_accessible`` property

Example:

.. code:: cpp

   #include <cuda/buffer>
   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <cuda/stream>

   void use_buffers(cuda::stream_ref stream) {
    auto device_mr = cuda::device_default_memory_pool(cuda::devices[0]);
    auto host_mr = cuda::pinned_default_memory_pool();

     cuda::device_buffer<int> dev_buf{stream, device_mr, 1000};
     cuda::host_buffer<int> host_buf{stream, host_mr, 1000};
   }

Construction
------------
.. _cccl-runtime-buffer-construction:

Buffers can be constructed in several ways, for different ways to initialize the memory:

- Empty buffer: ``buffer(stream, resource)``
- With size (uninitialized): ``buffer(stream, resource, size, no_init)``
- With size and value: ``buffer(stream, resource, size, value)``
- From iterator range: ``buffer(stream, resource, first, last)``
- From initializer list: ``buffer(stream, resource, {val1, val2, ...})``
- From range: ``buffer(stream, resource, range)``

In each case the memory is allocated and initialized in stream order on the provided stream.

Example:

.. code:: cpp

   #include <cuda/buffer>
   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <vector>

   void construct_buffers(cuda::stream_ref stream) {
    auto mr = cuda::device_default_memory_pool(cuda::devices[0]);

     // Empty buffer
     cuda::device_buffer<int> buf1{stream, mr};

     // Uninitialized buffer
     cuda::device_buffer<int> buf2{stream, mr, 1000, cuda::no_init};

     // Initialized with value
     cuda::device_buffer<int> buf3{stream, mr, 1000, 42};

     // From iterator range
     std::vector<int> vec{1, 2, 3, 4, 5};
     cuda::device_buffer<int> buf4{stream, mr, vec.begin(), vec.end()};

     // From initializer list
     cuda::device_buffer<int> buf5{stream, mr, {1, 2, 3, 4, 5}};
   }

Stored Stream Management and Deallocation
-----------------------------------------
.. _cccl-runtime-buffer-stream-management:

Buffers store the stream they were constructed with, and can have their stream changed:

- ``stream()`` - Get the associated stream
- ``set_stream(new_stream)`` - Change the associated stream (synchronizes with old stream)

When the buffer is destroyed, the memory is deallocated using the stored stream.
Buffer can also be explicitly destroyed with ``destroy()`` or ``destroy(stream_ref)``, which will deallocate the memory using the provided stream.

Example:

.. code:: cpp

   #include <cuda/buffer>
   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <cuda/stream>

   void manage_stream_and_deallocate() {
    cuda::stream stream1{};
    cuda::stream stream2{};
    auto mr = cuda::device_default_memory_pool(cuda::devices[0]);

    // Allocate on stream1
    cuda::device_buffer<int> buf{stream1, mr, 1024, cuda::no_init};

    // Switch to stream2 (synchronizes with stream1)
    buf.set_stream(stream2);

    // Explicit deallocation on the stored stream (stream2)
    // Alternative would be to call buf.destroy(stream2)
    buf.destroy();
   }

``cuda::make_buffer``
---------------------
.. _cccl-runtime-buffer-make-buffer:

``cuda::make_buffer()`` is a factory function that creates buffers with automatic property deduction from the memory resource. It supports the same construction patterns as the buffer constructors.

Example:

.. code:: cpp

   #include <cuda/buffer>
   #include <cuda/devices>
   #include <cuda/memory_resource>

   void make_buffers(cuda::stream_ref stream) {
    auto mr = cuda::device_default_memory_pool(cuda::devices[0]);

     // Properties are automatically deduced from the memory resource
     auto buf = cuda::make_buffer<float>(stream, mr, 1024);
   }

Iterators and Access
--------------------
.. _cccl-runtime-buffer-iterators:

Buffers provide standard container-like iterators and access methods:

- ``begin()`` / ``end()`` - Iterator access
- ``cbegin()`` / ``cend()`` - Const iterator access
- ``rbegin()`` / ``rend()`` - Reverse iterator access
- ``data()`` - Pointer to underlying data
- ``size()`` - Number of elements
- ``empty()`` - Check if buffer is empty
- ``get_unsynchronized(n)`` - Access element without synchronization, instead of using ``operator[]``

Example:

.. code:: cpp

   #include <cuda/buffer>
   #include <cuda/devices>
   #include <cuda/memory_resource>
   #include <cuda/std/cstddef>
   #include <algorithm>

   void iterate_buffer(cuda::stream_ref stream) {
    auto mr = cuda::pinned_default_memory_pool();
     cuda::host_buffer<int> buf{stream, mr, {1, 2, 3, 4, 5}};

     // Unsynchronized element access by index
     for (cuda::std::size_t i = 0; i < buf.size(); ++i) {
      buf.get_unsynchronized(i) += 1;
     }

     // Use with algorithms
     auto it = std::find(buf.begin(), buf.end(), 3);
   }


Memory Resource Access
----------------------
.. _cccl-runtime-buffer-memory-resource:

Buffers provide access to their underlying memory resource:

- ``memory_resource()`` - Get a const reference to the memory resource

