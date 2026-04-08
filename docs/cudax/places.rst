.. _cudax-places:

Places
======

.. contents::
   :depth: 2

Places are abstractions that represent where code executes and where data
resides across the non-uniform memory of a CUDA system. They provide a
unified interface for managing execution affinity, stream pools, memory
allocation, and device context switching -- independently of any task-based
programming model.

Places come in two flavors:

- **Execution places** (``exec_place``) determine where code is executed.
- **Data places** (``data_place``) specify where data is located in memory.

The places API is part of the ``cuda::experimental::places`` C++ namespace
and can be used standalone via the ``cuda/experimental/places.cuh`` header,
without pulling in the full CUDASTF task-graph framework. For backward
compatibility, all places types are also available in the
``cuda::experimental::stf`` namespace.

.. _places-execution-places:

Execution places
----------------

An *execution place* describes a location where computation can occur.
The following factory methods create the most common execution places:

- ``exec_place::device(id)`` -- a specific CUDA device
- ``exec_place::host()`` -- the host CPU
- ``exec_place::current_device()`` -- the CUDA device that is currently active

When an execution place is activated, it sets the appropriate CUDA context
(e.g. calls ``cudaSetDevice``). Each execution place also has an *affine*
data place: the memory location naturally associated with it. For a device
execution place the affine data place is the device's global memory; for
the host it is pinned host memory (RAM).

.. _places-data-places:

Data places
-----------

A *data place* describes a memory location where data can reside. The
following factory methods are available:

- ``data_place::device(id)`` -- global memory of a specific CUDA device
- ``data_place::host()`` -- pinned host memory
- ``data_place::managed()`` -- CUDA managed (unified) memory
- ``data_place::affine()`` -- the data place naturally associated with the
  current execution place

The *affine* data place is the default: when no data place is specified,
data is placed in the memory that is local to the execution place. For
example, a task running on device 0 will access data in device 0's global
memory by default.

Non-affine placement is also supported: data can be placed on a different
device or in host memory regardless of where the computation runs. This is
useful for sparse accesses (leveraging CUDA Unified Memory page faulting)
or for addressing memory capacity constraints. Non-affine placement assumes
the hardware and OS support such accesses (NVLINK, UVM, etc.).

.. _places-container-keys:

Places as container keys
------------------------

Both ``exec_place`` and ``data_place`` can be used as keys in standard
associative containers. The library provides the required comparison and
hash support:

- **``std::map``** and **``std::set``** use ``operator<`` (strict weak
  ordering) for keys. Both place types implement ``operator<``, so they
  can be used as ordered map or set keys.

- **``std::unordered_map``** and **``std::unordered_set``** require a
  hash function and equality. The library specializes ``cuda::experimental::stf::hash``
  for both place types (accessible from both the ``stf`` and ``places`` namespaces),
  and both implement ``operator==``.

This allows, for example, maintaining per-place handles (e.g. CUBLAS or
CUSOLVER handles keyed by ``exec_place``) or per-place caches keyed by
``data_place``, using either ordered or hash-based containers as needed.
The following snippet shows lazy creation of a CUBLAS handle per execution
place using an ``std::unordered_map`` keyed by ``exec_place``:

.. code:: c++

   #include <cuda/experimental/places.cuh>
   #include <cublas_v2.h>

   using namespace cuda::experimental::places;

   cublasHandle_t& get_cublas_handle(const exec_place& ep = exec_place::current_device())
   {
     static std::unordered_map<exec_place, cublasHandle_t, hash<exec_place>> handles;
     auto& h = handles[ep];
     if (h == cublasHandle_t{})
     {
       exec_place_scope scope(ep);
       cuda_safe_call(cublasCreate(&h));
     }
     return h;
   }

.. _places-activate:

Setting the current device or context
--------------------------------------

The ``exec_place::activate()`` method provides a generic alternative to
``cudaSetDevice()`` that works uniformly across different execution place types.
This is useful when you want to set the current CUDA device or context without
using tasks.

The method returns an ``exec_place`` representing the previous state, which can
be used to restore the original device or context.

**Behavior by execution place type:**

- **Device places** (``exec_place::device(id)``): Calls ``cudaSetDevice(id)``
- **Green context places**: Sets the current CUDA driver context via ``cuCtxSetCurrent()``
- **Host places**: No-op

**Basic usage with devices:**

.. code:: cpp

    exec_place place = exec_place::device(1);
    exec_place prev = place.activate();  // Switch to device 1

    // ... perform operations on device 1 ...

    place.deactivate(prev);  // Restore previous device

**Alternative restoration pattern:**

You can also restore by calling ``activate()`` on the returned place:

.. code:: cpp

    exec_place place = exec_place::device(1);
    exec_place prev = place.activate();

    // ... work on device 1 ...

    prev.activate();  // Equivalent to place.deactivate(prev)

**Usage with green contexts (CUDA 12.4+):**

Green contexts provide SM-level partitioning of GPU resources. The
``activate()``/``deactivate()`` methods handle the underlying driver context
management:

.. code:: cpp

    // Create green contexts with 8 SMs each
    green_context_helper gc(8, device_id);
    auto view = gc.get_view(0);

    exec_place gc_place = exec_place::green_ctx(view);
    exec_place prev = gc_place.activate();  // Sets green context as current

    // ... GPU work runs with SM affinity ...

    gc_place.deactivate(prev);  // Restore original context

**RAII scope for scoped activation:**

For exception-safe code or when you want automatic restoration, use the
``exec_place_scope`` RAII helper:

.. code:: cpp

    {
        exec_place_scope scope(exec_place::device(1));
        // Device 1 is now active
        // ... perform operations on device 1 ...
    }
    // Previous device is automatically restored when scope goes out of scope

The guard automatically restores the previous execution place when it goes out
of scope, making it useful for exception-safe code.

.. _places-stream-management:

Stream management with execution places
----------------------------------------

Execution places can be used independently of any task system to manage CUDA
streams in a structured way. This is useful when you want to use place
abstractions (devices, green contexts) for stream management without the full
task-based programming model.

Each execution place owns a pool of CUDA streams. The
``exec_place::pick_stream`` method returns a CUDA stream from that pool.

The method accepts an optional ``for_computation`` hint (defaults to ``true``)
that may select between computation and data transfer stream pools to improve
overlapping. This is purely a performance hint, and it does not affect
correctness. Not all execution places enforce it.

.. code:: cpp

    #include <cuda/experimental/places.cuh>
    using namespace cuda::experimental::places;

    // Get a stream from the current device
    exec_place place = exec_place::current_device();
    cudaStream_t stream = place.pick_stream();

    // Use the stream for CUDA operations
    myKernel<<<grid, block, 0, stream>>>(d_data);

    // Get streams from specific devices
    cudaStream_t stream_dev0 = exec_place::device(0).pick_stream();
    cudaStream_t stream_dev1 = exec_place::device(1).pick_stream();

Stream pools are populated lazily -- CUDA streams are only created when first
requested via ``pick_stream()``.

.. _places-memory-allocation:

Memory allocation with data places
------------------------------------

Data places provide a unified interface for memory allocation that works across
different memory types (host, device, managed) and place extensions (green
contexts, user-defined places). This allows you to allocate memory while
benefiting from the place abstraction.

The ``data_place::allocate()`` and ``data_place::deallocate()`` methods provide
raw memory allocation. The stream parameter defaults to ``nullptr``, which is
convenient for non-stream-ordered allocations (host, managed) where the stream
is ignored:

.. code:: cpp

    #include <cuda/experimental/places.cuh>
    using namespace cuda::experimental::places;

    // Allocate on host (pinned memory) - stream defaults to nullptr
    void* host_ptr = data_place::host().allocate(1024);
    // ... use host_ptr ...
    data_place::host().deallocate(host_ptr, 1024);

    // Allocate on a specific device (stream-ordered)
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    void* dev_ptr = data_place::device(0).allocate(1024, stream);
    // ... use dev_ptr with stream ...
    data_place::device(0).deallocate(dev_ptr, 1024, stream);
    cudaStreamDestroy(stream);

    // Allocate managed memory - stream defaults to nullptr
    void* managed_ptr = data_place::managed().allocate(1024);
    // ... use managed_ptr from host or device ...
    data_place::managed().deallocate(managed_ptr, 1024);

**Stream-ordered vs immediate allocations:**

Different data places have different allocation behaviors:

- **Host** (``data_place::host()``): Uses ``cudaMallocHost()`` / ``cudaFreeHost()`` - immediate, stream parameter is ignored
- **Managed** (``data_place::managed()``): Uses ``cudaMallocManaged()`` / ``cudaFree()`` - immediate, stream parameter is ignored (note: ``cudaFree`` may introduce implicit synchronization)
- **Device** (``data_place::device(id)``): Uses ``cudaMallocAsync()`` / ``cudaFreeAsync()`` - stream-ordered
- **Extensions** (green contexts, etc.): Behavior depends on the extension implementation

You can query whether a place uses stream-ordered allocation with
``allocation_is_stream_ordered()``:

.. code:: cpp

    data_place place = data_place::device(0);
    if (place.allocation_is_stream_ordered()) {
        // Allocation is stream-ordered - synchronize via the stream
        void* ptr = place.allocate(size, stream);
        myKernel<<<grid, block, 0, stream>>>(ptr);
        place.deallocate(ptr, size, stream);
        cudaStreamSynchronize(stream);
    } else {
        // Allocation is immediate - stream is ignored, safe to use right away
        void* ptr = place.allocate(size);
        // ... use ptr ...
        place.deallocate(ptr, size);
    }

This abstraction is particularly useful when writing generic code that needs to
work with different types of places, including custom place extensions.

.. _places-vmm:

VMM-based allocation with mem_create
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For advanced use cases involving CUDA's Virtual Memory Management (VMM) API,
``data_place`` also provides the ``mem_create()`` method. This is a lower-level
interface used internally by localized arrays (``composite_slice``) to create
physical memory segments that are then mapped into a contiguous virtual address
space.

Unlike ``allocate()``, which returns a usable pointer directly, ``mem_create()``
returns a ``CUmemGenericAllocationHandle`` that must be subsequently mapped with
``cuMemMap()`` before use:

.. code:: cpp

    #include <cuda/experimental/places.cuh>
    using namespace cuda::experimental::places;

    // Create a physical memory handle for device 0
    CUmemGenericAllocationHandle handle;
    data_place::device(0).mem_create(&handle, size);

    // The handle must be mapped to a virtual address before use
    // (see CUDA VMM documentation for cuMemMap, cuMemSetAccess, etc.)

**When to use each method:**

- Use ``allocate()`` for most cases - it provides ready-to-use memory with
  stream-ordered semantics where applicable.

- Use ``mem_create()`` only when you need explicit control over virtual memory
  mapping, such as creating localized arrays that span multiple devices with a
  unified virtual address space.

**Limitations of mem_create:**

- Only supports device memory and host memory (pinned)
- Managed memory is **not supported** by the VMM API
- The returned handle requires additional VMM API calls to be usable

Custom place extensions can override ``mem_create()`` to provide specialized
VMM allocation behavior (e.g., memory localization for hardware partitions).

.. _places-grid:

Grid of places
--------------

It is possible to manipulate places which are a collection of multiple places.
In particular, it is possible to define an execution place which corresponds
to multiple device execution places.

Creating grids of places
^^^^^^^^^^^^^^^^^^^^^^^^

A grid of execution places is an ``exec_place`` that contains multiple
underlying places. Grids are created with the ``make_grid`` free function:

.. code:: c++

   // Create a 1D grid from a vector of places
   exec_place grid = make_grid(std::vector<exec_place>{
       exec_place::device(0), exec_place::device(1)
   });

The ``exec_place::all_devices()`` helper creates a grid of all available
CUDA devices:

.. code:: c++

   exec_place all = exec_place::all_devices();

Similarly, ``exec_place::n_devices(n)`` creates a grid from the first ``n``
devices:

.. code:: c++

   exec_place first_four = exec_place::n_devices(4);

It is possible to retrieve the total number of elements in a grid using
the ``size()`` method, and individual places with ``get_place(i)``:

.. code:: c++

   exec_place grid = exec_place::all_devices();
   for (size_t i = 0; i < grid.size(); i++) {
       exec_place dev = grid.get_place(i);
       // ...
   }

Shaped grids
^^^^^^^^^^^^

Grids of places need not be 1D arrays. They can be structured as a
multi-dimensional grid described with a ``dim4`` class by passing it to
``make_grid`` or ``n_devices``:

.. code:: c++

   // Create a shaped grid: 8 devices arranged as a 2x2x2 cube
   exec_place cube = exec_place::n_devices(8, dim4(2, 2, 2));

   // Or from an explicit vector
   exec_place shaped = make_grid(my_places, dim4(4, 2));

Note that the total size of the ``dim4`` must match the number of places.

It is possible to query the *shape* of the grid using ``get_dims()``,
which returns a ``dim4`` object. Individual places can be accessed by
multi-dimensional position using ``get_place(pos4)``.

.. _places-partitioning:

Partitioning grids
^^^^^^^^^^^^^^^^^^

The ``place_partition`` class partitions an execution place at a given
granularity. This is useful for splitting a multi-device grid into its
constituent devices, or for partitioning a device into green contexts or
CUDA streams.

The partitioning granularity is specified by ``place_partition_scope``:

- ``place_partition_scope::cuda_device`` -- partition into individual devices
- ``place_partition_scope::green_context`` -- partition into green contexts (CUDA 12.4+)
- ``place_partition_scope::cuda_stream`` -- partition into CUDA streams

.. code:: c++

   exec_place grid = exec_place::all_devices();

   // Partition into individual devices
   place_partition devices(grid, place_partition_scope::cuda_device);
   for (auto& dev : devices) {
       // dev is an exec_place for a single device
   }

   // Convert back to an exec_place grid
   exec_place new_grid = devices.to_exec_place();

The ``exec_place::partition_by_scope()`` method provides a shorthand that
returns a new ``exec_place`` grid directly:

.. code:: c++

   exec_place grid = exec_place::all_devices();
   exec_place by_device = grid.partition_by_scope(place_partition_scope::cuda_device);

.. _places-data-partitioning:

Data partitioning policies
^^^^^^^^^^^^^^^^^^^^^^^^^^

When using a grid of places with CUDASTF constructs such as ``parallel_for``,
*data partitioning policies* express how data and index spaces are dispatched
over the different places of a grid.

.. code:: c++

   class MyPartition : public partitioner_base {
   public:
       template <typename S_out, typename S_in>
       static const S_out apply(const S_in& in, pos4 position, dim4 grid_dims);

       pos4 get_executor(pos4 data_coords, dim4 data_dims, dim4 grid_dims);
   };

A partitioning class must implement an ``apply`` method which takes:

- a reference to a shape of type ``S_in``
- a position within a grid of execution places, described using an object of
  type ``pos4``
- the dimension of this grid expressed as a ``dim4`` object

``apply`` returns a shape which corresponds to the subset of the ``in``
shape associated to this entry of the grid. Note that the output shape
type ``S_out`` may be different from the ``S_in`` type of the input
shape.

To support different types of shapes, appropriate overloads of the
``apply`` method should be implemented.

This ``apply`` method is typically used by the ``parallel_for``
construct in order to dispatch indices over the different places.

A partitioning class must also implement the ``get_executor`` virtual
method which allows localized data allocators. This
method indicates, for each entry of a shape, on which place this entry
should *preferably* be allocated.

``get_executor`` returns a ``pos4`` coordinate in the execution place
grid, and its arguments are:

- a coordinate within the shape described as a ``pos4`` object
- the dimension of the shape expressed as a ``dim4`` object
- the dimension of the execution place grid expressed as a ``dim4`` object

Defining the ``get_executor`` makes it possible to map a piece of data
over an execution place grid. The ``get_executor`` method of a partitioning
policy in an execution place grid therefore defines the *affine data
place* of a logical data accessed on that grid.

Predefined partitioning policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are currently two policies readily available:

- ``tiled_partition<TILE_SIZE>`` dispatches entries of a shape using a
  *tiled* layout. For multi-dimensional shapes, the outermost dimension is
  dispatched into contiguous tiles of size ``TILE_SIZE``.
- ``blocked_partition`` dispatches entries of the shape using a *blocked*
  layout, where each entry of the grid of places receives approximately
  the same contiguous portion of the shape, dispatched along the outermost
  dimension.

This illustrates how a 2D shape is dispatched over 3 places using the
blocked layout:

.. code:: text

    __________________________________
   |           |           |         |
   |           |           |         |
   |           |           |         |
   |    P 0    |    P 1    |   P 2   |
   |           |           |         |
   |           |           |         |
   |___________|___________|_________|

This illustrates how a 2D shape is dispatched over 3 places using a
tiled layout, where the dimension of the tiles is indicated by the
``TILE_SIZE`` parameter:

.. code:: text

    ________________________________
   |     |     |     |     |     |  |
   |     |     |     |     |     |  |
   |     |     |     |     |     |  |
   | P 0 | P 1 | P 2 | P 0 | P 1 |P2|
   |     |     |     |     |     |  |
   |     |     |     |     |     |  |
   |_____|_____|_____|_____|_____|__|
