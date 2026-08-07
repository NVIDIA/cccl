.. _libcudacxx-extended-api-memory-is_address_from:

``cuda::device::is_address_from`` and ``cuda::device::is_object_from``
======================================================================

Defined in the ``<cuda/memory>`` header.

.. code:: cuda

   namespace cuda::device {

   enum class address_space
   {
     global,         // Global state space
     shared,         // Shared state space
     constant,       // Constant state space
     local,          // Local state space
     grid_constant,  // Kernel function parameter in the parameter state space
     cluster_shared, // Cluster shared window within the shared state space
   };

   } // namespace cuda::device

Enumeration of device address spaces used with the ``is_address_from()`` and ``is_object_from()`` functions. See the `PTX ISA documentation for state spaces <https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces>`_ for more details.

----

.. code:: cuda

   namespace cuda::device {

   [[nodiscard]] __device__ inline
   bool is_address_from(const volatile void* ptr, address_space space) noexcept; // (1)

   } // namespace cuda::device

Checks whether a generic-address pointer ``ptr`` is from the specified address space.

----

.. code:: cuda

   namespace cuda::device {

   template <typename T>
   [[nodiscard]] __device__ inline
   bool is_object_from(T& obj, address_space space) noexcept; // (2)

   } // namespace cuda::device

Checks whether an object ``obj`` with a generic address is from the specified address space.

----

Unlike the corresponding CUDA intrinsic functions ``__isGlobal()``, ``__isShared()``, ``__isConstant()``, ``__isLocal()``, ``__isGridConstant()``, and ``__isClusterShared()``, ``is_address_from()`` and ``is_object_from()`` are portable across all compute capabilities and, in debug mode, also checks that the pointer is not null.

**Parameters**

- ``ptr``: The pointer. (1)
- ``obj``: The object. (2)
- ``space``: The address space. (1, 2)

**Return value**

- Returns ``true`` if the pointer (1) or object (2) is from the specified address space; ``false`` otherwise.

.. note::

  If the GPU architecture does not support the requested address space, the function always returns ``false``.

**Preconditions**

- ``ptr`` must not be null. (1)

**Performance considerations**

- When available, the built-in functions (``__isGlobal()``, ``__isShared()``, ``__isConstant()``, ``__isLocal()``, ``__isGridConstant()``, or ``__isClusterShared()``) are used to determine the address space.
- If the memory space of the input pointer matches the requested address space,
  the function marks the pointer as belonging to that address space. For example, a subsequent store to a generic address that maps to shared memory emits an ``STS`` SASS instruction rather than the generic ``ST`` instruction.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    __device__   int global_var;
    __constant__ int constant_var;

    __global__ void kernel(const __grid_constant__ int grid_constant_var)
    {
        using cuda::device::address_space;
        __shared__ int shared_var;
        int local_var{};

        assert(cuda::device::is_address_from(&global_var, address_space::global));
        assert(cuda::device::is_address_from(&shared_var, address_space::shared));
        assert(cuda::device::is_address_from(&constant_var, address_space::constant));
        assert(cuda::device::is_address_from(&local_var, address_space::local));
        assert(cuda::device::is_address_from(&grid_constant_var, address_space::grid_constant));

        assert(cuda::device::is_object_from(global_var, address_space::global));
        assert(cuda::device::is_object_from(shared_var, address_space::shared));
        assert(cuda::device::is_object_from(constant_var, address_space::constant));
        assert(cuda::device::is_object_from(local_var, address_space::local));
        assert(cuda::device::is_object_from(grid_constant_var, address_space::grid_constant));
    }

    int main(int, char**)
    {
        kernel<<<1, 1>>>(42);
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/5ajhe37df>`__
