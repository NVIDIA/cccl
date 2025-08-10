.. _libcudacxx-extended-api-memory-is_address_from:

``cuda::device::is_address_from`` and ``cuda::device::is_object_from``
======================================================================

Defined in ``<cuda/memory>`` header.

.. code:: cpp

   enum class address_space
   {
     global,         // Global state space
     shared,         // Shared state space
     constant,       // Constant state space
     local,          // Local state space
     grid_constant,  // Kernel function parameter in the parameter state space
     cluster_shared, // Cluster shared window within the shared state space
   };

Enumeration of device address spaces to be used with the ``is_address_from()`` and ``is_object_from()`` functions. See `PTX ISA documentation for state spaces <https://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces>`_ for more details.

.. code:: cpp

   [[nodiscard]] __device__ inline
   bool is_address_from(const volatile void* ptr, address_space space) noexcept; // (1)

The function checks if a pointer ``ptr`` with a generic address is from a ``space`` address state space.

.. code:: cpp

   template <typename T>
   [[nodiscard]] __device__ inline
   bool is_object_from(T& obj, address_space space) noexcept; // (2)

The function checks if an object ``obj`` with a generic address is from a ``space`` address state space.

Compared to the corresponding CUDA C functions ``__isGlobal()``, ``__isShared()``, ``__isConstant()``, ``__isLocal()``, ``__isGridConstant()``, or ``__isClusterShared()``, ``is_address_from()`` is portable across any compute capability and checks that the pointer is not a null in debug mode.

**Parameters**

- ``ptr``: The pointer. (1)
- ``obj``: The object. (2)
- ``space``: The address space. (1, 2)

**Return value**

- ``true`` if the pointer/object is from the specified address space, ``false`` otherwise.

*Note: If the device architecture doesn't support the requested address space, the function will always return ``false``.*

**Preconditions**

- ``ptr`` is not a null pointer. (1)

**Performance considerations**

- If possible, the ``__isGlobal``, ``__isShared``, ``__isConstant``, ``__isLocal``, ``__isGridConstant``, or ``__isClusterShared`` built-in functions are used to determine the address space.

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

`See it on Godbolt 🔗 <https://godbolt.org/z/5ajhe37df>`_
