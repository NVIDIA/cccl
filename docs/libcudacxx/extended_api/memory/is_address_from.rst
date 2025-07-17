.. _libcudacxx-extended-api-memory-is_address_from:

``cuda::device::is_address_from``
=================================

.. code:: cuda

   enum class address_space
   {
     global,
     shared,
     constant,
     local,
     grid_constant,
     cluster_shared,
   };

   template <typename T>
   [[nodiscard]] __device__ inline
   bool is_address_from(address_space space, const void* ptr)

The function checks if a pointer ``ptr`` with a generic address is from a ``space`` address state space.

**Parameters**

- ``space``: The address space.
- ``ptr``: The pointer.

**Return value**

- ``true`` if the pointer is from the specified address space, ``false`` otherwise.

**Performance considerations**

- If possible, the ``__isGlobal``, ``__isShared``, ``__isConstant``, ``__isLocal``, ``__isGridConstant``, or ``__isClusterShared`` built-in functions are used to determine the address space.

Example
-------

.. code:: cuda

    #include <cuda/memory>

    struct MutableStruct
    {
        mutable int v;
    };

    __device__ int global_var;
    __constant__ int constant_var;

    __global__ void kernel(const __grid_constant__ MutableStruct grid_constant_var)
    {
        __shared__ int shared_var;
        int local_var{};

        assert(cuda::device::is_address_from(cuda::device::address_space::global, &global_var));
        assert(cuda::device::is_address_from(cuda::device::address_space::shared, &shared_var));
        assert(cuda::device::is_address_from(cuda::device::address_space::constant, &constant_var));
        assert(cuda::device::is_address_from(cuda::device::address_space::local, &local_var));
        assert(cuda::device::is_address_from(cuda::device::address_space::grid_constant, &grid_constant_var));
    }

    int main(int, char**)
    {
        kernel<<<1, 1>>>(MutableStruct{42});
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/r1qb31szz>`_
