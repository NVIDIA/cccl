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

   [[nodiscard]] __device__ inline
   bool is_address_from(address_space space, const void* ptr)

The function checks if a pointer ``ptr`` with a generic address is from a ``space`` address state space.

Compared to the corresponding CUDA C functions ``__isGlobal()``, ``__isShared()``, ``__isConstant()``, ``__isLocal()``, ``__isGridConstant()``, or ``__isClusterShared()``, ``is_address_from()`` is portable across any compute capability and checks that the pointer is not a null in debug mode.

**Parameters**

- ``ptr``: The pointer.
- ``space``: The address space.

**Return value**

- ``true`` if the pointer is from the specified address space, ``false`` otherwise.

**Preconditions**

- ``ptr`` is not a null pointer.

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
    }

    int main(int, char**)
    {
        kernel<<<1, 1>>>(42);
        cudaDeviceSynchronize();
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/fYWjbYqzo>`_
