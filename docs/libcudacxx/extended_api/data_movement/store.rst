.. _libcudacxx-extended-api-data-movement-store:

``store``: Store data to global memory
======================================

.. code:: cuda

    template <typename T>
    [[nodiscard]] __device__ inline
    void store(const T*                 ptr,
               /*eviction_policy_type*/ eviction_policy = eviction_none)

The function stores data to global memory and allows to specify memory access properties.

**Parameters**

- ``ptr``: Pointer to the data to be stored.
- ``eviction_policy``: The eviction policy. Requires ``SM >= 70`` and ``sizeof(T) = {1, 2, 4, 8, 16}``. See also `"Cache Eviction Priority Hints" <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id150>`_

    - ``eviction_none``: No eviction (default).
    - ``eviction_normal``: Normal eviction behavior. Currently equivent to ``eviction_none``.
    - ``eviction_unchanged``: Unchanged evict policy.
    - ``eviction_first``: First evict policy (streaming).
    - ``eviction_last``: Last evict policy (persisting).
    - ``eviction_no_alloc``: No allocation evict policy (streaming).

**Mandates**

- The functions requires a valid pointer to global memory.

Example
-------

.. code:: cuda

    #include <cuda/data_movement>

    __device__ unsigned output;

    __global__ void store_kernel() {
        auto ptr = &output;
        cuda::device::store(1, ptr);
        cuda::device::store(2, ptr, cuda::device::eviction_first);
        cuda::device::store(3, ptr, cuda::device::eviction_last);
    }

    int main() {
        store_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/fd8od9qfP>`_
