.. _libcudacxx-extended-api-data-movement-load:

``load``: Load data from global memory
======================================

.. code:: cuda

    template <typename T>
    [[nodiscard]] __device__ inline
    T load(const T*                  ptr,
           /*memory_access_type*/    memory_access   = read_write,
           /*eviction_policy_type*/  eviction_policy = eviction_none,
           /*prefetch_spatial_type*/ prefetch        = prefetch_spatial_none)

The function loads data from global memory and allows to specify memory access properties.

**Parameters**

- ``ptr``: Pointer to the data to be loaded.
- ``memory_access``:  The memory access property.

    - ``read_write``: Read and write access (default).
    - ``read_only``: Read-only access (non-coherent L1/Tex cache).. Requires ``sizeof(T) = {1, 2, 4, 8, 16}``.

- ``eviction_policy``: The eviction policy. Requires ``SM >= 70`` and ``sizeof(T) = {1, 2, 4, 8, 16}``. See `"Cache Eviction Priority Hints" <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id150>`_

    - ``eviction_none``: No eviction (default).
    - ``eviction_normal``: Normal eviction behavior. Currently equivent to ``eviction_none``.
    - ``eviction_unchanged``: Unchanged evict policy.
    - ``eviction_first``: First evict policy (streaming).
    - ``eviction_last``: Last evict policy (persisting).
    - ``eviction_no_alloc``: No allocation evict policy (streaming).

- ``prefetch``: The prefetch property. . Requires ``SM >= 75`` and ``sizeof(T) = {1, 2, 4, 8, 16}``. See `ld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>`_ and `ld.global.nc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`_ documentation

    - ``prefetch_spatial_none``: No prefetch (default).
    - ``prefetch_64B``: Prefetch 64 bytes.
    - ``prefetch_128B``: Prefetch 128 bytes.
    - ``prefetch_256B``: Prefetch 256 bytes. Requires ``SM >= 80``.

**Return value**

- The loaded value.

**Mandates**

- The functions requires a valid pointer to global memory.

Example
-------

.. code:: cuda

    #include <cuda/data_movement>

    __device__ unsigned input;
    __device__ unsigned output;

    __global__ void load_kernel() {
        auto ptr = &input;
        output   = cuda::device::load(ptr);
        output   = cuda::device::load(ptr, read_only, eviction_first);
        output   = cuda::device::load(ptr, read_write, eviction_last, prefetch_256B);
    }

    int main() {
        load_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/s8cj8nafc>`_
