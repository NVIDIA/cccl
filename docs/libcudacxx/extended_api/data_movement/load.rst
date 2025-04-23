.. _libcudacxx-extended-api-data-movement-load:

``load``: Load data from global memory
======================================

Defined in the header ``<cuda/data_movement>``.

.. code:: cuda

    template <typename T>
    [[nodiscard]] __device__ inline
    T load(const T*                  ptr,
           /*memory_access_type*/    memory_access   = read_write,
           /*eviction_policy_type*/  eviction_policy = eviction_none,
           /*prefetch_L2_type*/      prefetch_L2     = prefetch_L2_none,
           /*cuda::access_property*/ access_property = access_property::global{})

    template <size_t N, typename T, size_t Align>
    [[nodiscard]] __device__ inline
    cuda::std::array<T, N> load(const T*                    ptr,
                                cuda::aligned_size_t<Align> align,
                                /*memory_access_type*/      memory_access   = read_write,
                                /*eviction_policy_type*/    eviction_policy = eviction_none,
                                /*prefetch_L2_type*/        prefetch_L2     = prefetch_L2_none,
                                /*cuda::access_property*/   access_property = access_property::global{})

    template <typename T, typename Prop>
    [[nodiscard]] __device__ inline
    T load(cuda::annotated_ptr<T, Prop> ptr,
           /*memory_access_type*/       memory_access   = read_write,
           /*eviction_policy_type*/     eviction_policy = eviction_none,
           /*prefetch_L2_type*/         prefetch_L2     = prefetch_L2_none)

    template <size_t N, typename T, typename Prop, size_t Align>
    [[nodiscard]] __device__ inline
    cuda::std::array<T, N> load(cuda::annotated_ptr<T, Prop> ptr,
                                cuda::aligned_size_t<Align> align,
                                /*memory_access_type*/       memory_access   = read_write,
                                /*eviction_policy_type*/     eviction_policy = eviction_none,
                                /*prefetch_L2_type*/         prefetch_L2     = prefetch_L2_none)

The function loads data from global memory and allows to specify memory access properties.

**Parameters**

- ``ptr``: Pointer to the data to be loaded.
- ``memory_access``:  The memory access property.

    - ``read_write``: Read and write access (default).
    - ``read_only``: Read-only access (non-coherent L1/Tex cache).. Requires ``sizeof(T) = {1, 2, 4, 8, 16}``.

- ``eviction_policy``: The eviction policy. Requires ``SM >= 70`` and ``sizeof(T) = {1, 2, 4, 8, 16}``. See `"Cache Eviction Priority Hints" <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id150>`_

    - ``eviction_none``: Do not specify the eviction policy (default).
    - ``eviction_normal``: Normal eviction policy.
    - ``eviction_unchanged``: Unchanged evict policy.
    - ``eviction_first``: First evict policy (streaming).
    - ``eviction_last``: Last evict policy (persisting).
    - ``eviction_no_alloc``: No allocation evict policy (streaming).

- ``prefetch``: The prefetch property. Requires ``SM >= 75`` and ``sizeof(T) = {1, 2, 4, 8, 16}``. See `ld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>`_ and `ld.global.nc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`_ documentation

    - ``prefetch_L2_none``: No prefetch (default).
    - ``prefetch_L2_64B``: Prefetch 64 bytes.
    - ``prefetch_L2_128B``: Prefetch 128 bytes.
    - ``prefetch_L2_256B``: Prefetch 256 bytes. Requires ``SM >= 80``.

- ``access_property``: The L2 cache residency control property. See `cuda::access_property <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/access_property.html>`_ documentation

**Return value**

- The loaded value.

**Constraints**

Load of a *single element*:

- The pointer must be not a null pointer and a valid address in the global memory space.
- The pointer must be aligned to ``alignof(T)``.
- If a non-default property is specified, ``sizeof(T)`` must be ``{1, 2, 4, 8}`` or a multiple of ``16`` bytes.

Load of *multiple elements*:

- The pointer must be not a null pointer and a valid address in the global memory space.
- The pointer must be aligned to ``Align``.
- ``Align`` is a power of 2 (implicit for ``cuda::aligned_size_t``).
- ``N > 0``
- ``Align >= alignof(T)``
- ``(sizeof(T) * N) % Align == 0``

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
        output   = cuda::device::load(ptr, read_write, eviction_last, prefetch_L2_256B);
    }

    int main() {
        load_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/s8cj8nafc>`_
