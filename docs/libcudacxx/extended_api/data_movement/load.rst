.. _libcudacxx-extended-api-data-movement-load:

``cuda::device::load``: Load data from global memory
====================================================

Defined in the header ``<cuda/data_movement>``.

.. code:: cuda

    template <typename T>
    [[nodiscard]] __device__ inline
    T load(const T*                  ptr,
           /*Memory access kind*/    memory_access = read_write,
           /*L1 reuse policy*/       l1_reuse      = cache_reuse_unchanged,
           /*cuda::access_property*/ l2_hint       = access_property::global{},
           /*L2 prefetch policy*/    l2_prefetch   = L2_prefetch_none)

    template <size_t N, typename T, size_t Align>
    [[nodiscard]] __device__ inline
    cuda::std::array<T, N> load(const T*                    ptr,
                                cuda::aligned_size_t<Align> align,
                                /*Memory access kind*/      memory_access = read_write,
                                /*L1 reuse policy*/         l1_reuse      = cache_reuse_unchanged,
                                /*cuda::access_property*/   l2_hint       = access_property::global{},
                                /*L2 prefetch policy*/      l2_prefetch   = L2_prefetch_none)

    template <typename T, typename Prop>
    [[nodiscard]] __device__ inline
    T load(cuda::annotated_ptr<T, Prop> ptr,
           /*Memory access kind*/       memory_access = read_write,
           /*L1 reuse policy*/          l1_reuse      = cache_reuse_unchanged,
           /*L2 prefetch policy*/       l2_prefetch   = L2_prefetch_none)

    template <size_t N, typename T, typename Prop, size_t Align>
    [[nodiscard]] __device__ inline
    cuda::std::array<T, N> load(cuda::annotated_ptr<T, Prop> ptr,
                                cuda::aligned_size_t<Align>  align,
                                /*Memory access kind*/       memory_access = read_write,
                                /*L1 reuse policy*/          l1_reuse      = cache_reuse_unchanged,
                                /*L2 prefetch policy*/       l2_prefetch   = L2_prefetch_none)

The function loads data from global memory and allows to specify memory access properties.

**Parameters**

- ``ptr``: Pointer to the data to be loaded.
- ``memory_access``:  The memory access property. See `ld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>`_ and `ld.global.nc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`_  sections of the PTX Language Reference.

    - ``read_write``: Read and write access (default).
    - ``read_only``: Read-only access (non-coherent L1/Tex cache). Loading a value from a memory location that is not read-only for the entire lifetime of kernel is undefined behavior.

- ``l1_reuse``: L1 cache reuse policy. Requires ``SM >= 70``. See also `"Cache Eviction Priority Hints" <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id150>`_ section of the PTX Language Reference.

    - ``cache_reuse_unchanged``: Unchanged evict policy.
    - ``cache_reuse_normal``: Normal reuse eviction policy.
    - ``cache_reuse_low``: Low reuse eviction policy (streaming).
    - ``cache_reuse_high``: High reuse eviction policy (persisting).
    - ``cache_no_reuse``: No reuse eviction policy (streaming).

- ``access_property``: L2 cache residency control property. Requires ``SM >= 80``. See also `cuda::access_property <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/access_property.html>`_ documentation.

    - ``access_property::global``: Unchanged evict policy.
    - ``access_property::normal``: Normal reuse eviction policy.
    - ``access_property::streaming``: Low reuse eviction policy (streaming).
    - ``access_property::persisting``: High reuse eviction policy (persisting).
    - ``access_property`` also allows to specify custom properties, namely *fractional* and *address range*.

- ``prefetch``: L2 prefetch property. Requires ``SM >= 75``.

    - ``L2_prefetch_none``: No prefetch (default).
    - ``L2_prefetch_64B``: Prefetch 64 bytes.
    - ``L2_prefetch_128B``: Prefetch 128 bytes.
    - ``L2_prefetch_256B``: Prefetch 256 bytes. Requires ``SM >= 80``.

**NOTE**: if a feature is not available for a given GPU architecture, the function behavior falls back to the closest available feature. For example, if ``L2_prefetch_256B`` is not available, it will be replaced by ``L2_prefetch_128B``.

**Return value**

- The loaded value from the specified global memory address.

**Preconditions**

- If a non-default property is specified, ``sizeof(load_data)`` must be ``{1, 2, 4, 8}`` or a multiple of ``16`` bytes.

Addition constraints for loading *multiple elements* (``cuda::device::load<N>``):

- ``Align`` is a power of 2 (implicit for ``cuda::aligned_size_t``).
- ``N > 0``
- ``Align >= alignof(T)``
- ``(sizeof(T) * N) % Align == 0``

**Constraints**

- The pointer must not be a null pointer and a valid address in the global memory space.

- The pointer must be aligned to ``alignof(T)`` when loading a *single element*, or to ``Align`` when loading *multiple elements* (``cuda::device::load<N>``).

Example
-------

.. code:: cuda

    #include <cuda/annotated_ptr>
    #include <cuda/data_movement>

    __device__ int input;
    __device__ int output;

    __device__ int input2[8];

    __global__ void load_kernel() {
        auto ptr = &input;
        output   = cuda::device::load(ptr);
        output   = cuda::device::load(ptr, read_only, cache_reuse_low);
        output   = cuda::device::load(ptr, read_write, cache_reuse_high, access_property::persisting);
        output   = cuda::device::load(ptr, read_write, cache_reuse_high, access_property::persisting, L2_prefetch_256B);
        output   = cuda::device::load<4>(input2, read_only, cache_reuse_high)[0];

        auto ptr2 = cuda::std::annotated_ptr<int, cuda::access_property::normal>(&input);
        output    = cuda::device::load(ptr2);
    }

    int main() {
        load_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/s8cj8nafc>`_
