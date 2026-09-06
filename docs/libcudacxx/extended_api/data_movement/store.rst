.. _libcudacxx-extended-api-data-movement-store:

``cuda::device::store``: Store data to global memory
====================================================

Defined in the header ``<cuda/data_movement>``.

.. code:: cuda

    template <typename T>
    __device__ inline
    void store(T                         data,
               T*                        ptr,
               /*L1 reuse policy*/       l1_reuse = cache_reuse_unchanged,
               /*cuda::access_property*/ l2_hint  = access_property::global{})

    template <size_t N, typename T, size_t Align>
    __device__ inline
    void store(cuda::std::array<T, N>      data,
               T*                          ptr,
               cuda::aligned_size_t<Align> align,
               /*L1 reuse policy*/         l1_reuse        = cache_reuse_unchanged,
               /*cuda::access_property*/   l2_hint  = access_property::global{})

    template <typename T, typename Prop>
    __device__ inline
    void store(T                            data,
               cuda::annotated_ptr<T, Prop> ptr,
               /*L1 reuse policy*/          l1_reuse = cache_reuse_unchanged)

    template <size_t N, typename T, typename Prop, size_t Align>
    __device__ inline
    void store(cuda::std::array<T, N>       data,
               cuda::annotated_ptr<T, Prop> ptr,
               cuda::aligned_size_t<Align>  align,
               /*L1 reuse policy*/          l1_reuse = cache_reuse_unchanged)

The function stores data to global memory and allows to specify memory access properties.

**Parameters**

- ``ptr``: Pointer to the data to be stored. See also `cuda::annotated_ptr <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/annotated_ptr.html>`_ documentation.

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

**Preconditions**

- If a non-default property is specified, ``sizeof(load_data)`` must be ``{1, 2, 4, 8}`` or a multiple of ``16`` bytes.

Addition constraints for storing *multiple elements* (``cuda::device::store<N>``):

- ``Align`` is a power of 2 (implicit for ``cuda::aligned_size_t``).
- ``N > 0``
- ``Align >= alignof(T)``
- ``(sizeof(T) * N) % Align == 0``

**Constraints**

- The pointer must not be a null pointer and a valid address in the global memory space.

- The pointer must be aligned to ``alignof(T)`` when storing a *single element*, or to ``Align`` when storing *multiple elements* (``cuda::device::store<N>``).

Example
-------

.. code:: cuda

    #include <cuda/annotated_ptr>
    #include <cuda/array>
    #include <cuda/data_movement>

    __device__ int output;
    __device__ int output2[8];

    __global__ void store_kernel() {
        auto ptr = &output;
        cuda::device::store(1, ptr);
        cuda::device::store(2, ptr, cuda::device::cache_reuse_low);
        cuda::device::store(3, ptr, cuda::device::cache_reuse_high);

        cuda::std::array array{1, 2, 3};
        cuda::device::store(array, output2);

        auto ptr2 = cuda::std::annotated_ptr<int, cuda::access_property::normal>(&input);
        cuda::device::store(1, ptr2);
    }

    int main() {
        store_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/fd8od9qfP>`_
