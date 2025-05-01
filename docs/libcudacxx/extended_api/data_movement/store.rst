.. _libcudacxx-extended-api-data-movement-store:

``store``: Store data to global memory
======================================

.. code:: cuda

    template <typename T>
    __device__ inline
    void store(T                         data,
               const T*                  ptr,
               /*eviction_policy_type*/  eviction_policy = L1_unchanged_reuse,
               /*cuda::access_property*/ access_property = access_property::global{})

    template <size_t N, typename T, size_t Align>
    __device__ inline
    void store(cuda::std::array<T, N>      data,
               const T*                    ptr,
               cuda::aligned_size_t<Align> align,
               /*eviction_policy_type*/    eviction_policy = L1_unchanged_reuse,
               /*cuda::access_property*/   access_property = access_property::global{})

    template <typename T, typename Prop>
    __device__ inline
    void store(T                            data,
               cuda::annotated_ptr<T, Prop> ptr,
               /*eviction_policy_type*/     eviction_policy = L1_unchanged_reuse)

    template <size_t N, typename T, typename Prop, size_t Align>
    __device__ inline
    void store(cuda::std::array<T, N>       data,
               cuda::annotated_ptr<T, Prop> ptr,
               cuda::aligned_size_t<Align>  align,
               /*eviction_policy_type*/     eviction_policy = L1_unchanged_reuse)

The function stores data to global memory and allows to specify memory access properties.

**Parameters**

- ``ptr``: Pointer to the data to be stored.
- ``eviction_policy``: The eviction policy. Requires ``SM >= 70`` and ``sizeof(T) = {1, 2, 4, 8, 16}``. See also `"Cache Eviction Priority Hints" <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#id150>`_

    - ``L1_unchanged_reuse``: No eviction (default).
    - ``L1_normal_reuse``: Normal eviction behavior. Currently equivent to ``L1_unchanged_reuse``.
    - ``L1_unchanged_reuse``: Unchanged evict policy.
    - ``L1_low_reuse``: First evict policy (streaming).
    - ``L1_high_reuse``: Last evict policy (persisting).
    - ``L1_no_reuse``: No allocation evict policy (streaming).

- ``access_property``: The L2 cache residency control property. See `cuda::access_property <https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_access_properties/access_property.html>`_ documentation

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

    __device__ unsigned output;

    __global__ void store_kernel() {
        auto ptr = &output;
        cuda::device::store(1, ptr);
        cuda::device::store(2, ptr, cuda::device::L1_low_reuse);
        cuda::device::store(3, ptr, cuda::device::L1_high_reuse);
    }

    int main() {
        store_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/fd8od9qfP>`_
