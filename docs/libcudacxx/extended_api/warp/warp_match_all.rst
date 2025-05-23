.. _libcudacxx-extended-api-warp-warp-match-all:

Warp Match All
==============

``warp_match_all``:

.. code:: cuda

    template <typename T>
    [[nodiscard]] __device__ bool
    warp_match_all(const T& data, uint32_t lane_mask = 0xFFFFFFFF)

The functionality provides a generalized and safe alternative to CUDA warp match all intrinsic ``__match_all_sync``.
The function allows to exchange data of any data size, including raw arrays, pointers, and structs.

**Parameters**

- ``data``: data to exchange.
- ``lane_mask``: mask of the active lanes

**Return value**

- ``true`` if all lanes in the ``lane_mask`` have the same value for ``data``. ``false`` otherwise

**Preconditions**

- The functionality is only supported on ``SM >= 70``
- ``lane_mask`` must be a subset of the active mask and be non-zero

**Performance considerations**

- The function calls the PTX instruction ``match.sync`` :math:`ceil\left(\frac{sizeof(data)}{4}\right)` times.

**References**

- `CUDA match_all Intrinsics <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-match-functions>`_
- `PTX match.sync instruction <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync>`_

Example
-------

.. code:: cuda

    #include <cuda/std/array>
    #include <cuda/std/cassert>
    #include <cuda/warp>

    struct MyStruct {
        double x;
        int    y;
    };

    __global__ void warp_match_kernel() {
        assert(cuda::device::warp_match_all(2));
        assert(cuda::device::warp_match_all(MyStruct{1.0, 3}));
        assert(!cuda::device::warp_match_all(threadIdx.x));
    }

    int main() {
        warp_match_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/MYv7jMsss>`_
