.. _libcudacxx-extended-api-warp-warp-match-all:

``cuda::device::warp_match_all``
================================

Defined in ``<cuda/warp>`` header.

.. code:: cuda

    namespace cuda::device {

    template <typename T>
    [[nodiscard]] __device__ bool
    warp_match_all(const T& data, lane_mask = lane_mask::all());

    } // namespace cuda::device

The functionality provides a generalized and safe alternative to CUDA warp match all intrinsic ``__match_all_sync``.
The function allows bitwise comparison of any data size, including raw arrays, pointers, and structs.

**Parameters**

- ``data``: data to compare.
- ``lane_mask``: mask of the active lanes.

**Return value**

- ``true`` if all lanes in the ``lane_mask`` have the same value for ``data``. ``false`` otherwise.

**Preconditions**

- The functionality is only supported on ``SM >= 70``.
- ``lane_mask`` must be non-zero.
- ``T`` shall have no padding bits, that is, ``T``'s value representation shall be identical to its object representation.

**Undefined Behavior**

- ``lane_mask`` must represent a subset of the active lanes, undefined behavior otherwise.

**Performance considerations**

- The function calls the PTX instruction ``match.sync`` :math:`ceil\left(\frac{sizeof(data)}{4}\right)` times.
- The function is slightly faster when called with a mask of all active lanes (overload function) even if all lanes participates in the call.
- The function is slower when called with a non-fully active warp.

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
        double x; // 8 bytes
        int    y; // 4 bytes
    };            // 4 bytes of padding

    __global__ void warp_match_kernel() {
        assert(cuda::device::warp_match_all(2));
        assert(cuda::device::warp_match_all(2, cuda::device::lane_mask::all()));
        assert(cuda::device::warp_match_all(MyStruct{1.0, 3})); // Undefined Behavior
        assert(!cuda::device::warp_match_all(threadIdx.x));
    }

    int main() {
        warp_match_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/Eq81fTb8z>`_
