.. _libcudacxx-extended-api-warp-warp-match-any:

``cuda::device::warp_match_any``
================================

Defined in ``<cuda/warp>`` header.

.. code:: cuda

    namespace cuda::device {

    template <typename T>
    [[nodiscard]] __device__ lane_mask
    warp_match_any(const T& data, lane_mask = lane_mask::all());

    } // namespace cuda::device

The functionality provides a generalized and safe alternative to CUDA warp match any intrinsic ``__match_any_sync``.
The function allows bitwise comparison of any data size, including raw arrays, pointers, and structs.

.. note::

  The underlying CUDA intrinsic does not provide memory ordering.

**Parameters**

- ``data``: data to compare.
- ``lane_mask``: mask of the active lanes.

**Return value**

- A ``lane_mask`` representing the non-exited lanes in ``lane_mask`` that have the same bitwise value for ``data`` as  the calling lane.

**Constraints**

- ``T`` shall be trivially copyable, see :ref:`cuda::is_trivially_copyable <libcudacxx-extended-api-type_traits-is_trivially_copyable>`.
- When ``__builtin_clear_padding`` is not supported, ``T`` shall have no padding bits, that is, ``T``'s value representation shall be identical to its object representation.

**Preconditions**

- The functionality is only supported on ``SM >= 70``.
- ``lane_mask`` must be non-zero.

**Undefined Behavior**

- ``lane_mask`` must represent a subset of the active lanes.
- All non-exited lanes specified by ``lane_mask`` must execute the function with the same ``lane_mask`` value.

**Performance considerations**

- The function calls the PTX instruction ``match.sync`` :math:`ceil\left(\frac{sizeof(data)}{4}\right)` times.
- The function is faster when called with a mask representing all active lanes in a warp (default value of the second parameter ``lane_mask``).
- The function uses ``__ballot_sync`` when ``T`` is ``bool``.

**References**

- `CUDA match_any Intrinsics <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-match-functions>`_
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
        {
            auto mask     = cuda::device::warp_match_any(threadIdx.x / 4);
            auto expected = cuda::device::lane_mask{0b1111 << ((threadIdx.x / 4) * 4)};
            assert(mask == expected);
        }
        {
            auto mask     = cuda::device::warp_match_any(2);
            auto expected = cuda::device::lane_mask{0xFFFFFFFF};
            assert(mask == expected);
        }
        {
            // compile error, except when __builtin_clear_padding is supported
            auto mask     = cuda::device::warp_match_any(MyStruct{1.0, 3});
            auto expected = cuda::device::lane_mask{0xFFFFFFFF};
            assert(mask == expected);
        }
    }

    int main() {
        warp_match_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/Ys1McG8nv>`_
