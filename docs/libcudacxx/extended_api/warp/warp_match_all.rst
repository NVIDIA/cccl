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

.. note::

  The underlying CUDA intrinsic does not provide memory ordering.

**Parameters**

- ``data``: data to compare.
- ``lane_mask``: mask of the active lanes.

**Return value**

- ``true`` if all lanes in the ``lane_mask`` have the same value for ``data``. ``false`` otherwise.

**Constraints**

- ``T`` shall be trivially copyable, see :ref:`cuda::is_trivially_copyable <libcudacxx-extended-api-type_traits-is_trivially_copyable>`.
- ``T`` shall be bitwise comparable, see :ref:`cuda::is_bitwise_comparable <libcudacxx-extended-api-type_traits-is_bitwise_comparable>`, except when ``__builtin_clear_padding`` is supported. In the latter case, ``T`` can have padding bits.

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

- `CUDA match_all Intrinsics <https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-match-functions>`_
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
        assert(cuda::device::warp_match_all(MyStruct{1.0, 3})); // compile error, except when __builtin_clear_padding is supported
        assert(!cuda::device::warp_match_all(threadIdx.x));
    }

    int main() {
        warp_match_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/x1sWbx14r>`_
