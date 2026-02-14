.. _libcudacxx-extended-api-warp-warp-shuffle:

``cuda::device::warp_shuffle_idx/up/down/xor``
==============================================

Defined in ``<cuda/warp>`` header.

``warp_shuffle_idx``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_idx(const T& data,
                     int      src_lane,
                     uint32_t lane_mask = 0xFFFFFFFF,
                     cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_idx(const T& data,
                     int      src_lane,
                     cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

``warp_shuffle_up``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_up(const T& data,
                    int      delta,
                    uint32_t lane_mask = 0xFFFFFFFF,
                    cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_up(const T& data,
                    int      delta,
                    cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

``warp_shuffle_down``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_down(const T& data,
                      int      delta,
                      uint32_t lane_mask = 0xFFFFFFFF,
                      cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_down(const T& data,
                      int      delta,
                      cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

``warp_shuffle_xor``:

.. code:: cuda

    namespace cuda::device {

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_xor(const T& data,
                     int      xor_mask,
                     uint32_t lane_mask = 0xFFFFFFFF,
                     cuda::std::integral_constant<int, Width> = {})

    template <int Width = 32, typename T>
    [[nodiscard]] __device__ warp_shuffle_result<T>
    warp_shuffle_xor(const T& data,
                     int      xor_mask,
                     cuda::std::integral_constant<int, Width>) // lane_mask is 0xFFFFFFFF

    } // namespace cuda::device

Result type:

.. code:: cuda

    namespace cuda::device {

    template <typename T>
    struct warp_shuffle_result {
        T    data;
        bool pred;

        __device__ operator T() const { return data; }
    };

    } // namespace cuda::device

The functionality provides a generalized and safe alternative to CUDA warp shuffle intrinsics.
The functions allow to exchange data of any data size, including raw arrays, pointers, and structs.

**Parameters**

- ``data``: data to exchange.
- ``src_lane``: source lane.
- ``delta``: offset from the source lane.
- ``xor_mask``: XOR mask to apply to the source lane.

**Return value**

``warp_shuffle_result``:

- ``data``: data of the destination lane.
- ``pred``: ``true`` if the destination lane is within the source lane window. ``false`` otherwise.

**Constrains**

- ``Width`` must be a power of two in the range [1, 32]
- ``T``: all ``T`` are allowed except if ``T`` is a pointer, in which case it must be a ``void`` pointer to avoid bug-prone code

**Preconditions**

- The destination lane must be a member of the ``lane_mask``.
- ``delta`` and ``xor_mask`` must be less than ``Width``. Modulo behavior is allowed for ``src_lane``.
- ``lane_mask`` must be non-zero.

**Undefined Behavior**

- ``lane_mask`` must represent a subset of the active lanes, undefined behavior otherwise.
- All lanes must have the same value for ``lane_mask``, ``delta`` and ``xor_mask``

**Performance considerations**

- The function calls the PTX instruction ``shfl.sync`` :math:`ceil\left(\frac{sizeof(data)}{4}\right)` times.

**References**

- `CUDA Warp Shuffle Intrinsics <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle>`_
- `PTX Shfl.sync instruction <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync>`_

Example
-------

.. code:: cuda

    #include <cuda/std/array>
    #include <cuda/std/type_traits>
    #include <cuda/warp>
    #include <cstdio>

    struct MyStruct {
        double x;
        int    y;
    };

    __global__ void warp_shuffle_kernel() {
        cuda::std::integral_constant<int, 16> half_warp;
        auto                     laneid      = cuda::ptx::get_sreg_laneid();
        int                      raw_array[] = {threadIdx.x, threadIdx.x + 1, threadIdx.x + 2};
        cuda::std::array<int, 3> array       = {threadIdx.x, threadIdx.x + 1, threadIdx.x + 2};
        MyStruct                 my_structs{static_cast<double>(threadIdx.x), threadIdx.x + 1};
        if (laneid < 16) {
            // lanes [0, 15] get an array with values {5, 6, 7}
            auto ret = cuda::device::warp_shuffle_idx(raw_array, 5, 0xFFFF, half_warp);
            printf("lane %2d: [%d, %d, %d]\n", laneid, ret.data[0], ret.data[1], ret.data[2]);

            // lanes [1, 15] get an array with values {threadIdx.x - 1, threadIdx.x, threadIdx.x + 1}
            // lane 0 keeps the original values
            auto array_ret = cuda::device::warp_shuffle_up(array, 1, half_warp).data;
            printf("lane %2d: [%d, %d, %d]\n", laneid, array[0], array[1], array_ret[2]);
        }
        // lanes [0, 13] get my_structs with values {threadIdx.x + 2, threadIdx.x + 3} and pred=true
        auto ret = cuda::device::warp_shuffle_down<16>(my_structs, 2);
        printf("lane %2d: {%f, %d}, pred %d\n", laneid, ret.data.x, ret.data.y, ret.pred);
    }

    int main() {
        warp_shuffle_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/soWTaG6Eb>`_
