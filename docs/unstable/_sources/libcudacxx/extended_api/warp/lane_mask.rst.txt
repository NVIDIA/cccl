.. _libcudacxx-extended-api-warp-lane-mask:

``cuda::device::lane_mask``
===========================

Defined in ``<cuda/warp>`` header.

.. code:: cuda

    #include <cuda/std/cstdint>

    namespace cuda::device
    {

    class lane_mask
    {
        // constructors
        __device__ explicit constexpr lane_mask(cuda::std::uint32_t v = 0) noexcept;

        // member functions
        [[nodiscard]] __device__ constexpr cuda::std::uint32_t value() const noexcept;

        // conversion operators
        __device__ explicit constexpr operator cuda::std::uint32_t() const noexcept;

        // static member functions
        [[nodiscard]] __device__ static constexpr lane_mask none() noexcept;
        [[nodiscard]] __device__ static constexpr lane_mask all() noexcept;
        [[nodiscard]] __device__ static lane_mask all_active() noexcept;
        [[nodiscard]] __device__ static lane_mask this_lane() noexcept;
        [[nodiscard]] __device__ static lane_mask all_less() noexcept;
        [[nodiscard]] __device__ static lane_mask all_less_equal() noexcept;
        [[nodiscard]] __device__ static lane_mask all_greater() noexcept;
        [[nodiscard]] __device__ static lane_mask all_greater_equal() noexcept;
        [[nodiscard]] __device__ static lane_mask all_not_equal() noexcept;

        // bitwise assignment operators
        __device__ constexpr lane_mask& operator&=(lane_mask mask) noexcept;
        __device__ constexpr lane_mask& operator|=(lane_mask mask) noexcept;
        __device__ constexpr lane_mask& operator^=(lane_mask mask) noexcept;
        __device__ constexpr lane_mask& operator<<=(int shift) noexcept;
        __device__ constexpr lane_mask& operator>>=(int shift) noexcept;

        // bitwise operators
        [[nodiscard]] __device__ friend constexpr lane_mask operator~(lane_mask mask) noexcept;
        [[nodiscard]] __device__ friend constexpr lane_mask operator&(lane_mask lhs, lane_mask rhs) noexcept;
        [[nodiscard]] __device__ friend constexpr lane_mask operator|(lane_mask lhs, lane_mask rhs) noexcept;
        [[nodiscard]] __device__ friend constexpr lane_mask operator^(lane_mask lhs, lane_mask rhs) noexcept;
        [[nodiscard]] __device__ friend constexpr lane_mask operator<<(lane_mask mask, int shift) noexcept;
        [[nodiscard]] __device__ friend constexpr lane_mask operator>>(lane_mask mask, int shift) noexcept;

        // comparison operators
        [[nodiscard]] __device__ friend constexpr bool operator==(lane_mask lhs, lane_mask rhs) noexcept;
        [[nodiscard]] __device__ friend constexpr bool operator!=(lane_mask lhs, lane_mask rhs) noexcept;
    };

    } // namespace cuda::device

``cuda::device::lane_mask`` is a class that represents a mask of lanes in a warp. It is a fancy wrapper around a single 32-bit unsigned integer value that allows for bitwise operations and comparisons, making it easier and safer to work with lane masks in CUDA device code.

The class provides several ``static`` member functions to create common lane masks:

- ``none()`` and ``all()`` are equivalent to ``lane_mask{0x0}`` and ``lane_mask{0xFFFFFFFF}``, respectively
- ``all_active()`` returns a mask with all currently active lanes in the warp, equivalent to the result ``__activemask()``, and finally
- ``this_lane()`` and other functions like ``all_greater()`` or ``all_less_equal()`` return masks depending on the current lane index. They are implemented using the PTX special registers.

**Preconditions**

- ``shift`` is in the range ``[0, 32)``.

Example
-------

.. code:: cuda

    #include <cuda/std/cassert>
    #include <cuda/std/type_traits>
    #include <cuda/warp>

    __global__ void lane_mask_kernel() {
        // import lane_mask symbol to current scope
        using cuda::device::lane_mask;
        // this_lane() is equivalent to ~(all_less() | all_greater())
        assert(lane_mask::this_lane() == ~(lane_mask::all_less() | lane_mask::all_greater()));
    }

    int main() {
        lane_mask_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/W7hExs16v>`_
