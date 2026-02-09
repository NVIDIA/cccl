//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_D2H_H2D_H
#define __CUDAX_COPY_MDSPAN_D2H_H2D_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/traits.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/experimental/__copy/cute/logical_divide.cuh>
#  include <cuda/experimental/__copy/cute/max_common_layout.cuh>
#  include <cuda/experimental/__copy/mdspan_to_cute.cuh>

#  include <stdexcept>
#  include <vector>

#  include <cute/layout.hpp>
#  include <cute/tensor_impl.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Runtime equivalent of CuTe's `right_inverse` for layouts with dynamic strides.
 *
 * CuTe's `right_inverse` requires fully-static strides. This function implements the same algorithm at runtime.
 *
 * @par Algorithm
 * 1. Coalesce the layout to merge consecutive stride-compatible modes.
 * 2. Compute the exclusive prefix product of shapes.
 * 3. Sort dimensions by stride (ascending) to determine contiguity order.
 * 4. Iterate over dimensions in stride-sorted order and keep only the contiguous dimensions,
 *    i.e. when a stride matches the expected next contiguous one.
 *
 * @par Postcondition
 * The returned layout `R` satisfies `layout(R(i)) == i` for all `i < size(R)`, i.e., it maps the contiguous
 * codomain indices back to the domain of `__layout`.
 */
/*
template <class _Shape, class _Stride>
_CCCL_HOST_API auto __right_inverse_dynamic(const ::cute::Layout<_Shape, _Stride>& __layout) noexcept
{
  const auto __clayout                 = ::cuda::experimental::__coalesce(__layout);
  const auto __lshape                  = __clayout.shape();
  const auto __lstride                 = __clayout.stride();
  constexpr ::cuda::std::size_t __rank = decltype(::cute::rank(__lshape))::value;
  if constexpr (__rank == 0)
  {
    return __clayout;
  }
  else
  {
    using ::cuda::std::int64_t;
    constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};
    ::cuda::std::array<int64_t, __rank> __shapes;
    ::cuda::std::array<int64_t, __rank> __strides;
    ::cuda::std::array<int64_t, __rank> __orders;
    ::cuda::experimental::__init_and_sort_layout(__lshape, __lstride, __shapes, __strides, __orders, __rank_seq);

    ::cuda::std::array<int64_t, __rank> __preprods{}; // Prefix product of the shape shape[0] * ... * shape[i-1]
    ::cuda::std::array<int64_t, __rank> __result_shapes{};
    ::cuda::std::array<int64_t, __rank> __result_strides{};
    // compute the exclusive prefix product of the shapes
    __preprods[0] = 1;
    for (::cuda::std::size_t __i = 1; __i < __rank; ++__i)
    {
      __preprods[__i] = __preprods[__i - 1] * __shapes[__i - 1];
    }
    int64_t __curr_stride = 1;
    for (::cuda::std::size_t __pos = 0; __pos < __rank; ++__pos)
    {
      const auto __idx = __orders[__pos];
      if (__strides[__idx] == __curr_stride)
      {
        __result_shapes[__pos]  = __shapes[__idx];
        __result_strides[__pos] = __preprods[__idx];
        __curr_stride *= __shapes[__idx];
      }
      else
      {
        __result_shapes[__pos]  = 1;
        __result_strides[__pos] = 0;
      }
    }
    const auto __shapes_tuple  = ::cuda::experimental::__to_cute_tuple(__result_shapes, __rank_seq);
    const auto __strides_tuple = ::cuda::experimental::__to_cute_tuple(__result_strides, __rank_seq);
    return ::cuda::experimental::__coalesce(::cute::make_layout(__shapes_tuple, __strides_tuple));
  }
}
*/

enum class __copy_direction
{
  host_to_device,
  device_to_host,
};

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void __copy_impl(
  ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
  ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
  __copy_direction __direction,
  ::cuda::stream_ref __stream)
{
  static_assert(::cuda::std::is_trivially_copyable_v<_TpIn>, "TpIn must be trivially copyable");
  static_assert(::cuda::std::is_trivially_copyable_v<_TpOut>, "TpOut must be trivially copyable");
  static_assert(!::cuda::std::is_const_v<_TpOut>, "TpOut must not be const");
  static_assert(::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>,
                "TpIn and TpOut must be the same type");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyIn>,
                "LayoutPolicyIn must be a predefined layout policy");
  static_assert(::cuda::__is_cuda_mdspan_layout_v<_LayoutPolicyOut>,
                "LayoutPolicyOut must be a predefined layout policy");
  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>,
                "AccessorPolicyIn must be convertible to cuda::std::default_accessor");
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>,
                "AccessorPolicyOut must be convertible to cuda::std::default_accessor");
  if (__src.size() != __dst.size())
  {
    _CCCL_THROW(std::invalid_argument, "mdspans must have the same size");
  }
  if (__src.size() == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(std::invalid_argument, "mdspan data handle must not be nullptr");
  }
  if constexpr (_ExtentsIn::rank() == 0)
  {
    ::cuda::__driver::__memcpyAsync(__dst.data_handle(), __src.data_handle(), sizeof(_TpIn), __stream.get());
  }
  else
  {
    using ::cuda::std::size_t;
    const auto __src1 = ::cuda::experimental::to_cute(__src);
    cute::print(__src1.layout());
    printf("\n");
    cute::print(::cute::coalesce(cute::composition(__src1.layout(), ::cute::right_inverse(__src1.layout()))));
    printf("\n");
    const auto __dst1 = ::cuda::experimental::to_cute(__dst);
    printf("\n");
    cute::print(::cute::right_inverse(__dst1.layout()));

    constexpr auto __rank_in  = _ExtentsIn::rank();
    constexpr auto __rank_out = _ExtentsOut::rank();
    constexpr auto __rank_max = ::cuda::std::max(__rank_in, __rank_out);
    auto __src2               = ::cute::make_tensor(__src1.data(), ::cute::append<__rank_max>(__src1.layout()));
    auto __dst2               = ::cute::make_tensor(__dst1.data(), ::cute::append<__rank_max>(__dst1.layout()));
    printf("\n------------------------\n");
    cute::print(__src2.layout());
    printf("\n");
    cute::print(__dst2.layout());
    // Find the maximal common layout between the two layouts and divide the tensors into tiles.

    const auto __max_common_layout = ::cuda::experimental::__max_common_layout(__src2.layout(), __dst2.layout());
    printf("\n------------------------\n");
    cute::print(__max_common_layout);

    const auto __src_tiles = ::cuda::experimental::__logical_divide(__src2, __max_common_layout);
    printf("\n");
    cute::print(__src_tiles);
    const auto __dst_tiles = ::cuda::experimental::__logical_divide(__dst2, __max_common_layout);
    printf("\n");
    cute::print(__dst_tiles);
    printf("\n------------------------\n");
    // After logical_divide, size<0>: tile size, size<1>: number of tiles.
    const auto __tile_size  = static_cast<size_t>(::cute::size(__max_common_layout));
    const auto __copy_bytes = __tile_size * sizeof(_TpIn);
    const auto __num_tiles  = static_cast<size_t>(::cute::size<1>(__src_tiles));
#  if _CCCL_CTK_AT_LEAST(13, 0)
    // Use the memcpy batch API to copy all tiles in one call
    ::std::vector<const void*> __src_ptr_vector(__num_tiles);
    ::std::vector<void*> __dst_ptr_vector(__num_tiles);
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      __src_ptr_vector[__tile_idx] = static_cast<const void*>(__src_tiles(::cute::_, __tile_idx).data());
      __dst_ptr_vector[__tile_idx] = static_cast<void*>(__dst_tiles(::cute::_, __tile_idx).data());
    }
    ::std::vector<size_t> __sizes(__num_tiles, __copy_bytes);

    constexpr auto __h2d_attributes = ::CUmemcpyAttributes{
      ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0},
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, 0},
      0};
    constexpr auto __d2h_attributes = ::CUmemcpyAttributes{
      ::CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE, 0},
      ::CUmemLocation{CU_MEM_LOCATION_TYPE_HOST, 0},
      0};
    auto __attributes = (__direction == __copy_direction::host_to_device) ? __h2d_attributes : __d2h_attributes;

    size_t __zero = 0;
    ::cuda::__driver::__memcpyBatchAsync(
      __dst_ptr_vector.data(),
      __src_ptr_vector.data(),
      __sizes.data(),
      __num_tiles,
      &__attributes,
      &__zero,
      1,
      __stream.get());
#  else
    // Use individual memcpy calls for each tile
    for (size_t __tile_idx = 0; __tile_idx < __num_tiles; ++__tile_idx)
    {
      auto __src_ptr = static_cast<const void*>(__src_tiles(::cute::_, __tile_idx).data());
      auto __dst_ptr = static_cast<void*>(__dst_tiles(::cute::_, __tile_idx).data());
      ::cuda::__driver::__memcpyAsync(__dst_ptr, __src_ptr, __copy_bytes, __stream.get());
    }
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  }
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void copy(::cuda::host_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                         ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                         ::cuda::stream_ref __stream)
{
  using __src_type = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
  using __dst_type = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
  ::cuda::experimental::__copy_impl(
    static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __copy_direction::host_to_device, __stream);
}

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void copy(::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                         ::cuda::host_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
                         ::cuda::stream_ref __stream)
{
  using __src_type = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
  using __dst_type = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
  ::cuda::experimental::__copy_impl(
    static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __copy_direction::device_to_host, __stream);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_D2H_H2D_H
