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
#  include <cuda/std/__algorithm/fill.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/mdspan>

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
 * @brief Computes the maximal common contiguous layout between two layouts.
 *
 * Given two layouts @p __layoutA and @p __layoutB, this function finds the largest contiguous region (stride-1 prefix)
 * that both layouts share.

 * @par Algorithm
 * 1. Compute the right inverse of layout B: `B^-1 = right_inverse(B)`
 * 2. Compose A with `B^-1` to find how A maps through B's inverse: `common = A . B^-1`
 * 3. Coalesce to merge consecutive stride-compatible modes
 * 4. Extract the first mode (contiguous portion) and compose back through `B^-1`
 *
 * @return A layout representing the maximal contiguous region common to both layouts, expressed in the coordinate space
 *         of layout B.
 */
template <class _ShapeA, class _StrideA, class _ShapeB, class _StrideB>
[[nodiscard]] _CCCL_HOST_API constexpr auto __max_common_layout(
  const ::cute::Layout<_ShapeA, _StrideA>& __layoutA, const ::cute::Layout<_ShapeB, _StrideB>& __layoutB) noexcept
{
  ::cute::Layout __inv_layoutB = ::cute::right_inverse(__layoutB);
  ::cute::Layout __common      = ::cute::coalesce(::cute::composition(__layoutA, __inv_layoutB));
  return ::cute::composition(__inv_layoutB, ::cute::layout<0>(__common));
}

template <typename _TpIn,
          typename _ExtentsIn,
          typename _LayoutPolicyIn,
          typename _AccessorPolicyIn,
          typename _TpOut,
          typename _ExtentsOut,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyOut>
_CCCL_HOST_API void __copy_impl(::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn> __src,
                                ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut> __dst,
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
    // find the maximal common layout between the two layouts and divide the tensors into tiles
    const auto __src1              = ::cuda::experimental::to_cute(__src);
    const auto __dst1              = ::cuda::experimental::to_cute(__dst);
    const auto __max_common_layout = ::cuda::experimental::__max_common_layout(__src1.layout(), __dst1.layout());
    const auto __src_tiles         = ::cute::logical_divide(__src1, __max_common_layout);
    const auto __dst_tiles         = ::cute::logical_divide(__dst1, __max_common_layout);
    _CCCL_ASSERT(::cute::size<1>(__src_tiles) == ::cute::size<1>(__dst_tiles),
                 "tensors must have the same number of tiles");
    // copy the tiles host <-> device
    const auto __copy_bytes = ::cute::size<0>(__src_tiles) * sizeof(_TpIn);
#  if _CCCL_CTK_AT_LEAST(12, 9)
    // use the memcpy batch API to copy the tiles
    const auto __num_copies = ::cute::size<1>(__src_tiles);
    ::std::vector<const void*> __src_ptr_vector(__num_copies);
    ::std::vector<void*> __dst_ptr_vector(__num_copies);
    for (int __tile_idx = 0; __tile_idx < ::cute::size<1>(__src_tiles); ++__tile_idx)
    {
      auto __src_tile = __src_tiles(::cute::_, __tile_idx);
      auto __dst_tile = __dst_tiles(::cute::_, __tile_idx);
      auto __src_ptr  = ::cuda::std::addressof(__src_tile[0]);
      auto __dst_ptr  = ::cuda::std::addressof(__dst_tile[0]);
      __src_ptr_vector.push_back(static_cast<const void*>(__src_ptr));
      __dst_ptr_vector.push_back(static_cast<void*>(__dst_ptr));
    }
    ::std::vector<::cuda::std::size_t> __sizes(__num_copies);
    ::cuda::std::fill(__sizes.begin(), __sizes.end(), __copy_bytes);
    ::cuda::__driver::__memcpyBatchAsync(
      __dst_ptr_vector.data(),
      __src_ptr_vector.data(),
      __sizes.data(),
      __num_copies,
      nullptr,
      nullptr,
      0,
      __stream.get());
#  else
    // use the memcpy API to copy the tiles
    for (int __tile_idx = 0; __tile_idx < ::cute::size<1>(__src_tiles); ++__tile_idx)
    {
      auto __src_tile = __src_tiles(::cute::_, __tile_idx);
      auto __dst_tile = __dst_tiles(::cute::_, __tile_idx);
      auto __src_ptr  = static_cast<const void*>(::cuda::std::addressof(__src_tile[0]));
      auto __dst_ptr  = static_cast<void*>(::cuda::std::addressof(__dst_tile[0]));
      ::cuda::__driver::__memcpyAsync(__dst_ptr, __src_ptr, __copy_bytes, __stream.get());
    }
#  endif // _CCCL_CTK_AT_LEAST(12, 9)
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
  ::cuda::experimental::__copy_impl(static_cast<__src_type>(__src), static_cast<__dst_type>(__dst), __stream);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_D2H_H2D_H
