//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_TO_CUTE_H
#define __CUDAX_COPY_MDSPAN_TO_CUTE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/driver_api.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <cuda/experimental/__copy/mdspan_to_cute.cuh>

#include <cute/layout.hpp>
#include <cute/tensor_impl.hpp>

#if !_CCCL_COMPILER(NVRTC)
#  include <stdexcept>
#  include <vector>
#endif // !_CCCL_COMPILER(NVRTC)
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
_CCCL_HOST_API inline void __batched_copy(
  ::cuda::std::span<const void*> __src_ptrs,
  ::cuda::std::span<void*> __dst_ptrs,
  ::cuda::std::size_t __size_bytes,
  ::cuda::stream_ref __stream)
{
  const auto __num_copies = __src_ptrs.size();
#if _CCCL_CTK_AT_LEAST(12, 9)
  ::std::vector<::cuda::std::size_t> __sizes(__num_copies);
  for (::cuda::std::size_t __i = 0; __i < __num_copies; ++__i)
  {
    __sizes[__i] = __size_bytes;
  }
  ::cuda::__driver::__memcpyBatchAsync(
    __dst_ptrs.data(), __src_ptrs.data(), __sizes.data(), __num_copies, nullptr, nullptr, 0, __stream.get());
#else // ^^^^^ _CCCL_CTK_AT_LEAST(12, 9) / vvvvv _CCCL_CTK_BELOW(12, 9)
  for (::cuda::std::size_t __i = 0; __i < __num_copies; ++__i)
  {
    ::cuda::__driver::__memcpyAsync(__dst_ptrs[__i], __src_ptrs[__i], __size_bytes, __stream.get());
  }
#endif // _CCCL_CTK_AT_LEAST(12, 9)
}

template <int _Pos = 0, class _EngineIn, class _LayoutIn, class _EngineOut, class _LayoutOut, class... _Indices>
_CCCL_HOST_API void __gather_ptrs(
  const ::cute::Tensor<_EngineIn, _LayoutIn>& __src,
  ::cute::Tensor<_EngineOut, _LayoutOut>& __dst,
  ::cuda::std::span<const void*> __src_ptrs,
  ::cuda::std::span<void*> __dst_ptrs,
  int& __count,
  _Indices... __indices) noexcept
{
  if constexpr (_Pos == _LayoutIn::rank - 1)
  {
    __src_ptrs[__count] = static_cast<const void*>(&__src(__indices..., 0));
    __dst_ptrs[__count] = static_cast<void*>(&__dst(__indices..., 0));
    __count++;
  }
  else if constexpr (_Pos < _LayoutIn::rank - 1)
  {
    for (::cuda::std::size_t __i = 0; __i < ::cute::size<_Pos>(__src); ++__i)
    {
      ::cuda::experimental::__gather_ptrs<_Pos + 1>(__src, __dst, __src_ptrs, __dst_ptrs, __count, __indices..., __i);
    }
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
  static_assert(::cuda::std::is_trivially_copyable_v<_TpIn>, "TpIn must be trivially copyable");
  static_assert(::cuda::std::is_trivially_copyable_v<_TpOut>, "TpOut must be trivially copyable");
  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>,
                "AccessorPolicyIn must be convertible to cuda::std::default_accessor");
  static_assert(::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>,
                "AccessorPolicyOut must be convertible to cuda::std::default_accessor");
  static_assert(!::cuda::std::is_const_v<_TpOut>, "TpOut must not be const");
  static_assert(::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>,
                "TpIn and TpOut must be the same type");
  if (__src.extents() != __dst.extents())
  {
    _CCCL_THROW(std::invalid_argument, "mdspan extents must be equal");
  }
  if (__src.size() == 0)
  {
    return;
  }
  if (__src.data_handle() == nullptr || __dst.data_handle() == nullptr)
  {
    _CCCL_THROW(std::invalid_argument, "mdspan data handle must not be nullptr");
  }
  const auto __src1 = ::cuda::experimental::to_cute(__src);
  const auto __dst1 = ::cuda::experimental::to_cute(__dst);
  const auto __src2 = ::cute::coalesce(__src1);
  const auto __dst2 = ::cute::coalesce(__dst1);
  // check compatibility of the two tensors
  if constexpr (::cute::rank(__src2) == 0)
  {
  }
  else
  {
    const auto __num_copies = ::cute::size(__src2) / ::cute::size<0>(__src2);
    ::std::vector<const void*> __src_ptrs(__num_copies);
    ::std::vector<void*> __dst_ptrs(__num_copies);
    int __count = 0;
    ::cuda::experimental::__gather_ptrs(__src2, __dst2, __src_ptrs, __dst_ptrs, __count);
    const auto __size_bytes = ::cute::size<0>(__src2) * sizeof(_TpIn);
    ::cuda::experimental::__batched_copy(__src_ptrs, __dst_ptrs, __size_bytes, __stream);
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_COPY_MDSPAN_TO_CUTE_H
