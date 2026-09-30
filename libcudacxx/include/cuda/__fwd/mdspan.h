//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FWD_MDSPAN_H
#define _CUDA___FWD_MDSPAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Class to describe the strides of a multi-dimensional array layout.
//!
//! Similar to extents, but for strides. Supports both static (compile-time known)
//! and dynamic (runtime) stride values. Uses dynamic_stride as the tag for dynamic values.
//!
//! @tparam _OffsetType The signed integer type for stride values (supports negative strides)
//! @tparam _Strides... The stride values, where dynamic_stride indicates a runtime value
template <class _OffsetType, ::cuda::std::ptrdiff_t... _Strides>
class strides;

//! @brief Tag value indicating a dynamic stride (similar to dynamic_extent for extents)
inline constexpr ::cuda::std::ptrdiff_t dynamic_stride = (::cuda::std::numeric_limits<::cuda::std::ptrdiff_t>::min)();

namespace __strides_detail
{
template <class _OffsetType, class _Seq>
struct __make_dstrides_impl;

template <class _OffsetType, ::cuda::std::ptrdiff_t... _Idx>
struct __make_dstrides_impl<_OffsetType, ::cuda::std::integer_sequence<::cuda::std::ptrdiff_t, _Idx...>>
{
  using type = strides<_OffsetType, ((void) _Idx, dynamic_stride)...>;
};
} // namespace __strides_detail

//! @brief Alias template for strides with all dynamic stride values
template <class _OffsetType, ::cuda::std::size_t _Rank>
using dstrides = typename __strides_detail::
  __make_dstrides_impl<_OffsetType, ::cuda::std::make_integer_sequence<::cuda::std::ptrdiff_t, _Rank>>::type;

template <::cuda::std::size_t _Rank, class _OffsetType = ::cuda::std::ptrdiff_t>
using steps = dstrides<_OffsetType, _Rank>;

template <class _Tp>
inline constexpr bool __is_cuda_strides_v = false;

template <class _OffsetType, ::cuda::std::ptrdiff_t... _Strides>
inline constexpr bool __is_cuda_strides_v<strides<_OffsetType, _Strides...>> = true;

//! @brief Layout policy with relaxed stride mapping that supports negative strides and offsets.
//!
//! Unlike `layout_stride`, this layout allows:
//! - Negative strides (for reverse iteration)
//! - Zero strides (for broadcasting)
//! - A base offset (to accommodate negative strides)
//!
//! @note This layout is NOT always unique, exhaustive, or strided in the standard sense.
struct layout_stride_relaxed
{
  template <class _Extents,
            class _Strides    = dstrides<::cuda::std::make_signed_t<typename _Extents::index_type>, _Extents::rank()>,
            class _OffsetType = ::cuda::std::ptrdiff_t>
  class mapping;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FWD_MDSPAN_H
