//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_PROBING_SCHEME_BASE_CUH
#define _CUDAX___CUCO___DETAIL_PROBING_SCHEME_BASE_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Base class of public probing schemes.
//!
//! @tparam _CgSize Cooperative group size
template <int _CgSize>
class __probing_scheme_base
{
public:
  static constexpr int __cg_size = _CgSize;
};

//! @brief Probing iterator class.
//!
//! Yields slot offsets and wraps modulo the total capacity (in slots). The capacity is held as a
//! `cuda::std::extents` so a static slot count folds the reduction to a constant.
//!
//! @tparam _CapacityExtent Capacity extent type (total slots), a `cuda::std::extents`
template <class _CapacityExtent>
class __probing_iterator
{
public:
  using __capacity_extent_type = _CapacityExtent;
  using __size_type            = typename _CapacityExtent::index_type;

  _CCCL_HOST_DEVICE_API constexpr __probing_iterator(
    __size_type __start, __size_type __step, _CapacityExtent __capacity) noexcept
      : __curr_index{__start}
      , __step_size{__step}
      , __capacity_{__capacity}
  {}

  _CCCL_HOST_DEVICE_API constexpr auto operator*() const noexcept
  {
    return __curr_index;
  }

  _CCCL_HOST_DEVICE_API constexpr auto operator++() noexcept
  {
    __curr_index = (__curr_index + __step_size) % __capacity_.extent(0);
    return *this;
  }

  _CCCL_HOST_DEVICE_API constexpr auto operator++(int) noexcept
  {
    auto __temp = *this;
    ++(*this);
    return __temp;
  }

private:
  __size_type __curr_index;
  __size_type __step_size;
  _CCCL_NO_UNIQUE_ADDRESS _CapacityExtent __capacity_;
};
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_PROBING_SCHEME_BASE_CUH
