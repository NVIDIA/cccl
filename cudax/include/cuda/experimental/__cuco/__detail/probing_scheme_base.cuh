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
//! @tparam _Extent Extent type
template <class _Extent>
class __probing_iterator
{
public:
  using __extent_type = _Extent;
  using __size_type   = typename __extent_type::index_type;

  _CCCL_HOST_DEVICE_API constexpr __probing_iterator(
    __size_type __start, __size_type __step, __extent_type __upper_bound) noexcept
      : __curr_index{__start}
      , __step_size{__step}
      , __upper_bound{__upper_bound}
  {}

  _CCCL_HOST_DEVICE_API constexpr auto operator*() const noexcept
  {
    return __curr_index;
  }

  _CCCL_HOST_DEVICE_API constexpr auto operator++() noexcept
  {
    __curr_index = (__curr_index + __step_size) % __upper_bound.extent(0);
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
  __extent_type __upper_bound;
};
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_PROBING_SCHEME_BASE_CUH
