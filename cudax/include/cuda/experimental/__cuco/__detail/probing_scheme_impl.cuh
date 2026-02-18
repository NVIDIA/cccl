//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_PROBING_SCHEME_IMPL_CUH
#define _CUDAX___CUCO___DETAIL_PROBING_SCHEME_IMPL_CUH

#include <cuda/experimental/__cuco/__detail/utils.cuh>
#include <cuda/experimental/__cuco/probing_scheme.cuh>
#include <cuda/experimental/__cuco/traits.hpp>

#include <algorithm>
#include <cstdint>

namespace cuda::experimental::cuco::__detail
{
//! @brief Probing iterator class.
//!
//! Template parameter:
//! - `_Extent`: Extent type

template <class _Extent>
class __probing_iterator
{
public:
  using __extent_type = _Extent;
  using __size_type   = typename __extent_type::index_type;

  _CCCL_API constexpr __probing_iterator(__size_type __start, __size_type __step, __extent_type __upper_bound) noexcept
      : __curr_index{__start}
      , __step_size{__step}
      , __upper_bound{__upper_bound}
  {}

  _CCCL_API constexpr auto operator*() const noexcept
  {
    return __curr_index;
  }

  _CCCL_API constexpr auto operator++() noexcept
  {
    __curr_index = (__curr_index + __step_size) % __upper_bound.extent(0);
    return *this;
  }

  _CCCL_API constexpr auto operator++(int) noexcept
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

namespace cuda::experimental::cuco
{
template <int _CgSize, class _Hash>
template <int _BucketSize, class _ProbeKey, class _Extent>
_CCCL_API constexpr auto
linear_probing<_CgSize, _Hash>::make_iterator(_ProbeKey __probe_key, _Extent __upper_bound) const noexcept
{
  using __size_type        = typename _Extent::index_type;
  __size_type const __init = ::cuda::experimental::cuco::__detail::__sanitize_hash<__size_type>(__hash(__probe_key))
                           % (__upper_bound.extent(0) / _BucketSize) * _BucketSize;
  return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{
    __init, static_cast<__size_type>(_BucketSize), __upper_bound};
}

template <int _CgSize, class _Hash>
template <int _BucketSize, class _ProbeKey, class _Extent, class _ParentCG>
_CCCL_API constexpr auto linear_probing<_CgSize, _Hash>::make_iterator(
  ::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
  _ProbeKey __probe_key,
  _Extent __upper_bound) const noexcept
{
  using __size_type              = typename _Extent::index_type;
  constexpr __size_type __stride = cg_size * _BucketSize;
  __size_type const __init =
    ::cuda::experimental::cuco::__detail::__sanitize_hash<__size_type>(__hash(__probe_key))
      % (__upper_bound.extent(0) / __stride) * __stride
    + static_cast<__size_type>(__group.thread_rank() * _BucketSize);
  return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{__init, __stride, __upper_bound};
}

template <int _CgSize, class _Hash1, class _Hash2>
template <int _BucketSize, class _ProbeKey, class _Extent>
_CCCL_API constexpr auto
double_hashing<_CgSize, _Hash1, _Hash2>::make_iterator(_ProbeKey __probe_key, _Extent __upper_bound) const noexcept
{
  using __size_type = typename _Extent::index_type;
  return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{
    static_cast<__size_type>(::cuda::experimental::cuco::__detail::__sanitize_hash<__size_type>(__hash1(__probe_key))
                             % (__upper_bound.extent(0) / _BucketSize) * _BucketSize),
    static_cast<__size_type>(
      (::cuda::experimental::cuco::__detail::__sanitize_hash<__size_type>(__hash2(__probe_key))
         % (__upper_bound.extent(0) / _BucketSize - 1)
       + 1)
      * _BucketSize),
    __upper_bound};
}

template <int _CgSize, class _Hash1, class _Hash2>
template <int _BucketSize, class _ProbeKey, class _Extent, class _ParentCG>
_CCCL_API constexpr auto double_hashing<_CgSize, _Hash1, _Hash2>::make_iterator(
  ::cooperative_groups::thread_block_tile<cg_size, _ParentCG> __group,
  _ProbeKey __probe_key,
  _Extent __upper_bound) const noexcept
{
  using __size_type              = typename _Extent::index_type;
  constexpr __size_type __stride = cg_size * _BucketSize;

  return ::cuda::experimental::cuco::__detail::__probing_iterator<_Extent>{
    static_cast<__size_type>(::cuda::experimental::cuco::__detail::__sanitize_hash<__size_type>(__hash1(__probe_key))
                               % (__upper_bound.extent(0) / __stride) * __stride
                             + static_cast<__size_type>(__group.thread_rank() * _BucketSize)),
    static_cast<__size_type>(
      (::cuda::experimental::cuco::__detail::__sanitize_hash<__size_type>(__hash2(__probe_key))
         % (__upper_bound.extent(0) / __stride - 1)
       + 1)
      * __stride),
    __upper_bound};
}
} // namespace cuda::experimental::cuco

#endif // _CUDAX___CUCO___DETAIL_PROBING_SCHEME_IMPL_CUH
