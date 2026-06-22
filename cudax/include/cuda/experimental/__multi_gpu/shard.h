//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_SHARD_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_SHARD_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class __shard_base
{};

template <class _Iterator>
class shard : public __shard_base
{
public:
  using iterator_type = _Iterator;
  using size_type     = ::cuda::std::make_unsigned_t<::cuda::std::iter_difference_t<iterator_type>>;

  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
  shard(_Range&& __range, logical_device __dev)
      : shard{::cuda::std::ranges::begin(__range), ::cuda::std::ranges::end(__range), __dev}
  {}

  _CCCL_TEMPLATE(class _Range)
  _CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
  shard(_Range&& __range, device_ref __dev)
      : shard{::cuda::std::forward<_Range>(__range), logical_device{__dev}}
  {}

  shard(iterator_type __b, iterator_type __e, device_ref __dev)
      : shard{__b, __e, logical_device{__dev}}
  {}

  shard(iterator_type __b, iterator_type __e, logical_device __dev)
      : __begin_{__b}
      , __end_{__e}
      , __device_{__dev}
  {}

  [[nodiscard]] constexpr iterator_type begin() const noexcept
  {
    return __begin_;
  }

  [[nodiscard]] constexpr iterator_type end() const noexcept
  {
    return __end_;
  }

  [[nodiscard]] constexpr size_type size() const noexcept
  {
    return static_cast<size_type>(::cuda::std::ranges::distance(begin(), end()));
  }

  [[nodiscard]] constexpr const logical_device& device() const noexcept
  {
    return __device_;
  }

private:
  iterator_type __begin_;
  iterator_type __end_;
  logical_device __device_;
};

template <class _Iter>
shard(_Iter, _Iter, device_ref) -> shard<_Iter>;

template <class _Iter>
shard(_Iter, _Iter, logical_device) -> shard<_Iter>;

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
shard(_Range&&, logical_device) -> shard<::cuda::std::ranges::iterator_t<_Range>>;

_CCCL_TEMPLATE(class _Range)
_CCCL_REQUIRES(::cuda::std::ranges::range<_Range>)
shard(_Range&&, device_ref) -> shard<::cuda::std::ranges::iterator_t<_Range>>;
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_SHARD_H
