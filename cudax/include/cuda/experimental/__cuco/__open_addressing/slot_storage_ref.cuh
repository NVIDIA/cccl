//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___OPEN_ADDRESSING_SLOT_STORAGE_REF_CUH
#define _CUDAX___CUCO___OPEN_ADDRESSING_SLOT_STORAGE_REF_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Lightweight non-owning reference to a contiguous slot array with bucket abstraction.
//!
//! Provides indexing into the slot array organized as buckets. The probing scheme produces
//! bucket indices; within each bucket there are `_BucketSize` value-typed slots.
//!
//! @tparam _Value The slot value type (e.g. `::cuda::std::pair<Key, T>`)
//! @tparam _BucketSize Number of slots per bucket (compile-time constant)
template <class _Value, int _BucketSize>
struct __slot_storage_ref
{
  using __value_type     = _Value;
  using __size_type      = ::cuda::std::size_t;
  using __extent_type    = ::cuda::std::extents<__size_type, ::cuda::std::dynamic_extent>;
  using __iterator       = _Value*;
  using __const_iterator = const _Value*;
  using __bucket_type    = ::cuda::std::span<_Value, _BucketSize>;

  static constexpr int __bucket_size = _BucketSize;

  _Value* __data_;
  __extent_type __num_buckets_;

  //! @brief Constructs a slot storage ref.
  //!
  //! @param __data Pointer to the first slot
  //! @param __num_buckets Number of buckets (total slots = __num_buckets * _BucketSize)
  _CCCL_HOST_DEVICE constexpr __slot_storage_ref(_Value* __data, __size_type __num_buckets) noexcept
      : __data_{__data}
      , __num_buckets_{__num_buckets}
  {}

  //! @brief Returns the bucket at position `__i`.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __bucket_type operator[](__size_type __i) const noexcept
  {
    return __bucket_type{__data_ + __i * _BucketSize, static_cast<typename __bucket_type::size_type>(_BucketSize)};
  }

  //! @brief Returns the total number of slots.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __size_type capacity() const noexcept
  {
    return static_cast<__size_type>(__num_buckets_.extent(0)) * _BucketSize;
  }

  //! @brief Returns the number of buckets, as an extent object.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __extent_type extent() const noexcept
  {
    return __num_buckets_;
  }

  //! @brief Returns the number of buckets as a raw value.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __size_type num_buckets() const noexcept
  {
    return static_cast<__size_type>(__num_buckets_.extent(0));
  }

  //! @brief Returns a pointer to the underlying slot array.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr _Value* data() const noexcept
  {
    return __data_;
  }

  //! @brief Returns an iterator to the first slot.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __iterator begin() const noexcept
  {
    return __data_;
  }

  //! @brief Returns an iterator to one past the last slot.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __iterator end() const noexcept
  {
    return __data_ + capacity();
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___OPEN_ADDRESSING_SLOT_STORAGE_REF_CUH
