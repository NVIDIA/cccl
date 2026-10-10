//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_SLOT_STORAGE_REF_CUH
#define _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_SLOT_STORAGE_REF_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/is_aligned.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Lightweight non-owning reference to a contiguous slot array with bucket abstraction.
//!
//! Provides indexing into the slot array organized as buckets; within each bucket there are
//! `_BucketSize` value-typed slots. The total slot count is carried as a `cuda::std::extents`, so a
//! static `_Capacity` folds the probing reduction to a constant while a dynamic `_Capacity` stores
//! the slot count. The probing layer works in slot offsets bounded by `capacity()`.
//!
//! @tparam _Value The slot value type (e.g. `::cuda::std::pair<Key, T>`)
//! @tparam _BucketSize Number of slots per bucket (compile-time constant)
//! @tparam _Capacity Valid total slot count, or `cuda::std::dynamic_extent` for runtime sizing
template <class _Value, int _BucketSize, ::cuda::std::size_t _Capacity = ::cuda::std::dynamic_extent>
struct __slot_storage_ref
{
  using __size_type            = ::cuda::std::size_t;
  using __value_type           = _Value;
  using __capacity_extent_type = ::cuda::std::extents<__size_type, _Capacity>;
  using __iterator             = _Value*;
  using __const_iterator       = const _Value*;

  static constexpr int __bucket_size = _BucketSize;

  using __bucket_type = ::cuda::std::span<_Value, _BucketSize>;

  static_assert(_BucketSize > 0, "bucket size must be greater than zero");
  static_assert(_Capacity == ::cuda::std::dynamic_extent || _Capacity % _BucketSize == 0,
                "static capacity must be divisible by the bucket size");

  _Value* __data_;
  _CCCL_NO_UNIQUE_ADDRESS __capacity_extent_type __capacity_;

  //! @brief Constructs a slot storage ref.
  //!
  //! @param __data Pointer to the first slot
  //! @param __capacity Total slot count (must equal the static `_Capacity` when it is static)
  _CCCL_HOST_DEVICE_API constexpr __slot_storage_ref(_Value* __data, __size_type __capacity) noexcept
      : __data_{__data}
      , __capacity_{__capacity}
  {}

  //! @brief Returns the bucket at position `__i`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __bucket_type operator[](__size_type __i) const noexcept
  {
    return __bucket_type{__data_ + __i, typename __bucket_type::size_type{_BucketSize}};
  }

  //! @brief Returns the total number of slots.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __size_type capacity() const noexcept
  {
    return __capacity_.extent(0);
  }

  //! @brief Returns the number of buckets.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __size_type num_buckets() const noexcept
  {
    return capacity() / __size_type{_BucketSize};
  }

  //! @brief Returns the total slot count as a `cuda::std::extents` (the probing reduction bound).
  //!
  //! Returning the extent rather than a plain size keeps the static slot count in the type, so the
  //! probing iterator's modular reduction folds to a constant for static `_Capacity`.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __capacity_extent_type capacity_extent() const noexcept
  {
    return __capacity_;
  }

  //! @brief Returns a pointer to the underlying slot array.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Value* data() const noexcept
  {
    return __data_;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API bool __is_packed_cas_aligned() const noexcept
  {
    return ::cuda::is_aligned(__data_, sizeof(_Value));
  }

  //! @brief Returns an iterator to the first slot.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __iterator begin() const noexcept
  {
    return __data_;
  }

  //! @brief Returns an iterator to one past the last slot.
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __iterator end() const noexcept
  {
    return __data_ + capacity();
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_SLOT_STORAGE_REF_CUH
