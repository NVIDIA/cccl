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
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Lightweight non-owning reference to a contiguous slot array with bucket abstraction.
//!
//! Provides indexing into the slot array organized as buckets; within each bucket there are
//! `bucket_size` value-typed slots. The total slot count is carried by a `cuco::valid_capacity`
//! descriptor, so a static capacity folds the probing reduction to a constant while a dynamic
//! capacity stores the slot count. The probing layer works in slot offsets bounded by `capacity()`.
//!
//! @tparam _Value The slot value type (e.g. `::cuda::std::pair<Key, T>`)
//! @tparam _CapacityDescriptor The `cuco::valid_capacity` descriptor type carrying the slot count
template <class _Value, class _CapacityDescriptor>
struct __slot_storage_ref
{
  using __size_type           = ::cuda::std::size_t;
  using __value_type          = _Value;
  using __capacity_descriptor = _CapacityDescriptor;
  using __iterator            = _Value*;
  using __const_iterator      = const _Value*;

  static constexpr int __bucket_size = _CapacityDescriptor::bucket_size;

  using __bucket_type = ::cuda::std::span<_Value, __bucket_size>;

  static_assert(__bucket_size > 0, "bucket size must be greater than zero");

  _Value* __data_;
  _CCCL_NO_UNIQUE_ADDRESS __capacity_descriptor __capacity_;

  //! @brief Constructs a slot storage ref.
  //!
  //! @param __data Pointer to the first slot
  //! @param __capacity Validated capacity descriptor (total slots = `__capacity.capacity()`)
  _CCCL_HOST_DEVICE constexpr __slot_storage_ref(_Value* __data, __capacity_descriptor __capacity) noexcept
      : __data_{__data}
      , __capacity_{__capacity}
  {}

  //! @brief Returns the bucket at position `__i`.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __bucket_type operator[](__size_type __i) const noexcept
  {
    return __bucket_type{__data_ + __i * __bucket_size, static_cast<typename __bucket_type::size_type>(__bucket_size)};
  }

  //! @brief Returns the total number of slots.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __size_type capacity() const noexcept
  {
    return __capacity_.capacity();
  }

  //! @brief Returns the number of buckets.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __size_type num_buckets() const noexcept
  {
    return __capacity_.num_buckets();
  }

  //! @brief Returns the validated capacity descriptor (used as the probing reduction bound).
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr __capacity_descriptor capacity_descriptor() const noexcept
  {
    return __capacity_;
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
