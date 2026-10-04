//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_FUNCTORS_CUH
#define _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_FUNCTORS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tie.h>

#include <cuda/experimental/__cuco/detail/bitwise_compare.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__open_addressing
{
//! @brief Device functor returning the content of a slot.
//!
//! @tparam _HasPayload Whether the slot contains a mapped value
//! @tparam _StorageRef Slot storage reference type
template <bool _HasPayload, class _StorageRef>
struct __get_slot
{
  _StorageRef __storage_ref_;

  //! @brief Constructs a slot accessor.
  //!
  //! @param __storage_ref Slot storage reference
  _CCCL_HOST_DEVICE_API explicit constexpr __get_slot(_StorageRef __storage_ref) noexcept
      : __storage_ref_{__storage_ref}
  {}

  //! @brief Returns the content of the slot at the given index.
  //!
  //! @param __idx Slot index
  //!
  //! @return Slot content
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto operator()(typename _StorageRef::__size_type __idx) const noexcept
  {
    const auto& __slot = *(__storage_ref_.data() + __idx);
    if constexpr (_HasPayload)
    {
      return ::cuda::std::make_tuple(__slot.first, __slot.second);
    }
    else
    {
      return __slot;
    }
  }
};

//! @brief Device predicate indicating whether a slot is filled.
//!
//! @tparam _HasPayload Whether the slot contains a mapped value
//! @tparam _Key Key type
template <bool _HasPayload, class _Key>
struct __slot_is_filled
{
  _Key __empty_key_sentinel_;
  _Key __erased_key_sentinel_;

  //! @brief Constructs a filled-slot predicate.
  //!
  //! @param __empty_key_sentinel Empty key sentinel
  //! @param __erased_key_sentinel Erased key sentinel
  _CCCL_HOST_DEVICE_API explicit constexpr __slot_is_filled(
    _Key __empty_key_sentinel, _Key __erased_key_sentinel) noexcept
      : __empty_key_sentinel_{__empty_key_sentinel}
      , __erased_key_sentinel_{__erased_key_sentinel}
  {}

  //! @brief Indicates whether the given slot is filled.
  //!
  //! @tparam _Slot Slot type
  //!
  //! @param __slot Slot to inspect
  //!
  //! @return `true` if the slot contains an element
  template <class _Slot>
  [[nodiscard]] _CCCL_DEVICE_API constexpr bool operator()(const _Slot& __slot) const noexcept
  {
    if constexpr (_HasPayload)
    {
      return __is_filled(::cuda::std::get<0>(__slot));
    }
    else
    {
      return __is_filled(__slot);
    }
  }

private:
  [[nodiscard]] _CCCL_DEVICE_API constexpr bool __is_filled(const _Key& __key) const noexcept
  {
    return !detail::__bitwise_compare(__key, __empty_key_sentinel_)
        && !detail::__bitwise_compare(__key, __erased_key_sentinel_);
  }
};
} // namespace cuda::experimental::cuco::__open_addressing

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO_DETAIL_OPEN_ADDRESSING_FUNCTORS_CUH
