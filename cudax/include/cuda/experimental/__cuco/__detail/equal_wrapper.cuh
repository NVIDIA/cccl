//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX___CUCO___DETAIL_EQUAL_WRAPPER_CUH
#define _CUDAX___CUCO___DETAIL_EQUAL_WRAPPER_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/__detail/bitwise_compare.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental::cuco::__detail
{
//! @brief Enum of equality comparison results.
//!
//! NOTE: Enum values are order-sensitive.
enum class __equal_result : ::cuda::std::int8_t
{
  __unequal   = 0,
  __equal     = 1,
  __empty     = 2,
  __available = 3
};

//! @brief Enum indicating whether the operation is an insert.
enum class __is_insert : ::cuda::std::int8_t
{
  __yes,
  __no
};

//! @brief Key equality wrapper.
//!
//! @tparam _Tp Right-hand side element type
//! @tparam _Equal Equality callable
//! @tparam _AllowsDuplicates Duplicate key flag
template <class _Tp, class _Equal, bool _AllowsDuplicates>
struct __equal_wrapper
{
  _Tp __empty_sentinel;
  _Tp __erased_sentinel;
  _Equal __equal;

  //! @brief Equality wrapper constructor.
  //!
  //! @param __empty Empty sentinel value
  //! @param __erased Erased sentinel value
  //! @param __eq Equality binary callable
  _CCCL_API constexpr __equal_wrapper(_Tp __empty, _Tp __erased, _Equal const& __eq) noexcept
      : __empty_sentinel{__empty}
      , __erased_sentinel{__erased}
      , __equal{__eq}
  {}

  //! @brief Equality check with the given equality callable.
  //!
  //! @tparam _Lhs Left-hand side element type
  //! @tparam _Rhs Right-hand side element type
  //!
  //! @param __lhs Left-hand side element to check equality
  //! @param __rhs Right-hand side element to check equality
  //!
  //! @return `__equal` if `__lhs` and `__rhs` are equivalent, `__unequal` otherwise
  template <class _Lhs, class _Rhs>
  _CCCL_DEVICE constexpr __equal_result __equal_to(const _Lhs& __lhs, const _Rhs& __rhs) const noexcept
  {
    return __equal(__lhs, __rhs) ? __equal_result::__equal : __equal_result::__unequal;
  }

  //! @brief Order-sensitive equality operator.
  //!
  //! @note This function always compares the right-hand side element against sentinel values first
  //! then performs an equality check with the given `__equal` callable, i.e., `__equal(__lhs, __rhs)`.
  //! @note Container (like set or map) slots MUST always be on the right-hand side.
  //!
  //! @tparam _IsInsert Flag indicating whether it's an insert equality check or not. Insert probing
  //! stops when it's an empty or erased slot while query probing stops only when it's empty.
  //! @tparam _Lhs Left-hand side element type
  //! @tparam _Rhs Right-hand side element type
  //!
  //! @param __lhs Left-hand side element to check equality
  //! @param __rhs Right-hand side element to check equality
  //!
  //! @return Three-way equality comparison result
  template <__is_insert _IsInsert, class _Lhs, class _Rhs>
  _CCCL_DEVICE constexpr __equal_result operator()(const _Lhs& __lhs, const _Rhs& __rhs) const noexcept
  {
    if constexpr (_IsInsert == __is_insert::__yes)
    {
      if (__detail::__bitwise_compare(__rhs, __empty_sentinel) || __detail::__bitwise_compare(__rhs, __erased_sentinel))
      {
        return __equal_result::__available;
      }
      if constexpr (_AllowsDuplicates)
      {
        return __equal_result::__unequal;
      }
      else
      {
        return this->__equal_to(__lhs, __rhs);
      }
    }
    else
    {
      return __detail::__bitwise_compare(__rhs, __empty_sentinel)
             ? __equal_result::__empty
             : this->__equal_to(__lhs, __rhs);
    }
  }
};
} // namespace cuda::experimental::cuco::__detail

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX___CUCO___DETAIL_EQUAL_WRAPPER_CUH
