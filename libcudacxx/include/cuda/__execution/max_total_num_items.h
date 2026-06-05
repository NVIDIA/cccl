//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_MAX_TOTAL_NUM_ITEMS_H
#define __CUDA___EXECUTION_MAX_TOTAL_NUM_ITEMS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__execution/guarantee.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/limits>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

//! @brief Guarantee describing an upper bound on the total number of items processed by an algorithm (e.g. the combined
//! size of all segments handled by cub::DeviceBatchedTopK).
//!
//! The bound is carried as an integral value whose type is inferred from the argument; that type distinguishes, for
//! example, a 32-bit from a 64-bit bound and lets algorithms size intermediate offset types accordingly. The bound can
//! be expressed as a compile-time bound (@c static_highest), a runtime bound (@c highest()), or both. A composable
//! @c min_total_num_items lower-bound guarantee may be added in the future.
struct __get_max_total_num_items_t;

template <class _Tp, _Tp _StaticHighest>
struct _CCCL_DECLSPEC_EMPTY_BASES __max_total_num_items_holder_t : __guarantee
{
  static_assert(::cuda::std::is_integral_v<_Tp>, "max_total_num_items requires an integral bound type");

  using element_type = _Tp;

  static constexpr element_type static_highest = _StaticHighest;

  element_type __highest_;

  //! @brief Returns the effective (runtime) upper bound on the total number of items.
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto highest() const noexcept -> element_type
  {
    return __highest_;
  }

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_max_total_num_items_t&) const noexcept
    -> const __max_total_num_items_holder_t&
  {
    return *this;
  }
};

struct __get_max_total_num_items_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_max_total_num_items_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)));
    return __env.query(*this);
  }

  [[nodiscard]]
  _CCCL_NODEBUG_API static constexpr auto query(::cuda::std::execution::forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT auto __get_max_total_num_items = __get_max_total_num_items_t{};

//! @brief Creates a guarantee with a compile-time upper bound on the total number of items.
//!
//! The bound type is inferred from the non-type template parameter, which must be integral.
//!
//! @tparam _Highest Compile-time upper bound on the total number of items.
//! @return A guarantee that can be passed to @c cuda::execution::guarantee.
template <auto _Highest>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto max_total_num_items() noexcept
  -> __max_total_num_items_holder_t<decltype(_Highest), _Highest>
{
  static_assert(::cuda::std::is_integral_v<decltype(_Highest)>, "max_total_num_items requires an integral bound");
  return __max_total_num_items_holder_t<decltype(_Highest), _Highest>{{}, _Highest};
}

//! @brief Creates a guarantee with a runtime upper bound on the total number of items.
//!
//! The bound type is inferred from the argument, which must be integral. The compile-time bound spans the whole type.
//!
//! @param __highest Runtime upper bound on the total number of items.
//! @return A guarantee that can be passed to @c cuda::execution::guarantee.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(::cuda::std::is_integral_v<_Tp>)
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto max_total_num_items(_Tp __highest) noexcept
  -> __max_total_num_items_holder_t<_Tp, (::cuda::std::numeric_limits<_Tp>::max)()>
{
  return __max_total_num_items_holder_t<_Tp, (::cuda::std::numeric_limits<_Tp>::max)()>{{}, __highest};
}

//! @brief Creates a guarantee with both a compile-time and a runtime upper bound on the total number of items.
//!
//! The bound type is inferred from the non-type template parameter. The runtime bound must not exceed the compile-time
//! bound.
//!
//! @tparam _Highest Compile-time upper bound on the total number of items.
//! @param __highest Runtime upper bound on the total number of items, must be `<= _Highest`.
//! @return A guarantee that can be passed to @c cuda::execution::guarantee.
template <auto _Highest, class _Tp>
[[nodiscard]] _CCCL_NODEBUG_API constexpr auto max_total_num_items(_Tp __highest) noexcept
  -> __max_total_num_items_holder_t<decltype(_Highest), _Highest>
{
  static_assert(::cuda::std::is_integral_v<decltype(_Highest)>,
                "max_total_num_items requires an integral static bound");
  static_assert(::cuda::std::is_integral_v<_Tp>, "max_total_num_items requires an integral runtime bound");
  _CCCL_ASSERT(::cuda::std::cmp_less_equal(__highest, _Highest),
               "max_total_num_items: the runtime bound must not exceed the static bound");
  return __max_total_num_items_holder_t<decltype(_Highest), _Highest>{{}, static_cast<decltype(_Highest)>(__highest)};
}

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_MAX_TOTAL_NUM_ITEMS_H
