//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_TIE_BREAK_H
#define __CUDA___EXECUTION_TIE_BREAK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__execution/require.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_one_of.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

//! @brief Requirement describing how an algorithm breaks ties among elements that compare equal at its selection
//! boundary (e.g. the K-th element of cub::DeviceBatchedTopK).
//!
//! A tie-break requirement only constrains *which* of the equal-comparing elements end up in the result set; it says
//! nothing about the order in which the results are written (that is controlled independently by
//! cuda::execution::output_ordering). A tie-break is only meaningful together with a deterministic execution
//! requirement (cuda::execution::determinism::run_to_run or gpu_to_gpu).
namespace tie_break
{
struct __get_tie_break_t;

enum class __tie_break_t
{
  __unspecified, //!< Any (implementation-defined) deterministic tie-break is acceptable.
  __prefer_smaller_index, //!< Among elements that compare equal, prefer the one(s) with the smaller source index.
  __prefer_larger_index //!< Among elements that compare equal, prefer the one(s) with the larger source index.
};

template <__tie_break_t _Preference>
struct __tie_break_holder_t : __requirement
{
  static constexpr __tie_break_t value = _Preference;

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_tie_break_t&) const noexcept
    -> __tie_break_holder_t<_Preference>
  {
    return *this;
  }
};

using unspecified_t          = __tie_break_holder_t<__tie_break_t::__unspecified>;
using prefer_smaller_index_t = __tie_break_holder_t<__tie_break_t::__prefer_smaller_index>;
using prefer_larger_index_t  = __tie_break_holder_t<__tie_break_t::__prefer_larger_index>;

_CCCL_GLOBAL_CONSTANT unspecified_t unspecified{};
_CCCL_GLOBAL_CONSTANT prefer_smaller_index_t prefer_smaller_index{};
_CCCL_GLOBAL_CONSTANT prefer_larger_index_t prefer_larger_index{};

struct __get_tie_break_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_tie_break_t>)
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

_CCCL_GLOBAL_CONSTANT auto __get_tie_break = __get_tie_break_t{};
} // namespace tie_break

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_TIE_BREAK_H
