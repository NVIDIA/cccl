//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_DETERMINISM_H
#define __CUDA___EXECUTION_DETERMINISM_H

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
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_one_of.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

namespace determinism
{
struct __get_determinism_t;

enum class __determinism_t
{
  __not_guaranteed,
  __run_to_run,
  __gpu_to_gpu
};

//! @brief Tie-break preference that can be attached to a deterministic guarantee. It selects *which* of the elements
//! that compare equal at an algorithm's selection boundary are kept (e.g. at the K-th element of
//! cub::DeviceBatchedTopK). It has no effect on the output order (that is controlled independently by
//! cuda::execution::output_ordering) and is only valid on a deterministic guarantee (run_to_run / gpu_to_gpu).
enum class __tie_break_t
{
  __unspecified, //!< Any (implementation-defined) deterministic tie-break is acceptable.
  __prefer_smaller_index, //!< Among elements that compare equal, prefer the one(s) with the smaller source index.
  __prefer_larger_index //!< Among elements that compare equal, prefer the one(s) with the larger source index.
};

//! @brief Tag selecting a tie-break preference. These are intentionally *not* requirements on their own: a tie-break is
//! only meaningful when attached to a deterministic guarantee, e.g.
//! `determinism::run_to_run(determinism::tie_break::prefer_smaller_index)`. This makes it impossible to request a
//! tie-break without also requesting determinism.
template <__tie_break_t _Preference>
struct __tie_break_holder_t : ::cuda::std::integral_constant<__tie_break_t, _Preference>
{};

namespace tie_break
{
using unspecified_t          = __tie_break_holder_t<__tie_break_t::__unspecified>;
using prefer_smaller_index_t = __tie_break_holder_t<__tie_break_t::__prefer_smaller_index>;
using prefer_larger_index_t  = __tie_break_holder_t<__tie_break_t::__prefer_larger_index>;

_CCCL_GLOBAL_CONSTANT unspecified_t unspecified{};
_CCCL_GLOBAL_CONSTANT prefer_smaller_index_t prefer_smaller_index{};
_CCCL_GLOBAL_CONSTANT prefer_larger_index_t prefer_larger_index{};
} // namespace tie_break

template <__determinism_t _Guarantee, __tie_break_t _TieBreak = __tie_break_t::__unspecified>
struct __determinism_holder_t : __requirement
{
  static constexpr __determinism_t value   = _Guarantee;
  static constexpr __tie_break_t tie_break = _TieBreak;

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_determinism_t&) const noexcept
    -> __determinism_holder_t<_Guarantee, _TieBreak>
  {
    return *this;
  }

  //! @brief Attaches a tie-break preference to this (deterministic) guarantee.
  template <__tie_break_t _Preference>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(__tie_break_holder_t<_Preference>) const noexcept
    -> __determinism_holder_t<_Guarantee, _Preference>
  {
    static_assert(_Guarantee != __determinism_t::__not_guaranteed,
                  "A tie-break can only be attached to a deterministic guarantee "
                  "(cuda::execution::determinism::run_to_run or gpu_to_gpu), not to not_guaranteed.");
    return {};
  }
};

using gpu_to_gpu_t     = __determinism_holder_t<__determinism_t::__gpu_to_gpu>;
using run_to_run_t     = __determinism_holder_t<__determinism_t::__run_to_run>;
using not_guaranteed_t = __determinism_holder_t<__determinism_t::__not_guaranteed>;

_CCCL_GLOBAL_CONSTANT gpu_to_gpu_t gpu_to_gpu{};
_CCCL_GLOBAL_CONSTANT run_to_run_t run_to_run{};
_CCCL_GLOBAL_CONSTANT not_guaranteed_t not_guaranteed{};

//! @brief Compile-time guard for algorithms that do not support a tie-break preference. Routing the requested
//! determinism type through this alias yields it unchanged when no tie-break is attached, and otherwise fails with a
//! single, shared diagnostic instead of a deep template error. Algorithms that do honor a tie-break (e.g. batched
//! top-k) simply do not route their determinism through this alias.
template <class _DeterminismT>
struct __no_tie_break_guard
{
  static_assert(_DeterminismT::tie_break == __tie_break_t::__unspecified,
                "This algorithm does not support a tie-break preference. The cuda::execution::determinism guarantee "
                "passed via "
                "cuda::execution::require(...) must not carry a tie-break (cuda::execution::determinism::tie_break::"
                "prefer_smaller_index / prefer_larger_index); only algorithms that explicitly document tie-break "
                "support honor it.");
  using type = _DeterminismT;
};

template <class _DeterminismT>
using __validate_no_tie_break_t = typename __no_tie_break_guard<_DeterminismT>::type;

struct __get_determinism_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_determinism_t>)
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

_CCCL_GLOBAL_CONSTANT auto __get_determinism = __get_determinism_t{};
} // namespace determinism

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_DETERMINISM_H
