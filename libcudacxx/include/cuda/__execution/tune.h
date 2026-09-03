//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_TUNE_H
#define __CUDA___EXECUTION_TUNE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/semiregular.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/is_empty.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_EXECUTION

struct __get_tuning_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_tuning_t>)
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

_CCCL_GLOBAL_CONSTANT auto __get_tuning = __get_tuning_t{};

struct converts_to_anything
{
  template <typename T>
  operator T() const
  {
    return {};
  };
};

template <class _PolicySelector>
_CCCL_NODEBUG_API constexpr auto __detect_policy_return_type()
{
  if constexpr (::cuda::std::is_invocable_v<_PolicySelector, ::cuda::compute_capability>)
  {
    return decltype(_PolicySelector{}(::cuda::compute_capability{})){};
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<_PolicySelector, ::cuda::compute_capability, converts_to_anything>,
                  "Policy selectors must be invocable with cuda::compute_capability and an optional policy");

    using ret_t = decltype(_PolicySelector{}(::cuda::compute_capability{}, converts_to_anything{}));
    static_assert(!::cuda::std::is_same_v<::cuda::std::remove_cvref_t<ret_t>, converts_to_anything>,
                  "Policy selectors must return a concrete policy type");
    static_assert(::cuda::std::is_invocable_v<_PolicySelector, ::cuda::compute_capability, ret_t>,
                  "Policy selectors with more than one argument must accept the returned policy type as second "
                  "argument");
    return ret_t{};
  }
}

//! @rst
//! Creates an environment from a pack of policy selectors that can be passed to device-wide parallel algorithms to
//! select tunings for different target architectures. See the :ref:`policy selector documentation
//! <cub-policy-selectors>` for more information on how algorithms can be tuned.
//! @endrst
template <class... _PolicySelectors>
[[nodiscard]] _CCCL_NODEBUG_API auto tune(_PolicySelectors...)
{
  static_assert((::cuda::std::is_empty_v<_PolicySelectors> && ...), "Policy selectors must be stateless");
  static_assert((::cuda::std::semiregular<_PolicySelectors> && ...), "Policy selectors must be semiregular types");

  // since all the tunings are stateless, let's ignore incoming parameters

  // we use the return type of the policy_selector as tag
  using tuning_env = ::cuda::std::execution::env<
    ::cuda::std::execution::prop<decltype(__detect_policy_return_type<_PolicySelectors>()), _PolicySelectors>...>;

  return ::cuda::std::execution::prop{__get_tuning_t{}, tuning_env{}};
}

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA___EXECUTION_TUNE_H
