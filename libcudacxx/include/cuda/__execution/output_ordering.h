//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA___EXECUTION_OUTPUT_ORDERING_H
#define __CUDA___EXECUTION_OUTPUT_ORDERING_H

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

namespace output_ordering
{
struct __get_output_ordering_t;

enum class __output_ordering_t
{
  __sorted,
  __unsorted
};

template <__output_ordering_t _Guarantee>
struct _CCCL_DECLSPEC_EMPTY_BASES __output_ordering_holder_t
    : __requirement
    , ::cuda::std::integral_constant<__output_ordering_t, _Guarantee>
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const __get_output_ordering_t&) const noexcept
    -> __output_ordering_holder_t<_Guarantee>
  {
    return *this;
  }
};

using sorted_t   = __output_ordering_holder_t<__output_ordering_t::__sorted>;
using unsorted_t = __output_ordering_holder_t<__output_ordering_t::__unsorted>;

_CCCL_GLOBAL_CONSTANT sorted_t sorted{};
_CCCL_GLOBAL_CONSTANT unsorted_t unsorted{};

struct __get_output_ordering_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, __get_output_ordering_t>)
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

_CCCL_GLOBAL_CONSTANT auto __get_output_ordering = __get_output_ordering_t{};
} // namespace output_ordering

_CCCL_END_NAMESPACE_CUDA_EXECUTION

#include <cuda/std/__cccl/epilogue.h>

#endif
