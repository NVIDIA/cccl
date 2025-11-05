//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_SCHEDULE_FROM
#define __CUDAX_EXECUTION_SCHEDULE_FROM

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no sys

#include <cuda/__utility/immovable.h>
#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/concepts.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct schedule_from_t
{
  _CUDAX_SEMI_PRIVATE:
  template <class _Sndr>
  struct __sndr_t;

  template <class _Sndr>
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Sndr __sndr) const noexcept
  {
    return __sndr_t<_Sndr>{{}, {}, _CCCL_MOVE(__sndr)};
  }
};

template <class _Sndr>
struct schedule_from_t::__sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class... _Env>
  _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return get_child_completion_signatures<_Self, _Sndr, _Env...>();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> connect_result_t<_Sndr, _Rcvr>
  {
    return execution::connect(_CCCL_MOVE(__sndr_), _CCCL_MOVE(__rcvr));
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> connect_result_t<const _Sndr&, _Rcvr>
  {
    return execution::connect(__sndr_, _CCCL_MOVE(__rcvr));
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }

  schedule_from_t __tag{};
  ::cuda::std::__ignore_t __ignore_;
  _Sndr __sndr_;
};

template <class _Sndr>
inline constexpr size_t structured_binding_size<schedule_from_t::__sndr_t<_Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT schedule_from_t schedule_from{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SCHEDULE_FROM
