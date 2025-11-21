//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_SNDR_REF
#define __CUDAX_EXECUTION_SNDR_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_ref
{
  using sender_concept = receiver_t;

  _CCCL_API explicit constexpr __sndr_ref(_Sndr&& __sndr) noexcept
      : __sndr_(static_cast<_Sndr&&>(__sndr))
  {}

  template <class _Self, class... _Env>
  _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    return execution::get_completion_signatures<_Sndr, _Env...>();
  }

  template <class _Rcvr>
  _CCCL_API constexpr auto connect(_Rcvr __rcvr) const
  {
    return execution::connect(static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr));
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> env_of_t<_Sndr>
  {
    return execution::get_env(__sndr_);
  }

private:
  _Sndr&& __sndr_;
};

template <class _Sndr>
_CCCL_HOST_DEVICE __sndr_ref(_Sndr&& __sndr) -> __sndr_ref<_Sndr>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SNDR_REF
