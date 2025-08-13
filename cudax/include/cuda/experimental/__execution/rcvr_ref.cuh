//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_RCVR_REF
#define __CUDAX_EXECUTION_RCVR_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_BEGIN_NV_DIAG_SUPPRESS(114) // function "foo" was referenced but not defined

namespace cuda::experimental::execution
{
template <class _Rcvr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_ref
{
  using receiver_concept = receiver_t;

  _CCCL_TRIVIAL_API constexpr explicit __rcvr_ref(_Rcvr& __rcvr) noexcept
      : __rcvr_(_CUDA_VSTD::addressof(__rcvr))
  {}

  template <class... _As>
  _CCCL_API constexpr void set_value(_As&&... __as) noexcept
  {
    execution::set_value(static_cast<_Rcvr&&>(*__rcvr_), static_cast<_As&&>(__as)...);
  }

  template <class _Error>
  _CCCL_API constexpr void set_error(_Error&& __err) noexcept
  {
    execution::set_error(static_cast<_Rcvr&&>(*__rcvr_), static_cast<_Error&&>(__err));
  }

  _CCCL_API constexpr void set_stopped() noexcept
  {
    execution::set_stopped(static_cast<_Rcvr&&>(*__rcvr_));
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> env_of_t<_Rcvr>
  {
    return execution::get_env(*__rcvr_);
  }

private:
  _Rcvr* __rcvr_;
};

template <class _Rcvr>
_CCCL_HOST_DEVICE __rcvr_ref(_Rcvr&) -> __rcvr_ref<_Rcvr>;

} // namespace cuda::experimental::execution

_CCCL_END_NV_DIAG_SUPPRESS() // function "foo" was references but not defined

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_RCVR_REF
