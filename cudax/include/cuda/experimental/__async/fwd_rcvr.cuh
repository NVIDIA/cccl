//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FWD_RCVR
#define __CUDAX_ASYNC_DETAIL_FWD_RCVR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/env.cuh>
#include <cuda/experimental/__detail/config.cuh>

namespace cuda::experimental::__async
{
template <class _Rcvr>
struct __fwd_rcvr : _Rcvr
{
  _CUDAX_API decltype(auto) get_env() const noexcept
  {
    // TODO: only forward the "forwarding" queries:
    return __async::get_env(static_cast<_Rcvr const&>(*this));
  }
};

template <class _Rcvr>
struct __fwd_rcvr<_Rcvr*>
{
  using receiver_concept = receiver_t;
  _Rcvr* __rcvr_;

  template <class... _As>
  _CUDAX_TRIVIAL_API void set_value(_As&&... __as) noexcept
  {
    __async::set_value(__rcvr_);
  }

  template <class _Error>
  _CUDAX_TRIVIAL_API void set_error(_Error&& __error) noexcept
  {
    __async::set_error(__rcvr_, static_cast<_Error&&>(__error));
  }

  _CUDAX_TRIVIAL_API void set_stopped() noexcept
  {
    __async::set_stopped(__rcvr_);
  }

  _CUDAX_API decltype(auto) get_env() const noexcept
  {
    // TODO: only forward the "forwarding" queries:
    return __async::get_env(__rcvr_);
  }
};
} // namespace cuda::experimental::__async

#endif
