//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_FWD_RCVR_H
#define __CUDAX_ASYNC_DETAIL_FWD_RCVR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/env.cuh>

namespace cuda::experimental::__async
{
template <class Rcvr>
struct _fwd_rcvr : Rcvr
{
  _CCCL_HOST_DEVICE decltype(auto) get_env() const noexcept
  {
    // TODO: only forward the "forwarding" queries:
    return __async::get_env(static_cast<Rcvr const&>(*this));
  }
};

template <class Rcvr>
struct _fwd_rcvr<Rcvr*>
{
  using receiver_concept = receiver_t;
  Rcvr* _rcvr;

  template <class... As>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_value(As&&... as) noexcept
  {
    __async::set_value(_rcvr);
  }

  template <class Error>
  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_error(Error&& error) noexcept
  {
    __async::set_error(_rcvr, static_cast<Error&&>(error));
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void set_stopped() noexcept
  {
    __async::set_stopped(_rcvr);
  }

  _CCCL_HOST_DEVICE decltype(auto) get_env() const noexcept
  {
    // TODO: only forward the "forwarding" queries:
    return __async::get_env(_rcvr);
  }
};
} // namespace cuda::experimental::__async

#endif
