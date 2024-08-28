//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_RCVR_REF_H
#define __CUDAX_ASYNC_DETAIL_RCVR_REF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/meta.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{

template <class Rcvr>
constexpr Rcvr* _rcvr_ref(Rcvr& rcvr) noexcept
{
  return &rcvr;
}

template <class Rcvr>
constexpr Rcvr* _rcvr_ref(Rcvr* rcvr) noexcept
{
  return rcvr;
}

template <class Rcvr>
using _rcvr_ref_t = decltype(__async::_rcvr_ref(DECLVAL(Rcvr)));

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
