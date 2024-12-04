//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_RCVR_REF
#define __CUDAX_ASYNC_DETAIL_RCVR_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/meta.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{

template <class _Rcvr>
constexpr _Rcvr* __rcvr_ref(_Rcvr& __rcvr) noexcept
{
  return &__rcvr;
}

template <class _Rcvr>
constexpr _Rcvr* __rcvr_ref(_Rcvr* __rcvr) noexcept
{
  return __rcvr;
}

template <class _Rcvr>
using __rcvr_ref_t = decltype(__async::__rcvr_ref(__declval<_Rcvr>()));

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
