//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_HAS_ALL_TO_ALL_H
#define _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_HAS_ALL_TO_ALL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/same_as.h> // IWYU pragma: keep
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/mgmn/__concepts/communicator.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN

template <class _Comm, class _Ptr = int*>
_CCCL_CONCEPT __has_all_to_all = _CCCL_REQUIRES_EXPR(
  (_Comm, _Ptr),
  _Comm& __comm,
  _Ptr __sendbuff,
  _Ptr __recvbuff,
  ::cuda::std::size_t __count,
  ::cuda::stream_ref __stream)(
  requires(__communicator<_Comm>),
  _Same_as(void)
    __comm.all_to_all(::cuda::std::declval<__group_guard_t<_Comm>&>(), __sendbuff, __recvbuff, __count, __stream));

_CCCL_END_NAMESPACE_CUDA_MGMN

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_HAS_ALL_TO_ALL_H
