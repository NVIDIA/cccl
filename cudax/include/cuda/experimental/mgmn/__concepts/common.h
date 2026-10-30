//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_COMMON_H
#define _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__ranges/concepts.h>

#include <cuda/experimental/mgmn/__concepts/communicator.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
_CCCL_BEGIN_NAMESPACE_CUDA_MGMN

template <class _Range>
_CCCL_CONCEPT __range_of_communicators = _CCCL_REQUIRES_EXPR((_Range), )(
  requires(::cuda::std::ranges::forward_range<_Range>),
  requires(__communicator<::cuda::std::ranges::range_reference_t<_Range>>));

_CCCL_END_NAMESPACE_CUDA_MGMN

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL_MGMN___CONCEPTS_COMMON_H
