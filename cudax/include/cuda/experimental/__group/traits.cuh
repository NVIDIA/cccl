//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH
#define _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Mapping, class _Unit, class _Level, class _Hierarchy>
using __group_mapping_result_t =
  decltype(::cuda::std::declval<_Mapping>().map(_Unit{}, _Level{}, ::cuda::std::declval<const _Hierarchy&>()));

template <class _Synchronizer, class _Unit, class _Level, class _Mapping, class _MappingResult>
using __group_synchronizer_instance_t = decltype(::cuda::std::declval<_Synchronizer>().make_instance(
  _Unit{}, _Level{}, ::cuda::std::declval<const _Mapping&>(), ::cuda::std::declval<const _MappingResult&>()));
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_TRAITS_CUH
