//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_FWD
#define __CUDAX_GRAPH_FWD

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/array>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph;
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_node;
struct _CCCL_TYPE_VISIBILITY_DEFAULT graph_node_ref;

template <class... _Nodes>
_CCCL_TRIVIAL_HOST_API constexpr auto depends_on(const _Nodes&... __nodes) noexcept
  -> _CUDA_VSTD::array<cudaGraphNode_t, sizeof...(_Nodes)>;
} // namespace cuda::experimental

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_GRAPH_FWD
