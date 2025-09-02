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

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
struct graph_builder;
struct graph_node_ref;
struct graph;
struct path_builder;

template <class... _Nodes>
_CCCL_NODEBUG_HOST_API constexpr auto depends_on(const _Nodes&... __nodes) noexcept
  -> ::cuda::std::array<cudaGraphNode_t, sizeof...(_Nodes)>;
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_FWD
