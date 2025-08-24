//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_GRAPH_CONCEPTS
#define __CUDAX_GRAPH_CONCEPTS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__graph/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

// Concept to check if T is a graph dependency or contains them (either path_builder or graph_node_ref)
// TODO we might do something more abstract here rather than just checking specific types
template <typename T>
_CCCL_CONCEPT graph_dependency =
  ::cuda::std::is_same_v<::cuda::std::decay_t<T>, path_builder>
  || ::cuda::std::is_same_v<::cuda::std::decay_t<T>, graph_node_ref>;

// Concept to check if T can insert nodes into a graph
// TODO we might do something more abstract here rather than just checking specific types
template <typename T>
_CCCL_CONCEPT graph_inserter = ::cuda::std::is_same_v<::cuda::std::decay_t<T>, path_builder>;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_GRAPH_CONCEPTS
