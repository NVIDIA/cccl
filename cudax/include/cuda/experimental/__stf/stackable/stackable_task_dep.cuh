//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Task dependencies in a stackable context

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental::stf
{

template <typename T, typename reduce_op, bool initialize>
class stackable_task_dep;

namespace reserved
{

template <typename T>
struct is_stackable_task_dep : ::std::false_type
{};

template <typename T, typename ReduceOp, bool Init>
struct is_stackable_task_dep<stackable_task_dep<T, ReduceOp, Init>> : ::std::true_type
{};

template <typename T>
inline constexpr bool is_stackable_task_dep_v = is_stackable_task_dep<T>::value;

// This helper converts stackable_task_dep to the underlying task_dep. If we
// have a stackable_logical_data A, A.read() is indeed a stackable_task_dep,
// which we can pass to stream_ctx/graph_ctx constructs by extracting the
// underlying task_dep.
template <typename U>
decltype(auto) to_task_dep(U&& u)
{
  if constexpr (is_stackable_task_dep_v<::std::decay_t<U>>)
  {
    return ::std::forward<U>(u).underlying_dep();
  }
  else
  {
    return ::std::forward<U>(u);
  }
}

} // end namespace reserved

} // end namespace cuda::experimental::stf
