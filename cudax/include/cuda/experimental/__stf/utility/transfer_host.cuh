//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of transfer_host
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__stf/internal/logical_data.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

namespace cuda::experimental::stf
{

template <typename T>
struct owning_container_of;

template <typename T>
auto transfer_host(context& ctx, logical_data<T>& ldata)
{
  using valT = typename owning_container_of<T>::type;
  valT out;

  bool is_graph = ctx.is_graph_ctx();
  if (is_graph)
  {
    ctx.host_launch(ldata.read()).set_symbol("transfer_host")->*[&](auto data) {
      out = owning_container_of<T>::get_value(data);
    };

    /* This forces the completion of the host callback, so that the host
     * thread can use the content for dynamic control flow */
    cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
  }
  else
  {
    ctx.task(exec_place::host, ldata.read()).set_symbol("transfer_host")->*[&](cudaStream_t stream, auto data) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      out = owning_container_of<T>::get_value(data);
    };
  }

  return out;
}

} // end namespace cuda::experimental::stf
