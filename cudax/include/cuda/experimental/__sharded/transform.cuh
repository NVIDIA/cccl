//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Elementwise transforms over sharded arrays (in-place, unary,
 *        binary). No cross-place stage: each shard transforms locally.
 */

#pragma once

#include <cuda/__cccl_config>
#include <cuda/stream>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/// @brief In-place unary transform: data[i] = op(data[i]).
template <typename _Tp, typename _UnaryOp>
_CCCL_HOST_API void transform(place_group&, sharded_array<_Tp>& data, _UnaryOp op, bool blocking = true)
{
  if (data.empty())
  {
    return;
  }

  data.each_shard->*[op](auto& s) {
    thrust::transform(thrust::cuda::par_nosync.on(s.stream), s.data, s.data + s.size, s.data, op);
    cuda_safe_call(cudaGetLastError());
  };

  if (blocking)
  {
    data.sync();
  }
}

/**
 * @brief Out-of-place unary transform: output[i] = op(input[i]).
 *
 * Input and output must be compatible (same shard sizes and places); each
 * output stream waits on the corresponding input stream.
 *
 * @throws std::invalid_argument when the layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _UnaryOp>
_CCCL_HOST_API void
transform(place_group&, const sharded_array<_Tp>& input, sharded_array<_Up>& output, _UnaryOp op, bool blocking = true)
{
  check_compatible(input, output, "transform (unary out-of-place)");

  if (input.empty())
  {
    return;
  }

  // Make each output stream wait for the corresponding input stream
  for (size_t g = 0; g < input.num_shards(); g++)
  {
    ::cuda::stream_ref{output.shard(g).stream}.wait(::cuda::stream_ref{input.shard(g).stream});
  }

  output.each_shard->*[&input, op](const size_t g, auto& out_shard) {
    const auto& in_shard = input.shard(g);
    thrust::transform(
      thrust::cuda::par_nosync.on(out_shard.stream), in_shard.data, in_shard.data + in_shard.size, out_shard.data, op);
    cuda_safe_call(cudaGetLastError());
  };

  if (blocking)
  {
    output.sync();
  }
}

/**
 * @brief Binary transform: output[i] = op(input1[i], input2[i]).
 *
 * All three arrays must be compatible (same shard sizes and places).
 *
 * @throws std::invalid_argument when the layouts are not compatible
 */
template <typename _Tp, typename _Up, typename _BinaryOp>
_CCCL_HOST_API void transform(
  place_group&,
  const sharded_array<_Tp>& input1,
  const sharded_array<_Tp>& input2,
  sharded_array<_Up>& output,
  _BinaryOp op,
  bool blocking = true)
{
  check_compatible(input1, input2, "transform (binary): input1 vs input2");
  check_compatible(input1, output, "transform (binary): inputs vs output");

  if (input1.empty())
  {
    return;
  }

  // Make each output stream wait for both input streams
  for (size_t g = 0; g < input1.num_shards(); g++)
  {
    const auto& out_shard = output.shard(g);
    ::cuda::stream_ref{out_shard.stream}.wait(::cuda::stream_ref{input1.shard(g).stream});
    ::cuda::stream_ref{out_shard.stream}.wait(::cuda::stream_ref{input2.shard(g).stream});
  }

  output.each_shard->*[&input1, &input2, op](const size_t g, auto& out_shard) {
    const auto& in1_shard = input1.shard(g);
    const auto& in2_shard = input2.shard(g);
    thrust::transform(
      thrust::cuda::par_nosync.on(out_shard.stream),
      in1_shard.data,
      in1_shard.data + in1_shard.size,
      in2_shard.data,
      out_shard.data,
      op);
    cuda_safe_call(cudaGetLastError());
  };

  if (blocking)
  {
    output.sync();
  }
}
} // namespace cuda::experimental::sharded
