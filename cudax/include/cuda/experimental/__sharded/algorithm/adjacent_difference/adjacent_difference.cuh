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
 * @brief Adjacent difference over sharded arrays. Each shard computes its
 *        differences locally; the only cross-place traffic is one boundary
 *        element per shard (the predecessor of the shard's first element),
 *        read directly from the previous shard through the shared address
 *        space.
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

#include <cuda/std/functional>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/concepts/guards.cuh>
#include <cuda/experimental/__sharded/container/default_envs.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/engine/visit_shards.cuh>

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
/**
 * @brief Per-shard adjacent difference kernel.
 *
 * output[i] = op(input[i], input[i-1]) for i > 0.
 * output[0] = op(input[0], *prev_last) when a predecessor exists (the last
 * element of the previous non-empty shard, read in place), otherwise input[0].
 */
template <typename _Tp, typename _BinaryOp>
__global__ void adjacent_difference_kernel(const _Tp* input, _Tp* output, size_t n, const _Tp* prev_last, _BinaryOp op)
{
  // Promote before multiplying: blockIdx.x * blockDim.x overflows unsigned
  // int for grids past 2^32 threads. Callers never launch over empty shards,
  // so thread 0 writing output[0] unconditionally is safe.
  const size_t idx = size_t{blockIdx.x} * blockDim.x + threadIdx.x;

  if (idx == 0)
  {
    output[0] = prev_last ? op(input[0], *prev_last) : input[0];
  }
  else if (idx < n)
  {
    output[idx] = op(input[idx], input[idx - 1]);
  }
}
} // namespace reserved

// ============================================================================
// Concept-generic tier: adjacent difference over any pair of sharded views
// ============================================================================

/**
 * @brief Out-of-place adjacent difference over sharded views:
 * `out[i] = op(in[i], in[i-1])` across the global index space (`out[0] =
 * in[0]`). Each shard's kernel reads its predecessor's last element directly
 * from the previous non-empty shard (the degenerate one-element halo): no
 * staging buffer, no host round trip.
 *
 * A map-family call with one extra edge per shard boundary: shard g's kernel
 * waits (event edge, non-blocking, capture-legal) on the lane of the shard
 * whose element it reads, so a predecessor still being produced on its own
 * lane is observed complete. Everything else follows the map-family contract
 * (`__visit_shards`): stream on the call environment = asynchronous,
 * lane-ordered by default, `composition::bracketed` on request; no stream =
 * synchronous convenience (refused under `sync_policy::forbid`).
 *
 * The direct read requires the previous shard's memory to be addressable
 * from the reading shard's place — always true within one device (locality
 * domains share the address space), and for peer-mapped multi-device arrays.
 *
 * Views must be co-partitioned.
 *
 * @pre `in` and `out` do not overlap: `in[i-1]` is read while `out[i]` is
 *      written, on per-shard streams, so any overlap (including the exact
 *      in-place form `std::adjacent_difference` permits) is a data race.
 *      Not checked: address-range overlap across shards on different
 *      lanes cannot be validated reliably here. In-place support is a
 *      recorded follow-up (per-tile predecessor read, as CUB's
 *      SubtractLeft does).
 *
 * @throws std::invalid_argument on partition mismatch or environment
 *         count mismatch.
 */
_CCCL_TEMPLATE(class _SIn, class _Envs, class _SOut, class _BinaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND
    sharded_env_range<::cuda::std::remove_cvref_t<_Envs>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void
adjacent_difference(const _SIn& in, const _Envs& envs, _SOut&& out, _BinaryOp op, const _CallEnv& call_env = {})
{
  using elem_t = shard_element_t<shard_descriptor_t<::cuda::std::remove_cvref_t<_SIn>>>;

  reserved::__check_copartitioned(out, in, "sharded::adjacent_difference");
  const ::std::size_t num_shards = reserved::__shard_count(out);
  if (num_shards == 0)
  {
    return;
  }

  // Predecessor per shard: index of the previous NON-EMPTY shard (host-known
  // from the sizes alone; no data is inspected).
  constexpr ::std::size_t no_pred = static_cast<::std::size_t>(-1);
  ::std::vector<::std::size_t> pred(num_shards, no_pred);
  {
    ::std::size_t running = no_pred;
    for (const auto g : each(num_shards))
    {
      pred[g] = running;
      if (in.shard(g).size != 0)
      {
        running = g;
      }
    }
  }

  __detail::__visit_shards(
    out, envs, call_env, "sharded::adjacent_difference", [&](::std::size_t g, const auto& d_out, cudaStream_t s) {
      const elem_t* prev_last = nullptr;
      if (pred[g] != no_pred)
      {
        // The halo edge: the predecessor was (possibly) produced on its own
        // lane; make this lane observe that lane's enqueued work before the
        // read. A no-op when both shards share a stream.
        __detail::__wait_stream_on(s, ::cuda::get_stream(envs[pred[g]]).get());
        const auto& p = in.shard(pred[g]);
        prev_last     = p.data + p.size - 1;
      }
      constexpr int block_size = 256;
      const int num_blocks     = static_cast<int>((d_out.size + block_size - 1) / block_size);
      reserved::adjacent_difference_kernel<<<num_blocks, block_size, 0, s>>>(
        in.shard(g).data, d_out.data, d_out.size, prev_last, op);
      cuda_safe_call(cudaGetLastError());
    });
}

/**
 * @brief Out-of-place adjacent difference over self-bound sharded views:
 * environments derived from the output via `default_envs`.
 */
_CCCL_TEMPLATE(class _SIn, class _SOut, class _BinaryOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_SOut>>
                 _CCCL_AND(!sharded_env_range<::cuda::std::remove_cvref_t<_BinaryOp>>)
                   _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_BinaryOp>>))
_CCCL_HOST_API void adjacent_difference(const _SIn& in, _SOut&& out, _BinaryOp op, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(out);
  sharded::adjacent_difference(in, envs, ::cuda::std::forward<_SOut>(out), op, call_env);
}
} // namespace cuda::experimental::sharded
