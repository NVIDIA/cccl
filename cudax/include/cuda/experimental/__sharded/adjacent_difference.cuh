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
 *        element per shard (the predecessor of the shard's first element).
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
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

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
 * output[0] = op(input[0], *prev_last) when a predecessor exists (pinned host
 * boundary element from the previous shard), otherwise input[0].
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
 * in[0]`), with the boundary element of each shard's predecessor staged
 * through pinned host memory (one element per shard — the degenerate halo).
 *
 * SYNCHRONOUS-ONLY in this form: the boundary staging requires a host
 * synchronization mid-flight, so the call refuses at entry under
 * `sync_policy::forbid` and under CUDA graph capture, before any work is
 * enqueued. (An asynchronous variant reading the predecessor's last element
 * directly through the shared address space is the recorded follow-up.)
 *
 * Views must be co-partitioned and must not alias (`in[i-1]` is read while
 * `out[i]` is written). Boundary staging is drawn from a memory resource on
 * the call environment when one is present (`cuda::mr::get_memory_resource`,
 * host-accessible + async-transfer-capable), otherwise from the cached
 * pinned arena.
 *
 * @throws std::invalid_argument on partition mismatch, aliasing, or
 *         environment shortfall.
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
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::adjacent_difference: fewer environments than shards");
  }
  for (const auto g : each(num_shards))
  {
    if (static_cast<const void*>(in.shard(g).data) == static_cast<const void*>(out.shard(g).data)
        && in.shard(g).size != 0)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "sharded::adjacent_difference: input and output must not alias (element i-1 is "
                  "read while element i is written)");
    }
  }
  if (num_shards == 0)
  {
    return;
  }

  // Refusals first, before any CUDA call: the boundary staging synchronizes.
  require_sync_allowed(call_env, "sharded::adjacent_difference (boundary staging synchronizes)");
  places::check_not_capturing(nullptr, "sharded::adjacent_difference");
  for (const auto g : each(num_shards))
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::adjacent_difference");
  }

  // Boundary staging: one element per shard, host-accessible, from the call
  // environment's resource when present, the cached pinned arena otherwise.
  constexpr bool __env_has_mr = ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
                             || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  elem_t* h_last              = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_last          = static_cast<elem_t*>(staging_mr.allocate_sync(num_shards * sizeof(elem_t), alignof(elem_t)));
  }
  else
  {
    h_last = static_cast<elem_t*>(reserved::__pinned_staging(num_shards * sizeof(elem_t)));
  }

  // Phase 1: gather each non-empty input shard's last element on its
  // environment stream, then synchronize (the host must see the values and
  // the successor kernels read them zero-copy).
  for (const auto g : each(num_shards))
  {
    const auto& s = in.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    cuda_safe_call(cudaMemcpyAsync(
      &h_last[g], s.data + s.size - 1, sizeof(elem_t), cudaMemcpyDeviceToHost, ::cuda::get_stream(envs[g]).get()));
  }
  barrier(envs);

  // Predecessor per shard: the last element of the previous NON-EMPTY shard.
  ::std::vector<const elem_t*> prev(num_shards, nullptr);
  {
    const elem_t* running = nullptr;
    for (const auto g : each(num_shards))
    {
      prev[g] = running;
      if (in.shard(g).size != 0)
      {
        running = &h_last[g];
      }
    }
  }

  // Phase 2: per-shard difference kernels through the shared driver (its
  // synchronous tail provides this form's final join).
  __detail::__generic_map(
    out, envs, call_env, "sharded::adjacent_difference", [&](::std::size_t g, const auto& d_out, cudaStream_t s) {
      constexpr int block_size = 256;
      const int num_blocks     = static_cast<int>((d_out.size + block_size - 1) / block_size);
      reserved::adjacent_difference_kernel<<<num_blocks, block_size, 0, s>>>(
        in.shard(g).data, d_out.data, d_out.size, prev[g], op);
      cuda_safe_call(cudaGetLastError());
    });

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_last, num_shards * sizeof(elem_t), alignof(elem_t));
  }
  // (arena staging is cached; nothing to release)
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
