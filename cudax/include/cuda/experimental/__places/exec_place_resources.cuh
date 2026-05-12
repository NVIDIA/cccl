//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Standalone per-place stream-pool registry.
 *
 * `exec_place_resources` owns a `{compute, data}` `stream_pool` slot for every
 * pooled place it is queried with. Slots are created lazily on first use and
 * destroyed with the registry. The registry depends only on `stream_pool.cuh`
 * and a forward declaration of `exec_place`; it can be embedded in any
 * resource container (e.g. `async_resources_handle`) without pulling in STF.
 *
 * Keys are `exec_place::impl*` pointers. Pooled implementations (`device(N)`,
 * `host()`) live as process-wide singleton impls, so pointer identity matches
 * place identity for them. Self-contained implementations (`cuda_stream`,
 * green-context, grid) override `get_stream_pool` and never reach the
 * registry.
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

#include <cuda/experimental/__places/stream_pool.cuh>

#include <mutex>
#include <unordered_map>

namespace cuda::experimental::places
{
/**
 * @brief Default size of each per-place stream pool created by the registry.
 *
 * `exec_place::impl::pool_size` and `data_pool_size` are aliases to these
 * values so `places.cuh` can keep its public surface unchanged.
 */
inline constexpr ::std::size_t exec_place_default_pool_size      = 4;
inline constexpr ::std::size_t exec_place_default_data_pool_size = 4;

/**
 * @brief A registry of per-place stream pools keyed by `exec_place::impl*`.
 *
 * For every distinct pooled impl pointer the registry is queried with, it
 * owns one `{compute, data}` pair of `stream_pool`s, created lazily on first
 * lookup with sizes `exec_place_default_pool_size` /
 * `exec_place_default_data_pool_size`.
 *
 * The map itself is mutex-guarded. The mutex is only held across the
 * find/insert into the map; subsequent stream creation (which happens lazily
 * inside `stream_pool::next`) runs outside the lock, so contention is limited
 * to slow-path task submission.
 *
 * Lifetime: each entry's pool is owned by the registry. Destroying the
 * registry destroys every pool it has created (and their cached
 * `cudaStream_t` handles). Consequently, a registry must not outlive the
 * CUDA primary context(s) of the devices it has cached streams for; with
 * this design, registries are typically embedded in an
 * `async_resources_handle` and share the lifetime of the owning STF context.
 *
 * Caveats for externally-owned places:
 * - User-stream places (`exec_place::cuda_stream(s)`) carry their own
 *   single-stream pool and never participate in the registry.
 * - Green-context places carry their own pool (constructed from the
 *   `green_ctx_view`) and also bypass the registry. The user must keep the
 *   underlying `CUgreenCtx` alive as long as the place is used.
 */
class exec_place_resources
{
public:
  struct per_place_pools
  {
    per_place_pools()
        : compute(exec_place_default_pool_size)
        , data(exec_place_default_data_pool_size)
    {}

    stream_pool compute;
    stream_pool data;
  };

  exec_place_resources() = default;

  exec_place_resources(const exec_place_resources&)            = delete;
  exec_place_resources& operator=(const exec_place_resources&) = delete;
  exec_place_resources(exec_place_resources&&)                 = delete;
  exec_place_resources& operator=(exec_place_resources&&)      = delete;

  /**
   * @brief Look up (or lazily create) the `{compute, data}` pool slot for the
   * supplied impl pointer.
   *
   * Thread-safe: the mutex is held only across the find/insert. The returned
   * reference is stable for the lifetime of the registry (`std::unordered_map`
   * preserves node addresses across rehashes).
   */
  per_place_pools& get(const void* impl_key)
  {
    ::std::lock_guard<::std::mutex> lock(mtx_);
    auto it = map_.find(impl_key);
    if (it == map_.end())
    {
      it = map_.emplace(impl_key, per_place_pools{}).first;
    }
    return it->second;
  }

  /// @brief Number of per-place entries currently cached. Mainly for tests.
  ::std::size_t size() const
  {
    ::std::lock_guard<::std::mutex> lock(mtx_);
    return map_.size();
  }

private:
  mutable ::std::mutex mtx_;
  ::std::unordered_map<const void*, per_place_pools> map_;
};
} // namespace cuda::experimental::places
