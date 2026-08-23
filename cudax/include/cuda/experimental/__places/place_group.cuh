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
 * @brief `place_group`: a grid of execution places together with the
 *        resources it takes to execute on them (per-place stream pools and
 *        per-place memory resources).
 *
 * A grid of places names WHERE things can run; a `place_group` owns WHAT IT
 * TAKES to run there. The distinction is deliberate:
 *
 *  - A grid (or a `std::vector<exec_place>`) is a pure value: copyable,
 *    transient, derivable from `place_partition`, with no lifetime of its
 *    own.
 *  - A `place_group` is a resource scope: lazily created per-place stream
 *    pools, per-place memory resources, and a well-defined teardown order.
 *    Two groups over the same grid are two deliberately distinct isolation
 *    scopes.
 *
 * This mirrors the MPI precedent of `MPI_Group` (membership) versus
 * `MPI_Comm` (membership plus attached state), with explicit construction of
 * the second from the first. The alternatives are per-call resource creation
 * (stream-pool and green-context setup are measurably expensive) or a hidden
 * grid-keyed global registry (implicit primary-context-style lifetime).
 *
 * When a `place_group` coexists with an STF context, it can BORROW the
 * context's `async_resources_handle` stream-pool registry instead of owning
 * its own, so there is exactly one pool owner per program:
 *
 * @code
 * // Standalone: the group owns its stream pools.
 * auto group = place_group::by_locality_domains();
 *
 * // Coexisting with STF: borrow the context's pools (one pool owner).
 * cuda::experimental::stf::context ctx;
 * place_group group{some_places, ctx.async_resources()};
 * @endcode
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

#include <cuda/memory_resource>
#include <cuda/std/__execution/env.h>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__places/exec_place_resources.cuh>
#include <cuda/experimental/__places/machine.cuh>
#include <cuda/experimental/__places/place_partition.cuh>
#include <cuda/experimental/__places/places.cuh>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::places
{
// ============================================================================
// Free helpers: build vectors of exec_place for the common layouts
// ============================================================================

/// @brief Device ordinals of every visible CUDA device.
inline ::std::vector<int> all_device_ids()
{
  const int ndevs = cuda_try<cudaGetDeviceCount>();
  ::std::vector<int> ids(static_cast<size_t>(ndevs));
  for (int d = 0; d < ndevs; d++)
  {
    ids[static_cast<size_t>(d)] = d;
  }
  return ids;
}

/// @brief One `exec_place` per listed device ordinal.
inline ::std::vector<exec_place> places_from_devices(const ::std::vector<int>& device_ids)
{
  ::std::vector<exec_place> result;
  result.reserve(device_ids.size());
  for (int id : device_ids)
  {
    result.push_back(exec_place::device(id));
  }
  return result;
}

/// @brief Flatten an `exec_place` grid (or a scalar place) into a vector of places.
inline ::std::vector<exec_place> places_from_grid(const exec_place& grid)
{
  ::std::vector<exec_place> result;
  result.reserve(grid.size());
  for (size_t i = 0; i < grid.size(); i++)
  {
    result.push_back(grid.get_place(i));
  }
  return result;
}

/**
 * @brief One `exec_place` per locality domain of every listed device
 * (device-major order); an empty list means all visible devices.
 *
 * Devices without locality-domain support contribute a single whole-device
 * place, so this is safe on every machine.
 */
inline ::std::vector<exec_place> places_from_locality_domains(::std::vector<int> device_ids = {})
{
  if (device_ids.empty())
  {
    device_ids = all_device_ids();
  }

  ::std::vector<::std::shared_ptr<exec_place>> devices;
  devices.reserve(device_ids.size());
  for (int d : device_ids)
  {
    devices.push_back(::std::make_shared<exec_place>(exec_place::device(d)));
  }
  place_partition partition(devices, place_partition_scope::locality_domain);
  return ::std::vector<exec_place>(partition.begin(), partition.end());
}

// ============================================================================
// place_memory_resource: per-place memory resource over data_place
// ============================================================================

/**
 * @brief A memory resource that allocates from a `data_place`, modeling the
 * `cuda::mr` resource concepts.
 *
 * This makes any place (device, host, managed, green-context, locality
 * domain) usable wherever CCCL expects a memory resource — in particular in
 * the environments accepted by CUB's single-call device algorithms, so that
 * algorithm temporaries land on the same place as the data they serve.
 *
 * Stream-ordered places allocate/deallocate on the provided stream; other
 * places fall back to their immediate allocation path.
 */
class place_memory_resource
{
public:
  /// @brief Construct a memory resource allocating from @p place.
  explicit place_memory_resource(data_place place)
      : place_(mv(place))
      , is_stream_ordered_(place_.allocation_is_stream_ordered())
  {}

  /// @brief The underlying data place.
  const data_place& place() const
  {
    return place_;
  }

  /// @brief Whether allocations are stream-ordered on this place.
  bool is_stream_ordered() const
  {
    return is_stream_ordered_;
  }

  /// @brief Stream-ordered allocation (models the `cuda::mr` resource concept).
  [[nodiscard]] void*
  allocate(::cuda::stream_ref stream, ::std::size_t bytes, ::std::size_t /*alignment*/ = alignof(::std::max_align_t))
  {
    if (bytes == 0)
    {
      return nullptr;
    }
    cudaStream_t cuda_stream = is_stream_ordered_ ? stream.get() : nullptr;
    return place_.allocate(static_cast<::std::ptrdiff_t>(bytes), cuda_stream);
  }

  /// @brief Stream-ordered deallocation (models the `cuda::mr` resource concept).
  void deallocate(::cuda::stream_ref stream,
                  void* ptr,
                  ::std::size_t bytes,
                  ::std::size_t /*alignment*/ = alignof(::std::max_align_t)) noexcept
  {
    if (ptr == nullptr)
    {
      return;
    }
    cudaStream_t cuda_stream = is_stream_ordered_ ? stream.get() : nullptr;
    place_.deallocate(ptr, bytes, cuda_stream);
  }

  /// @brief Synchronous allocation (models the `cuda::mr` synchronous resource concept).
  [[nodiscard]] void* allocate_sync(::std::size_t bytes, ::std::size_t /*alignment*/ = alignof(::std::max_align_t))
  {
    if (bytes == 0)
    {
      return nullptr;
    }
    return place_.allocate(static_cast<::std::ptrdiff_t>(bytes), nullptr);
  }

  /// @brief Synchronous deallocation (models the `cuda::mr` synchronous resource concept).
  void
  deallocate_sync(void* ptr, ::std::size_t bytes, ::std::size_t /*alignment*/ = alignof(::std::max_align_t)) noexcept
  {
    if (ptr == nullptr)
    {
      return;
    }
    place_.deallocate(ptr, bytes, nullptr);
  }

  /// @brief Two resources are equal when they allocate from the same place.
  friend bool operator==(const place_memory_resource& lhs, const place_memory_resource& rhs) noexcept
  {
    return lhs.place_ == rhs.place_;
  }

  friend bool operator!=(const place_memory_resource& lhs, const place_memory_resource& rhs) noexcept
  {
    return !(lhs == rhs);
  }

private:
  data_place place_;
  bool is_stream_ordered_;
};

// ============================================================================
// Stream utilities
// ============================================================================

/**
 * @brief Make @p wait_stream wait for the work currently enqueued on
 * @p signal_stream, using a short-lived CUDA event.
 *
 * A no-op when both streams are the same.
 */
inline void make_stream_wait_for(cudaStream_t wait_stream, cudaStream_t signal_stream)
{
  if (wait_stream == signal_stream)
  {
    return;
  }

  cudaEvent_t event = nullptr;
  cuda_safe_call(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  cuda_safe_call(cudaEventRecord(event, signal_stream));
  cuda_safe_call(cudaStreamWaitEvent(wait_stream, event, 0));
  cuda_safe_call(cudaEventDestroy(event));
}

namespace detail
{
// Detects handle types exposing `get_place_resources() -> exec_place_resources&`
// (e.g. the STF `async_resources_handle`), without this header depending on them.
template <typename Handle, typename = void>
inline constexpr bool has_place_resources = false;

template <typename Handle>
inline constexpr bool has_place_resources<
  Handle,
  ::cuda::std::enable_if_t<
    ::cuda::std::is_same_v<decltype(::cuda::std::declval<Handle&>().get_place_resources()), exec_place_resources&>>> =
  true;
} // namespace detail

// ============================================================================
// place_group
// ============================================================================

/**
 * @brief A group of execution places plus the execution resources attached to
 * them: lazily initialized per-place stream pools and per-place memory
 * resources.
 *
 * See the file-level comment for the grid-versus-group rationale. In short: a
 * grid is a stateless value naming places; a `place_group` is the resource
 * scope you execute against. Construction from a grid stays a one-liner:
 *
 * @code
 * auto group = place_group{make_locality_domain_grid(0)};
 * @endcode
 *
 * Stream pools are drawn from an `exec_place_resources` registry. The group
 * either OWNS its registry (default) or BORROWS one — e.g. an STF context's
 * `async_resources_handle` — so that exactly one pool owner exists when both
 * layers coexist. Borrowed handles with shared-ownership semantics are kept
 * alive by the group.
 */
class place_group
{
public:
  /// @brief Sentinel color requesting automatic (round-robin) stream-color
  /// selection.
  static constexpr size_t auto_stream_color = static_cast<size_t>(-1);

  /// @brief Create a group owning its stream pools, over an explicit set of places.
  explicit place_group(::std::vector<exec_place> places)
      : places_(mv(places))
      , owned_resources_(::std::make_unique<exec_place_resources>())
      , resources_(owned_resources_.get())
  {
    init();
  }

  /// @brief Create a group from an `exec_place` grid (or a scalar place),
  /// flattened to one place per grid entry.
  explicit place_group(const exec_place& grid)
      : place_group(places_from_grid(grid))
  {}

  /**
   * @brief Create a group that BORROWS an existing stream-pool registry
   * instead of owning one.
   *
   * The registry must outlive the group. This is the low-level borrowing
   * seam; prefer the handle overload below when a handle with shared
   * ownership (such as STF's `async_resources_handle`) is available.
   */
  place_group(::std::vector<exec_place> places, exec_place_resources& resources)
      : places_(mv(places))
      , resources_(&resources)
  {
    init();
  }

  /**
   * @brief Create a group that borrows the stream pools of a resource handle
   * exposing `get_place_resources()` — e.g. STF's `async_resources_handle`.
   *
   * The group stores a copy of the handle (shared-ownership semantics), so
   * the borrowed pools remain valid for the lifetime of the group and there
   * is a single pool owner when a `place_group` coexists with an STF context.
   */
  template <typename ResourceHandle, typename = ::cuda::std::enable_if_t<detail::has_place_resources<ResourceHandle>>>
  place_group(::std::vector<exec_place> places, ResourceHandle handle)
      : places_(mv(places))
  {
    auto holder = ::std::make_shared<ResourceHandle>(mv(handle));
    resources_  = &holder->get_place_resources();
    keep_alive_ = mv(holder);
    init();
  }

  // ==========================================================================
  // One-call factories for the common place layouts
  // ==========================================================================

  /**
   * @brief Group with one place per device: all visible devices, or the
   * listed ones.
   */
  static place_group by_devices(::std::vector<int> device_ids = {})
  {
    if (device_ids.empty())
    {
      device_ids = all_device_ids();
    }
    return place_group(places_from_devices(device_ids));
  }

  /**
   * @brief Group with one place per locality domain of every device (or of
   * the listed devices) — compute and memory co-located per domain.
   *
   * Devices without locality-domain support contribute a single whole-device
   * place, so this is safe everywhere.
   */
  static place_group by_locality_domains(::std::vector<int> device_ids = {})
  {
    return place_group(places_from_locality_domains(mv(device_ids)));
  }

  // ==========================================================================
  // Places
  // ==========================================================================

  const ::std::vector<exec_place>& places() const
  {
    return places_;
  }

  size_t size() const
  {
    return places_.size();
  }

  const exec_place& place(size_t idx) const
  {
    _CCCL_ASSERT(idx < places_.size(), "place_group: place index out of range");
    return places_[idx];
  }

  const exec_place& operator[](size_t idx) const
  {
    return place(idx);
  }

  // ==========================================================================
  // Streams
  // ==========================================================================
  // Each place carries a pool of streams (its compute pool in the underlying
  // registry). A stream "color" is an index into that pool: work mapped to
  // different colors may overlap since it runs on different streams. Streams
  // are created lazily, on first use of each (place, color) slot.

  /// @brief Number of stream colors available per place.
  size_t num_stream_colors() const
  {
    return exec_place_default_pool_size;
  }

  /// @brief Get the stream of @p place for a given color (default color 0).
  cudaStream_t get_stream(const exec_place& place, size_t color = 0)
  {
    return get_or_create_streams(place)[color % exec_place_default_pool_size];
  }

  /// @brief Get the stream of the idx-th place for a given color.
  cudaStream_t get_stream(size_t place_idx, size_t color = 0)
  {
    return get_stream(place(place_idx), color);
  }

  /**
   * @brief Next stream color, round-robin. Thread-safe.
   *
   * Use to spread independent operations over the per-place pools.
   */
  size_t next_stream_color()
  {
    return stream_color_counter_.fetch_add(1, ::std::memory_order_relaxed) % exec_place_default_pool_size;
  }

  /// @brief Stream for a place, either at an explicit color or round-robin.
  cudaStream_t get_colored_stream(const exec_place& place, size_t color = auto_stream_color)
  {
    return get_stream(place, color == auto_stream_color ? next_stream_color() : color);
  }

  /// @brief Synchronize the stream of a place at the given color.
  void sync_stream(const exec_place& place, size_t color = 0)
  {
    exec_place_scope scope(place);
    cuda_safe_call(cudaStreamSynchronize(get_stream(place, color)));
  }

  void sync_stream(size_t place_idx, size_t color = 0)
  {
    sync_stream(place(place_idx), color);
  }

  /// @brief Synchronize every stream created so far, on every place.
  void sync()
  {
    for (const auto& p : places_)
    {
      exec_place_scope scope(p);
      for (cudaStream_t s : get_or_create_streams(p))
      {
        cuda_safe_call(cudaStreamSynchronize(s));
      }
    }
  }

  // ==========================================================================
  // Memory resources and environments
  // ==========================================================================

  /// @brief Memory resource allocating from the affine data place of the idx-th place.
  place_memory_resource memory_resource(size_t place_idx) const
  {
    return place_memory_resource(place(place_idx).affine_data_place());
  }

  /// @brief Memory resource allocating from an explicit data place.
  place_memory_resource memory_resource(const data_place& dplace) const
  {
    return place_memory_resource(dplace);
  }

  /**
   * @brief Environment combining a stream with the place's memory resource.
   *
   * Suitable for CUB's single-call device algorithms: temporaries are
   * allocated from the place that runs the work.
   */
  static auto env(const data_place& dplace, cudaStream_t stream)
  {
    auto stream_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{stream}};
    auto mr_prop     = ::cuda::std::execution::prop{::cuda::mr::get_memory_resource, place_memory_resource(dplace)};
    return ::cuda::std::execution::env{stream_prop, mr_prop};
  }

  /// @brief Environment for the idx-th place using an explicit stream.
  auto env(size_t place_idx, cudaStream_t stream) const
  {
    return env(place(place_idx).affine_data_place(), stream);
  }

  /// @brief Environment for the idx-th place using the group's stream at color 0.
  auto env(size_t place_idx)
  {
    return env(place_idx, get_stream(place_idx));
  }

  // ==========================================================================
  // Resource ownership
  // ==========================================================================

  /// @brief The stream-pool registry this group draws from (owned or borrowed).
  exec_place_resources& resources()
  {
    return *resources_;
  }

  /// @brief True when the group owns its stream-pool registry; false when it
  /// borrows one (e.g. from an STF `async_resources_handle`).
  bool owns_resources() const
  {
    return owned_resources_ != nullptr;
  }

  // Move-only: the group is a resource scope.
  place_group(place_group&& other) noexcept
      : places_(mv(other.places_))
      , owned_resources_(mv(other.owned_resources_))
      , resources_(other.resources_)
      , keep_alive_(mv(other.keep_alive_))
      , stream_cache_(mv(other.stream_cache_))
      , stream_color_counter_(other.stream_color_counter_.load(::std::memory_order_relaxed))
  {
    other.resources_ = nullptr;
  }

  place_group& operator=(place_group&&)      = delete;
  place_group(const place_group&)            = delete;
  place_group& operator=(const place_group&) = delete;

  ~place_group() = default;

private:
  void init()
  {
    // The machine singleton enables peer access (and memory-pool access)
    // between all device pairs once per process.
    auto& m       = reserved::machine::instance();
    ::std::ignore = m;

    stream_cache_.resize(places_.size());
  }

  // Materialize (lazily, once) the per-place streams from the registry's
  // compute pool. The registry owns the streams; the group only caches
  // handles so (place, color) lookups are stable and cheap.
  const ::std::vector<cudaStream_t>& get_or_create_streams(const exec_place& place)
  {
    // Locate the cache slot for this place.
    size_t idx = 0;
    for (; idx < places_.size(); idx++)
    {
      if (places_[idx] == place)
      {
        break;
      }
    }
    _CCCL_ASSERT(idx < places_.size(), "place_group: place does not belong to this group");

    ::std::lock_guard<::std::mutex> lock(mutex_);
    auto& cache = stream_cache_[idx];
    if (cache.empty())
    {
      cache = place.pick_all_streams(*resources_);
    }
    return cache;
  }

  ::std::vector<exec_place> places_;
  ::std::unique_ptr<exec_place_resources> owned_resources_; // set when owning
  exec_place_resources* resources_ = nullptr; // always valid: owned or borrowed
  ::std::shared_ptr<void> keep_alive_; // keeps a borrowed handle alive

  mutable ::std::mutex mutex_;
  ::std::vector<::std::vector<cudaStream_t>> stream_cache_; // one slot per place
  ::std::atomic<size_t> stream_color_counter_{0};
};
} // namespace cuda::experimental::places
