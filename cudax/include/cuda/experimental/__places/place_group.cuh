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
#include <cuda/experimental/__places/place_memory_resource.cuh>
#include <cuda/experimental/__places/place_partition.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

// Used only by the UNITTEST blocks below, never by the implementation: the
// borrowing tests exercise the seam against a real STF resource handle.
#ifdef UNITTESTED_FILE
#  include <cuda/experimental/__stf/internal/async_resources_handle.cuh>
#endif

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::places
{
// ============================================================================
// reserved: place-list builders backing the ctors and factories
// ============================================================================

namespace reserved
{
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
    device_ids = reserved::all_device_ids();
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
} // namespace reserved

// ============================================================================
// Stream-capture query
// ============================================================================

/**
 * @brief True when @p stream is part of an active CUDA stream capture, or
 * when an active global-mode capture elsewhere in the process would make
 * synchronizing operations on this thread illegal.
 *
 * `nullptr` queries the legacy default stream; because the legacy stream
 * implicitly interacts with every capture in `cudaStreamCaptureModeGlobal`,
 * the query then acts as a process-wide capture probe (the driver reports
 * `cudaErrorStreamCaptureImplicit`, which this function maps to `true`).
 */
inline bool stream_in_capture(cudaStream_t stream = nullptr)
{
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  const cudaError_t res          = cudaStreamIsCapturing(stream, &status);
  if (res == cudaErrorStreamCaptureImplicit)
  {
    (void) cudaGetLastError(); // clear the sticky error
    return true;
  }
  cuda_safe_call(res);
  return status != cudaStreamCaptureStatusNone;
}

/**
 * @brief Throw when @p stream is part of an active CUDA stream capture (or,
 * for `nullptr`, when a global-mode capture is active anywhere in the
 * process): the named operation synchronizes, allocates or performs host
 * transfers, none of which can be recorded into a CUDA graph.
 *
 * The check itself is a safe query: refusing an operation this way leaves the
 * ongoing capture VALID, so the caller can catch the exception and keep
 * capturing supported work.
 */
inline void check_not_capturing(cudaStream_t stream, const char* what)
{
  if (stream_in_capture(stream))
  {
    _CCCL_THROW(::std::runtime_error,
                ::std::string(what)
                  + ": not supported during CUDA graph capture (the operation cannot be "
                    "recorded into a graph; the capture stays valid)");
  }
}

// ============================================================================
// place_group
// ============================================================================

/**
 * @brief A group of execution places plus the execution resources attached to
 * them: lazily initialized per-place stream pools, per-place memory
 * resources, and a type-erased per-place library-state cache (see
 * `lib_state`).
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
      : place_group(reserved::places_from_grid(grid))
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
  template <typename ResourceHandle, typename = ::cuda::std::enable_if_t<reserved::has_place_resources<ResourceHandle>>>
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
      device_ids = reserved::all_device_ids();
    }
    return place_group(reserved::places_from_devices(device_ids));
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
    return place_group(reserved::places_from_locality_domains(mv(device_ids)));
  }

  // ==========================================================================
  // Places
  // ==========================================================================

  [[nodiscard]] const ::std::vector<exec_place>& places() const noexcept
  {
    return places_;
  }

  [[nodiscard]] size_t size() const noexcept
  {
    return places_.size();
  }

  [[nodiscard]] const exec_place& place(size_t idx) const
  {
    _CCCL_ASSERT(idx < places_.size(), "place_group: place index out of range");
    return places_[idx];
  }

  [[nodiscard]] const exec_place& operator[](size_t idx) const
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
  [[nodiscard]] size_t num_stream_colors() const noexcept
  {
    return exec_place_default_pool_size;
  }

  /// @brief Get the stream of @p place for a given color (default color 0).
  ///
  /// Colors wrap modulo the place's ACTUAL stream-pool size (places created
  /// with custom pool sizes may hold fewer or more streams than
  /// `exec_place_default_pool_size`; `num_stream_colors()` reports the
  /// default advertised by the group).
  cudaStream_t get_stream(const exec_place& place, size_t color = 0)
  {
    const auto& streams = get_or_create_streams(place);
    _CCCL_ASSERT(!streams.empty(), "place has an empty stream pool");
    return streams[color % streams.size()];
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
  [[nodiscard]] size_t next_stream_color() noexcept
  {
    return stream_color_counter_.fetch_add(1, ::std::memory_order_relaxed) % exec_place_default_pool_size;
  }

  /// @brief Stream for a place, either at an explicit color or round-robin.
  cudaStream_t get_colored_stream(const exec_place& place, size_t color = auto_stream_color)
  {
    return get_stream(place, color == auto_stream_color ? next_stream_color() : color);
  }

  /// @brief Synchronize every stream created so far, on every place.
  /// @throws std::runtime_error under an active CUDA stream capture
  /// (synchronization cannot be recorded into a graph).
  ///
  /// Lazy by design: places whose pools were never touched are skipped, so
  /// synchronizing does not create streams.
  void sync()
  {
    check_not_capturing(nullptr, "place_group::sync");
    // Snapshot under the lock, synchronize unlocked: a host function
    // enqueued on a cached stream may itself call get_stream() and would
    // deadlock against cudaStreamSynchronize() otherwise.
    ::std::vector<::std::vector<cudaStream_t>> snapshot;
    {
      ::std::lock_guard<::std::mutex> lock(mutex_);
      snapshot = stream_cache_;
    }
    for (size_t i = 0; i < snapshot.size() && i < places_.size(); i++)
    {
      if (snapshot[i].empty())
      {
        continue;
      }
      exec_place_scope scope(places_[i]);
      for (cudaStream_t s : snapshot[i])
      {
        if (stream_in_capture(s))
        {
          _CCCL_THROW(::std::runtime_error, "place_group::sync: not supported during CUDA stream capture");
        }
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
    const auto stream_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{stream}};
    const auto mr_prop = ::cuda::std::execution::prop{::cuda::mr::get_memory_resource, place_memory_resource(dplace)};
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
  [[nodiscard]] exec_place_resources& resources() noexcept
  {
    return *resources_;
  }

  /// @brief True when the group owns its stream-pool registry; false when it
  /// borrows one (e.g. from an STF `async_resources_handle`).
  [[nodiscard]] bool owns_resources() const noexcept
  {
    return owned_resources_ != nullptr;
  }

  // ==========================================================================
  // Per-place library state
  // ==========================================================================

  /**
   * @brief Type-erased per-place library state owned by the group.
   *
   * Vendor-call layers stash their PER-PLACE library state here — objects
   * whose natural scope is "this place, for as long as this resource scope
   * lives", such as a cuSPARSE/cuBLAS handle per place (the `raft::handle_t`
   * precedent). This is the group-scope counterpart of the per-container
   * `lib_state()` slots: handles are place-bound values and belong here;
   * descriptors, plans and workspaces are matrix-bound and stay with the
   * container whose addresses they describe.
   *
   * Entries are keyed by (place index, state type), created lazily on first
   * use, shared by every caller of the same (place, type) slot, and destroyed
   * when the group is destroyed (through a deleter captured at creation, so
   * this header stays vendor-free). `make()` is invoked on first use and must
   * return a `_State*` the group takes ownership of.
   *
   * Lifetime rule: a group must outlive anything that uses its places'
   * resources — containers built over the group, and any per-container state
   * referring to handles cached here (this is already the group contract for
   * streams and memory).
   *
   * Thread-safe: concurrent calls for the same slot yield the same object.
   */
  template <typename _State, typename _Make>
  _State& lib_state(size_t place_idx, _Make&& make)
  {
    _CCCL_ASSERT(place_idx < places_.size(), "place_group: place index out of range");
    ::std::lock_guard<::std::mutex> lock(mutex_);
    auto& slot = lib_state_cache_[place_idx];
    auto it    = slot.find(::std::type_index(typeid(_State)));
    if (it == slot.end())
    {
      _State* s = make();
      it        = slot
                    .emplace(::std::type_index(typeid(_State)),
                             ::std::shared_ptr<void>(s,
                                                     [](void* p) {
                                                delete static_cast<_State*>(p);
                                                     }))
                    .first;
    }
    return *static_cast<_State*>(it->second.get());
  }

  /// @brief True when per-place library state of type `_State` has already
  /// been created for the idx-th place (inspection hook; does not create).
  template <typename _State>
  bool has_lib_state(size_t place_idx) const
  {
    _CCCL_ASSERT(place_idx < places_.size(), "place_group: place index out of range");
    ::std::lock_guard<::std::mutex> lock(mutex_);
    const auto& slot = lib_state_cache_[place_idx];
    return slot.find(::std::type_index(typeid(_State))) != slot.end();
  }

  // Non-copyable and not move-assignable; move-CONSTRUCTIBLE so factories
  // and ownership transfer work. Moving requires exclusive
  // access to the source: no concurrent lazy stream creation (get_stream)
  // may run on `other` during the move.
  place_group(place_group&& other) noexcept
      : places_(mv(other.places_))
      , owned_resources_(mv(other.owned_resources_))
      , resources_(other.resources_)
      , keep_alive_(mv(other.keep_alive_))
      , stream_cache_(mv(other.stream_cache_))
      , lib_state_cache_(mv(other.lib_state_cache_))
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
    lib_state_cache_.resize(places_.size());
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
    if (idx >= places_.size())
    {
      _CCCL_THROW(::std::invalid_argument, "place_group: place does not belong to this group");
    }

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
  // Per-place type-erased library state: one (type -> object) map per place.
  // Objects are destroyed with the group, via deleters captured at creation.
  ::std::vector<::std::unordered_map<::std::type_index, ::std::shared_ptr<void>>> lib_state_cache_;
  ::std::atomic<size_t> stream_color_counter_{0};
};

#ifdef UNITTESTED_FILE

UNITTEST("place_group construction and factories")
{
  // From an explicit vector of places
  place_group g1(::std::vector<exec_place>{exec_place::device(0)});
  EXPECT(g1.size() == 1UL);
  EXPECT(g1.owns_resources());

  // From a grid (flattened) and from a scalar place
  auto grid = make_grid(::std::vector<exec_place>{exec_place::device(0), exec_place::device(0)});
  place_group g2(grid);
  EXPECT(g2.size() == grid.size());

  place_group g3(exec_place::device(0));
  EXPECT(g3.size() == 1UL);

  // by_devices covers every visible device
  const size_t ndevs = static_cast<size_t>(cuda_try<cudaGetDeviceCount>());
  auto g4            = place_group::by_devices();
  EXPECT(g4.size() == ndevs);

  auto g5 = place_group::by_devices({0});
  EXPECT(g5.size() == 1UL);

  // by_locality_domains covers every domain of every device (>= one place
  // per device even without domain support)
  size_t total_domains = 0;
  for (size_t d = 0; d < ndevs; d++)
  {
    total_domains += locality_domain_count(static_cast<int>(d));
  }
  auto g6 = place_group::by_locality_domains();
  EXPECT(g6.size() == total_domains);
  EXPECT(g6.size() >= ndevs);
};

UNITTEST("place_group per-place stream pools")
{
  auto group = place_group::by_locality_domains();

  // A stream can be picked and used on every place, for every color
  EXPECT(group.num_stream_colors() >= 1UL);
  for (size_t i = 0; i < group.size(); i++)
  {
    for (size_t color = 0; color < group.num_stream_colors(); color++)
    {
      cudaStream_t s = group.get_stream(i, color);
      EXPECT(s != nullptr);
      // Stable: the same (place, color) always yields the same stream
      EXPECT(s == group.get_stream(i, color));

      exec_place_scope scope(group.place(i));
      cuda_safe_call(cudaStreamSynchronize(s));
    }
    // Different colors are different streams
    EXPECT(group.get_stream(i, 0) != group.get_stream(i, 1));
  }

  // Streams actually execute work on their place
  for (size_t i = 0; i < group.size(); i++)
  {
    exec_place_scope scope(group.place(i));
    constexpr size_t n = 1024 * sizeof(int);
    auto dplace        = group.place(i).affine_data_place();
    cudaStream_t s     = group.get_stream(i);
    void* ptr          = dplace.allocate(n, s);
    cuda_safe_call(cudaMemsetAsync(ptr, 0xab, n, s));
    cuda_safe_call(cudaStreamSynchronize(s));
    dplace.deallocate(ptr, n, s);
    cuda_safe_call(cudaStreamSynchronize(s));
  }

  group.sync();

  // Two groups over the same places are distinct resource scopes: they own
  // distinct pools, hence distinct streams
  place_group a(exec_place::device(0));
  place_group b(exec_place::device(0));
  EXPECT(a.owns_resources());
  EXPECT(b.owns_resources());
  EXPECT(a.get_stream(0, 0) != b.get_stream(0, 0));
};

UNITTEST("place_group per-place memory resources")
{
  auto group = place_group::by_devices({0});

  auto mr        = group.memory_resource(0);
  cudaStream_t s = group.get_stream(0);

  void* p = mr.allocate(::cuda::stream_ref{s}, 1024);
  EXPECT(p != nullptr);
  mr.deallocate(::cuda::stream_ref{s}, p, 1024);
  cuda_safe_call(cudaStreamSynchronize(s));

  void* q = mr.allocate_sync(2048);
  EXPECT(q != nullptr);
  mr.deallocate_sync(q, 2048);

  // Equality follows the place
  EXPECT(mr == group.memory_resource(0));
  EXPECT(mr != group.memory_resource(data_place::host()));

  // Host resource yields pinned memory usable from device code paths
  auto host_mr = group.memory_resource(data_place::host());
  void* h      = host_mr.allocate_sync(64);
  EXPECT(h != nullptr);
  host_mr.deallocate_sync(h, 64);

  // Contract refusals: zero / unsupported alignment, oversize, foreign place
  EXPECT(!place_memory_resource::is_valid_alignment(0));
  bool threw = false;
  try
  {
    ::std::ignore = mr.allocate_sync(64, 0);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  threw = false;
  try
  {
    ::std::ignore = mr.allocate_sync(64, 2 * ::cuda::mr::default_cuda_malloc_alignment);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  threw = false;
  try
  {
    ::std::ignore = mr.allocate_sync(static_cast<::std::size_t>(PTRDIFF_MAX) + 1);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
  threw = false;
  try
  {
    ::std::ignore = group.get_stream(exec_place::host(), 0); // not a member of this group
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
};

UNITTEST("place_group borrows STF async_resources_handle pools")
{
  using ::cuda::experimental::stf::async_resources_handle;

  async_resources_handle handle;
  ::std::vector<exec_place> places{exec_place::device(0)};

  // Borrowing group: draws its pools from the handle's registry
  place_group borrowed(places, handle);
  EXPECT(!borrowed.owns_resources());
  EXPECT(&borrowed.resources() == &handle.get_place_resources());

  // One pool owner: the borrowed group's pool IS the handle's pool for the
  // same place (compare pool identity through stream_pool::operator==)
  auto& from_group  = borrowed.place(0).get_stream_pool(true, borrowed.resources());
  auto& from_handle = borrowed.place(0).get_stream_pool(true, handle.get_place_resources());
  EXPECT(from_group == from_handle);

  // The streams work
  cudaStream_t s = borrowed.get_stream(0);
  EXPECT(s != nullptr);
  exec_place_scope scope(borrowed.place(0));
  cuda_safe_call(cudaStreamSynchronize(s));

  // An owning group over the same places uses a DIFFERENT pool
  place_group owning(places);
  auto& from_owning = owning.place(0).get_stream_pool(true, owning.resources());
  EXPECT(!(from_owning == from_handle));

  // The low-level borrowing seam (raw registry reference) also works
  place_group raw_borrow(places, handle.get_place_resources());
  EXPECT(!raw_borrow.owns_resources());
  EXPECT(&raw_borrow.resources() == &handle.get_place_resources());
};

UNITTEST("place_group move semantics")
{
  place_group g(exec_place::device(0));
  cudaStream_t s = g.get_stream(0);

  place_group moved(mv(g));
  EXPECT(moved.size() == 1UL);
  EXPECT(moved.owns_resources());
  // The cached stream survives the move
  EXPECT(moved.get_stream(0) == s);
};

// A probe standing in for a per-place library handle: counts constructions
// and destructions, and remembers which place it was made for. Inline
// statics: this only exists in the unittest TU.
struct lib_state_probe
{
  static inline int live      = 0;
  static inline int created   = 0;
  static inline int destroyed = 0;

  static void reset()
  {
    live = created = destroyed = 0;
  }

  size_t place_idx;

  explicit lib_state_probe(size_t idx)
      : place_idx(idx)
  {
    created++;
    live++;
  }

  lib_state_probe(const lib_state_probe&)            = delete;
  lib_state_probe& operator=(const lib_state_probe&) = delete;

  ~lib_state_probe()
  {
    destroyed++;
    live--;
  }
};

// A second cached type: (place, type) keys must not collide across types.
struct lib_state_other
{
  int tag = 7;
};

UNITTEST("place_group lib_state: identity and laziness")
{
  lib_state_probe::reset();

  place_group group = place_group::by_locality_domains();
  const size_t n    = group.size();

  // Nothing is created before first use.
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(!group.has_lib_state<lib_state_probe>(i));
  }
  EXPECT(lib_state_probe::created == 0);

  // First use creates; the same (place, type) always yields the SAME object.
  ::std::vector<lib_state_probe*> first(n);
  for (size_t i = 0; i < n; i++)
  {
    first[i] = &group.lib_state<lib_state_probe>(i, [i] {
      return new lib_state_probe(i);
    });
    EXPECT(group.has_lib_state<lib_state_probe>(i));
    EXPECT(first[i]->place_idx == i);
  }
  EXPECT(lib_state_probe::created == static_cast<int>(n));

  for (size_t i = 0; i < n; i++)
  {
    auto* again = &group.lib_state<lib_state_probe>(i, [i]() -> lib_state_probe* {
      // Must not be invoked: the slot is already populated.
      EXPECT(false);
      return nullptr;
    });
    EXPECT(again == first[i]);
  }
  EXPECT(lib_state_probe::created == static_cast<int>(n));

  // Different places hold different objects.
  for (size_t i = 1; i < n; i++)
  {
    EXPECT(first[i] != first[0]);
  }

  // A different type on the same place is a different slot.
  auto& other = group.lib_state<lib_state_other>(0, [] {
    return new lib_state_other();
  });
  EXPECT(other.tag == 7);
  EXPECT(static_cast<void*>(&other) != static_cast<void*>(first[0]));
  // ...and creating it did not disturb the probe slots.
  EXPECT(lib_state_probe::created == static_cast<int>(n));
};

UNITTEST("place_group lib_state: isolation between groups")
{
  lib_state_probe::reset();

  // Two groups over the same places are distinct resource scopes: their
  // caches do not share state.
  place_group a(exec_place::device(0));
  place_group b(exec_place::device(0));

  auto* in_a = &a.lib_state<lib_state_probe>(0, [] {
    return new lib_state_probe(0);
  });
  EXPECT(!b.has_lib_state<lib_state_probe>(0));
  auto* in_b = &b.lib_state<lib_state_probe>(0, [] {
    return new lib_state_probe(0);
  });
  EXPECT(in_a != in_b);
  EXPECT(lib_state_probe::created == 2);
};

UNITTEST("place_group lib_state: teardown destroys exactly once, with the group")
{
  lib_state_probe::reset();

  {
    place_group group = place_group::by_locality_domains();
    for (size_t i = 0; i < group.size(); i++)
    {
      group.lib_state<lib_state_probe>(i, [i] {
        return new lib_state_probe(i);
      });
    }
    EXPECT(lib_state_probe::live == static_cast<int>(group.size()));
    // Cached objects live for the whole group lifetime...
  }
  // ...and are destroyed exactly once, with the group.
  EXPECT(lib_state_probe::live == 0);
  EXPECT(lib_state_probe::destroyed == lib_state_probe::created);
};

UNITTEST("place_group lib_state: cache survives a move")
{
  lib_state_probe::reset();

  auto* before = static_cast<lib_state_probe*>(nullptr);
  {
    place_group g(exec_place::device(0));
    before = &g.lib_state<lib_state_probe>(0, [] {
      return new lib_state_probe(0);
    });

    place_group moved(mv(g));
    // The cached object survives the move, no re-creation, no double destroy.
    EXPECT(moved.has_lib_state<lib_state_probe>(0));
    auto* after = &moved.lib_state<lib_state_probe>(0, []() -> lib_state_probe* {
      EXPECT(false);
      return nullptr;
    });
    EXPECT(after == before);
    EXPECT(lib_state_probe::created == 1);
  }
  // Global balance after the surviving owner dies.
  EXPECT(lib_state_probe::live == 0);
};

#endif // UNITTESTED_FILE
} // namespace cuda::experimental::places
