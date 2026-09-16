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
 * WHERE is always spelled with the existing place vocabulary (grids,
 * partitions); a `place_group` only attaches resources to it. WHEN — the
 * ordering of work — is spelled with LANES: a lane is one ordering domain
 * across the group (one stream per place, the same lane id on every place),
 * `group.lane(k)` is the value-typed view of the group on lane k, and lane 0
 * is the default everywhere (a plain `place_group&` converts to its lane-0
 * view). Lane ids are `[0, num_lanes())` and never wrap: sharing a lane
 * between two users of a group is always spelled with the same id, never
 * the accident of a hidden counter or a modulo.
 *
 * @code
 * // Standalone: the group owns its stream pools.
 * place_group group{make_locality_domain_grid()};
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
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__places/exec_place_resources.cuh>
#include <cuda/experimental/__places/machine.cuh>
#include <cuda/experimental/__places/place_memory_resource.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>
#include <cuda/experimental/__stf/utility/cuda_safe_call.cuh>

// Used only by the UNITTEST blocks below, never by the implementation: the
// borrowing tests exercise the seam against a real STF resource handle, and
// the construction tests spell their place layouts with the grid vocabulary.
#ifdef UNITTESTED_FILE
#  include <cuda/experimental/__places/exec/locality_domain.cuh>
#  include <cuda/experimental/__stf/internal/async_resources_handle.cuh>
#endif

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::places
{
// ============================================================================
// reserved: implementation details of place_group
// ============================================================================

namespace reserved
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
// Lane identity query
// ============================================================================

/**
 * @brief Query object for the lane an environment was manufactured on
 * (`place_group::lane_view::envs`, a container's `default_envs`).
 *
 * The value is a `cuda::std::optional<size_t>`: engaged with the lane id for
 * environments born from a `place_group` lane, disengaged for environments
 * whose stream came from elsewhere (an array allocated from explicit
 * `shard_spec`s, an adopted foreign stream). Environment types that do not
 * carry the property at all (foreign environments) read as disengaged
 * through `query_lane_id`.
 */
struct get_lane_id_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, get_lane_id_t>)
  [[nodiscard]] constexpr auto operator()(const _Env& __env) const noexcept
  {
    return __env.query(*this);
  }
};
_CCCL_GLOBAL_CONSTANT get_lane_id_t get_lane_id{};

//! @brief Lane id of an environment; disengaged when the environment carries
//! none (foreign stream) or cannot be asked (foreign environment type).
template <class _Env>
[[nodiscard]] constexpr ::cuda::std::optional<size_t> query_lane_id(const _Env& __env) noexcept
{
  if constexpr (::cuda::std::execution::__queryable_with<_Env, get_lane_id_t>)
  {
    return __env.query(get_lane_id);
  }
  else
  {
    (void) __env;
    return ::cuda::std::nullopt;
  }
}

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
 *
 * Ordering is spelled with lanes (see the Streams and lanes section and
 * `lane_view`): `group.lane(k)` is the group on lane k, plain `group` is lane
 * 0, and the group itself holds no lane-selection state — only places, the
 * registry and a lazily filled stream cache — so it is freely shared by
 * concurrent users.
 */
class place_group
{
public:
  class lane_view;

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
      : place_group(grid.places())
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
  // Streams and lanes
  // ==========================================================================
  // Each place carries a pool of streams (its compute pool in the underlying
  // registry). A "lane" is one ordering domain across the group — one stream
  // per place, the same lane id on every place: work is ordered within a
  // lane and may overlap across lanes. The group has a FIXED number of lanes
  // (`num_lanes()`), uniform over its places; lane ids are `[0, num_lanes())`
  // and never wrap — an out-of-range id is refused rather than aliased onto
  // another lane, so two users of one group share a lane only by naming the
  // same id. Lane 0 is the default everywhere (containers built from a plain
  // `place_group`, `envs()`): same lane means stream-ordered, and concurrency
  // is opt-in and visible (`group.lane(1)`). Streams are created lazily, on
  // first use of each place. (Naming: a lane here is a host-side stream
  // pipeline, not CUDA's intra-warp lane — the granularity gap keeps the
  // homonym unambiguous in context.)

  /// @brief Number of lanes of the group (the same on every place).
  [[nodiscard]] size_t num_lanes() const noexcept
  {
    return num_lanes_;
  }

  /// @brief Get the stream of @p place on lane @p lane_id (default lane 0).
  /// @throws std::out_of_range when `lane_id >= num_lanes()`: lane ids never
  ///         wrap (reduce derived ids against `num_lanes()` explicitly).
  /// @throws std::invalid_argument when @p place is not a member of the group.
  cudaStream_t get_stream(const exec_place& place, size_t lane_id = 0)
  {
    check_lane(lane_id, "place_group::get_stream");
    const auto& streams = get_or_create_streams(place);
    return streams[lane_id];
  }

  /// @brief Get the stream of the idx-th place on lane @p lane_id.
  cudaStream_t get_stream(size_t place_idx, size_t lane_id = 0)
  {
    return get_stream(place(place_idx), lane_id);
  }

  /**
   * @brief Environment combining a stream with a place's memory resource and,
   * when known, the lane the stream belongs to.
   *
   * Suitable for CUB's single-call device algorithms: temporaries are
   * allocated from the place that runs the work. The lane id is engaged for
   * streams drawn from a group lane (`lane_view::env`) and disengaged for
   * foreign streams. (Defined ahead of `lane_view`, which uses its deduced
   * return type.)
   */
  static auto
  env(const data_place& dplace, cudaStream_t stream, ::cuda::std::optional<size_t> lane_id = ::cuda::std::nullopt)
  {
    const auto stream_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{stream}};
    const auto mr_prop   = ::cuda::std::execution::prop{::cuda::mr::get_memory_resource, place_memory_resource(dplace)};
    const auto lane_prop = ::cuda::std::execution::prop{get_lane_id, lane_id};
    return ::cuda::std::execution::env{stream_prop, mr_prop, lane_prop};
  }

  /**
   * @brief The group seen on one lane: the value-typed handle for "these
   * places, ordered on lane `lane_id`".
   *
   * A `lane_view` is what containers are built from (`allocate(group.lane(k),
   * n)`, `adopt(group.lane(k), ...)`) and what manufactures per-shard
   * environments (`group.lane(k).envs()`): every environment carries the
   * place's pool stream on that lane, a memory resource at the place's
   * affine data place, and the lane id (`get_lane_id`). A plain
   * `place_group&` converts implicitly to its lane-0 view, so
   * `allocate(group, n)` means lane 0.
   *
   * The view borrows the group: the group must outlive it and anything built
   * from it. Copyable and cheap (a pointer and an index).
   */
  class lane_view
  {
  public:
    /// @brief View of @p group on lane @p lane_id.
    /// @throws std::out_of_range when `lane_id >= group.num_lanes()`.
    lane_view(place_group& group, size_t lane_id)
        : group_(&group)
        , lane_id_(lane_id)
    {
      group.check_lane(lane_id, "place_group::lane");
    }

    /// @brief The lane-0 view: the default lane of every group (implicit).
    lane_view(place_group& group)
        : lane_view(group, 0)
    {}

    [[nodiscard]] place_group& group() const noexcept
    {
      return *group_;
    }
    [[nodiscard]] size_t lane_id() const noexcept
    {
      return lane_id_;
    }
    [[nodiscard]] size_t size() const noexcept
    {
      return group_->size();
    }
    [[nodiscard]] const ::std::vector<exec_place>& places() const noexcept
    {
      return group_->places();
    }
    [[nodiscard]] const exec_place& place(size_t idx) const
    {
      return group_->place(idx);
    }

    /// @brief The idx-th place's stream on this lane.
    [[nodiscard]] cudaStream_t stream(size_t place_idx) const
    {
      return group_->get_stream(place_idx, lane_id_);
    }

    /// @brief Environment of the idx-th place on this lane: its stream, a
    /// memory resource at its affine data place, and the lane id.
    [[nodiscard]] auto env(size_t place_idx) const
    {
      return place_group::env(place(place_idx).affine_data_place(), stream(place_idx), lane_id_);
    }

    /**
     * @brief One environment per place on this lane: the per-shard
     * environment range the generic sharded algorithms consume
     * (`algo(view, envs, ...)`). This is how execution environments are
     * manufactured from places: e.g.
     * `place_group(exec_place::all_devices()).lane(1).envs()` binds one
     * environment per device on lane 1, streams born in each device's
     * context. The environments borrow the group's pool streams.
     */
    [[nodiscard]] auto envs() const
    {
      ::std::vector<decltype(env(size_t{}))> result;
      result.reserve(size());
      for (size_t i = 0; i < size(); i++)
      {
        result.push_back(env(i));
      }
      return result;
    }

  private:
    place_group* group_;
    size_t lane_id_;
  };

  /// @brief The group on lane @p lane_id (see `lane_view`).
  /// @throws std::out_of_range when `lane_id >= num_lanes()`.
  [[nodiscard]] lane_view lane(size_t lane_id)
  {
    return lane_view(*this, lane_id);
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

  /// @brief Environment for the idx-th place using an explicit (foreign)
  /// stream: no lane id.
  auto env(size_t place_idx, cudaStream_t stream) const
  {
    return env(place(place_idx).affine_data_place(), stream);
  }

  /// @brief Environment for the idx-th place on lane 0.
  auto env(size_t place_idx)
  {
    return lane(0).env(place_idx);
  }

  /// @brief One environment per place on lane @p lane_id (default lane 0):
  /// `lane(lane_id).envs()`.
  [[nodiscard]] auto envs(size_t lane_id = 0)
  {
    return lane(lane_id).envs();
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
      , num_lanes_(other.num_lanes_)
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
  // handles so (place, lane_id) lookups are stable and cheap.
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
      // The lane count is a group invariant: every place must be able to
      // back num_lanes() distinct streams, or lane k would alias lane j on
      // this place only. Refuse rather than wrap.
      if (cache.size() < num_lanes_)
      {
        const size_t have = cache.size();
        cache.clear();
        _CCCL_THROW(::std::runtime_error,
                    "place_group: place " + place.to_string() + " has a stream pool of " + ::std::to_string(have)
                      + " stream(s), fewer than the group's num_lanes() (" + ::std::to_string(num_lanes_)
                      + "); lanes must be uniform across the group");
      }
    }
    return cache;
  }

  void check_lane(size_t lane_id, const char* what) const
  {
    if (lane_id >= num_lanes_)
    {
      _CCCL_THROW(::std::out_of_range,
                  ::std::string(what) + ": lane id " + ::std::to_string(lane_id)
                    + " out of range (num_lanes() = " + ::std::to_string(num_lanes_) + "); lane ids never wrap");
    }
  }

  ::std::vector<exec_place> places_;
  ::std::unique_ptr<exec_place_resources> owned_resources_; // set when owning
  exec_place_resources* resources_ = nullptr; // always valid: owned or borrowed
  ::std::shared_ptr<void> keep_alive_; // keeps a borrowed handle alive

  mutable ::std::mutex mutex_;
  ::std::vector<::std::vector<cudaStream_t>> stream_cache_; // one slot per place
  size_t num_lanes_ = exec_place_default_pool_size; // uniform over the group, never wraps
};

#ifdef UNITTESTED_FILE

UNITTEST("place_group construction from the place vocabulary")
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

  // The all-devices grid covers every visible device
  const size_t ndevs = static_cast<size_t>(cuda_try<cudaGetDeviceCount>());
  place_group g4{exec_place::all_devices()};
  EXPECT(g4.size() == ndevs);

  // The all-devices locality-domain grid covers every domain of every device
  // (>= one place per device even without domain support)
  size_t total_domains = 0;
  for (size_t d = 0; d < ndevs; d++)
  {
    total_domains += locality_domain_count(static_cast<int>(d));
  }
  place_group g5{make_locality_domain_grid()};
  EXPECT(g5.size() == total_domains);
  EXPECT(g5.size() >= ndevs);
};

UNITTEST("place_group per-place stream pools")
{
  place_group group{make_locality_domain_grid()};

  // A stream can be picked and used on every place, for every lane_id
  EXPECT(group.num_lanes() >= 1UL);
  for (size_t i = 0; i < group.size(); i++)
  {
    for (size_t lane_id = 0; lane_id < group.num_lanes(); lane_id++)
    {
      cudaStream_t s = group.get_stream(i, lane_id);
      EXPECT(s != nullptr);
      // Stable: the same (place, lane_id) always yields the same stream
      EXPECT(s == group.get_stream(i, lane_id));

      exec_place_scope scope(group.place(i));
      cuda_safe_call(cudaStreamSynchronize(s));
    }
    // Different lanes are different streams
    EXPECT(group.get_stream(i, 0) != group.get_stream(i, 1));
    // Lane ids never wrap: out of range is refused, not aliased
    bool threw = false;
    try
    {
      ::std::ignore = group.get_stream(i, group.num_lanes());
    }
    catch (const ::std::out_of_range&)
    {
      threw = true;
    }
    EXPECT(threw);
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

UNITTEST("place_group lanes are views")
{
  place_group group{make_locality_domain_grid()};

  // lane(k) is the group on lane k; plain group converts to lane 0
  auto l1                   = group.lane(1);
  place_group::lane_view l0 = group;
  EXPECT(l0.lane_id() == 0UL);
  EXPECT(l1.lane_id() == 1UL);
  EXPECT(&l1.group() == &group);
  EXPECT(l1.size() == group.size());
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(l0.stream(i) == group.get_stream(i, 0));
    EXPECT(l1.stream(i) == group.get_stream(i, 1));
    EXPECT(l0.stream(i) != l1.stream(i));
  }

  // Environments carry the lane's stream and the lane id; envs() is lane 0
  auto e0 = group.envs();
  auto e1 = l1.envs();
  EXPECT(e0.size() == group.size());
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(::cuda::get_stream(e0[i]).get() == group.get_stream(i, 0));
    EXPECT(::cuda::get_stream(e1[i]).get() == group.get_stream(i, 1));
    EXPECT(query_lane_id(e0[i]) == ::cuda::std::optional<size_t>{0});
    EXPECT(query_lane_id(e1[i]) == ::cuda::std::optional<size_t>{1});
  }
  // A foreign stream yields no lane; a foreign environment type reads as none
  EXPECT(!query_lane_id(group.env(0, cudaStream_t{})).has_value());
  const auto foreign = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{group.get_stream(0)}};
  EXPECT(!query_lane_id(foreign).has_value());

  // Out-of-range lanes are refused at the view
  bool threw = false;
  try
  {
    ::std::ignore = group.lane(group.num_lanes());
  }
  catch (const ::std::out_of_range&)
  {
    threw = true;
  }
  EXPECT(threw);
};

UNITTEST("place_group per-place memory resources")
{
  // The resource models the cuda::mr concepts and declares the property set
  // containers built from it inherit.
  static_assert(::cuda::mr::resource<place_memory_resource>);
  static_assert(::cuda::mr::synchronous_resource<place_memory_resource>);
  static_assert(::cuda::mr::resource_with<place_memory_resource, ::cuda::mr::device_accessible>);
  static_assert(::cuda::mr::__has_default_queries<place_memory_resource>);

  place_group group{exec_place::device(0)};

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

#endif // UNITTESTED_FILE
} // namespace cuda::experimental::places
