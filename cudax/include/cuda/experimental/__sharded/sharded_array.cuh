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
 * @brief `sharded_array<T>`: a 1D array partitioned into placed shards.
 *
 * A sharded array is the container of the shared-address rung of the
 * cooperation-scope ladder: every byte has exactly one physical home (a
 * place), while the logical array spans all of them. Algorithms over sharded
 * arrays run each place's piece locally and combine across places through
 * what the rung shares — the common address space.
 *
 * Sharded arrays hold PARTITIONED data only: one home per element. The
 * replicated-operand role belongs to the binding tier (STF `logical_data`),
 * which manages coherent copies; the two compose rather than overlap.
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

#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__places/exec/locality_domain.cuh>
#include <cuda/experimental/__places/localized_array.cuh>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/fork_join.cuh>
#include <cuda/experimental/__sharded/shard.cuh>

#include <algorithm>
#include <functional>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
using ::cuda::experimental::places::exec_place_scope;
using ::cuda::experimental::places::make_locality_domain_grid;
using ::cuda::experimental::places::mv;
using ::cuda::experimental::places::place_group;

/// @brief How a container's memory is owned and released.
enum class ownership
{
  owning_shards, //!< one allocation per shard; each is freed on destruction
  owning_backing, //!< one VMM backing owns all shards' memory (`allocate_contiguous`)
  view //!< a view over memory owned elsewhere (adoption, `slice`)
};

/// @brief Allocation spec for one shard: (size, data place, exec place, stream).
using shard_spec = ::std::tuple<size_t, data_place, exec_place, cudaStream_t>;

/// @brief Whether this machine can back a contiguous allocation
/// (`sharded_array<T>::allocate_contiguous`): the VMM machinery the backing
/// is built on (`places::localized_array`) requires virtual address
/// management support on the device. `allocate_contiguous` throws where this
/// reports false.
[[nodiscard]] inline bool contiguous_backing_supported(int device_ordinal = 0)
{
  cuda_safe_call(cudaFree(nullptr)); // the driver query needs an initialized context
  CUdevice dev;
  cuda_safe_call(cuDeviceGet(&dev, device_ordinal));
  int supported = 0;
  cuda_safe_call(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));
  return supported == 1;
}

/**
 * @brief A 1D array sharded across multiple memory locations.
 *
 * Each shard is a contiguous span of the logical index space with its own
 * placement (see `shard<T>`). The container tier owns placement, adoption and
 * size bookkeeping; algorithms are free functions over the container.
 *
 * Shard sizes are FIXED at allocation time. Operations that would mutate
 * shard sizes (shrinking or redistributing elements) are not performed
 * implicitly; algorithms with size-changing outputs must either write to a
 * separately allocated result or refuse (throw) where the backing cannot
 * represent the result — see `allocate_contiguous`.
 */
template <typename _Tp>
class sharded_array
{
public:
  using value_type     = _Tp;
  using shard_type     = shard<_Tp>;
  using iterator       = typename ::std::vector<shard_type>::iterator;
  using const_iterator = typename ::std::vector<shard_type>::const_iterator;

  // ========== Constructors ==========

  /// @brief Default: empty container.
  sharded_array()
  {
    each_shard.parent_ = this;
  }

  /**
   * @brief Adopt existing shards (non-owning view).
   *
   * The caller keeps ownership of the memory and must keep it alive for the
   * lifetime of the view. Shard capacities are clamped up to their sizes.
   */
  explicit sharded_array(::std::vector<shard_type> shards)
      : shards_(mv(shards))
      , ownership_(ownership::view)
  {
    each_shard.parent_ = this;
    compute_total_size();
    for (auto& s : shards_)
    {
      if (s.capacity < s.size)
      {
        s.capacity = s.size;
      }
    }
  }

  /**
   * @brief Named factory for the adopting constructor above; identical
   * behavior.
   *
   * `adopt` states the naming contract explicitly: a zero-copy wrap of
   * caller-owned memory. The shards' data pointers are used as-is (no
   * allocation, no copy), the container becomes `ownership::view`, and the
   * caller owes the memory's lifetime for as long as the view (or anything
   * sliced from it) is used. By contrast, `from_*` factories build owned
   * storage by copying or transforming their inputs.
   */
  static sharded_array adopt(::std::vector<shard_type> shards)
  {
    return sharded_array(mv(shards));
  }

  // ========== Core allocation ==========

  /**
   * @brief Allocate an owning array with full per-shard specification.
   *
   * This is the primary allocation method; the other allocate methods reduce
   * to it. Each spec provides (size, data place, exec place, stream); the
   * stream, when non-null, becomes the shard's reference stream for
   * stream-ordered operations. Empty specs are skipped.
   */
  static sharded_array allocate(const ::std::vector<shard_spec>& specs)
  {
    places::check_not_capturing(nullptr, "sharded_array::allocate");
    for (const auto& spec : specs)
    {
      if (cudaStream_t stream = ::std::get<3>(spec))
      {
        places::check_not_capturing(stream, "sharded_array::allocate");
      }
    }

    sharded_array arr;
    arr.ownership_ = ownership::owning_shards;

    size_t offset = 0;
    for (const auto& [size, dplace, eplace, stream] : specs)
    {
      if (size == 0)
      {
        // Keep the shard (empty, no storage): shard positions correspond to
        // spec positions -- and to group places in the place_group overloads
        // -- even when a size is zero, the same way compaction leaves
        // emptied shards in place.
        shard_type s;
        s.size          = 0;
        s.capacity      = 0;
        s.global_offset = offset;
        s.place         = dplace;
        s.exec          = eplace;
        s.stream        = stream;
        s.data          = nullptr;
        arr.shards_.push_back(s);
        continue;
      }

      shard_type s;
      s.size          = size;
      s.capacity      = size;
      s.global_offset = offset;
      s.place         = dplace;
      s.exec          = eplace;
      s.stream        = stream;

      // Allocate in the place's context, on the reference stream when given
      exec_place_scope scope(eplace);
      s.data = allocate_memory(size, dplace, stream);

      // Ensure the allocation is complete before use on other streams
      if (stream)
      {
        cuda_safe_call(cudaStreamSynchronize(stream));
      }
      else if (dplace.allocation_is_stream_ordered())
      {
        cuda_safe_call(cudaDeviceSynchronize());
      }

      arr.shards_.push_back(s);
      offset += size;
    }

    arr.total_size_ = offset;
    return arr;
  }

  /// @brief Allocate from (size, data place) pairs; exec places are the
  /// affine ones, no reference streams.
  static sharded_array allocate(const ::std::vector<::std::pair<size_t, data_place>>& specs)
  {
    ::std::vector<shard_spec> full;
    full.reserve(specs.size());
    for (const auto& [size, dplace] : specs)
    {
      full.emplace_back(size, dplace, dplace.affine_exec_place(), nullptr);
    }
    return allocate(full);
  }

  // ========== place_group-based allocation ==========

  /**
   * @brief Allocate with explicit per-shard sizes over a `place_group`.
   *
   * Each shard lives on the affine data place of the corresponding group
   * place and gets a reference stream from the group's per-place pool at the
   * given lane_id (or a round-robin lane_id by default).
   *
   * @throws std::invalid_argument when `sizes.size() != group.size()`.
   */
  static sharded_array
  allocate(place_group& group, const ::std::vector<size_t>& sizes, size_t lane_id = place_group::auto_lane_id)
  {
    if (sizes.size() != group.size())
    {
      _CCCL_THROW(::std::invalid_argument,
                  "sharded_array::allocate: sizes count (" + ::std::to_string(sizes.size())
                    + ") must equal the number of places in the group (" + ::std::to_string(group.size()) + ")");
    }

    const size_t effective_lane = (lane_id == place_group::auto_lane_id) ? group.next_lane_id() : lane_id;

    ::std::vector<shard_spec> specs;
    specs.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++)
    {
      const auto& place = group.place(i);
      specs.emplace_back(sizes[i], place.affine_data_place(), place, group.get_stream(i, effective_lane));
    }
    return allocate(specs);
  }

  /// @brief Allocate `total_size` elements distributed evenly over a group's
  /// places (remainder to the first shards).
  static sharded_array allocate(place_group& group, size_t total_size, size_t lane_id = place_group::auto_lane_id)
  {
    return allocate(group, split_evenly(total_size, group.size()), lane_id);
  }

  // ========== Contiguous (VMM-backed) allocation ==========

  /**
   * @brief Allocate shards as views into ONE contiguous VA range, with each
   *        shard's bytes physically owned by its place (VMM backing via
   *        `places::localized_array`).
   *
   * Use when a consumer needs the array as ONE normal array
   * (`contiguous_data()`) while the shards keep per-place physical placement
   * — e.g. handing a row-partitioned output to an unmodified downstream
   * kernel. The logical shard boundaries are EXACT (`shard(i).data` is
   * exactly `base + global_offset`); only the physical ownership snaps to the
   * device allocation granularity (typically 2 MiB), so up to one granule per
   * boundary lands with a neighboring owner — negligible at the sizes where
   * placement matters, and irrelevant below them.
   *
   * Contract deltas vs `allocate()`:
   *  - contiguity: `contiguous_data()` returns the base pointer; shard views
   *    are offsets into it (dense, no inter-shard padding)
   *  - physical boundaries are approximate (granule-snapped internally)
   *  - fixed size: the VA range and mapping are created once (no resize);
   *    size-mutating algorithms must refuse contiguous arrays, since
   *    shrinking shard sizes would leave gaps between shards' valid elements
   *    and break the read-as-one-array contract
   *
   * One VA range means one physical home per byte, so this backing can never
   * replicate — consistent with the container holding partitioned data only.
   */
  static sharded_array allocate_contiguous(const ::std::vector<shard_spec>& specs)
  {
    places::check_not_capturing(nullptr, "sharded_array::allocate_contiguous");
    for (const auto& spec : specs)
    {
      if (cudaStream_t stream = ::std::get<3>(spec))
      {
        places::check_not_capturing(stream, "sharded_array::allocate_contiguous");
      }
    }

    sharded_array arr;
    arr.ownership_ = ownership::owning_backing; // released via contiguous_backing_

    size_t total = 0;
    ::std::vector<size_t> ends; // cumulative element ends per spec (incl. empty)
    ::std::vector<exec_place> eplaces;
    for (const auto& [size, dplace, eplace, stream] : specs)
    {
      // The contiguous backing places each spec's physical blocks at its
      // exec place's affine data place; a non-affine data_place in the spec
      // would be silently ignored, so refuse it up front.
      if (dplace.to_string() != eplace.affine_data_place().to_string())
      {
        _CCCL_THROW(::std::invalid_argument,
                    "sharded_array::allocate_contiguous: spec data_place (" + dplace.to_string()
                      + ") must be the exec place's affine data place (" + eplace.affine_data_place().to_string()
                      + "); non-affine placement is not supported by the contiguous backing");
      }
      total += size;
      ends.push_back(total);
      eplaces.push_back(eplace);
    }
    if (total == 0)
    {
      return arr;
    }

    // Physical blocks are placed by the owner of each element range: element
    // i belongs to the first spec whose cumulative end exceeds i.
    auto owner_of = [ends](size_t i) {
      const size_t idx = ::std::upper_bound(ends.begin(), ends.end(), i) - ends.begin();
      return places::pos4(static_cast<int>(idx));
    };
    arr.contiguous_backing_ = ::std::make_shared<places::localized_array>(
      places::make_grid(eplaces),
      ::std::function<places::pos4(size_t)>(owner_of),
      total,
      sizeof(_Tp),
      places::dim4(total));

    _Tp* base     = static_cast<_Tp*>(arr.contiguous_backing_->get_base_ptr());
    size_t offset = 0;
    for (const auto& [size, dplace, eplace, stream] : specs)
    {
      if (size == 0)
      {
        // Same invariant as allocate(): the shard exists (empty, a zero-length
        // view at the running offset) so shard positions keep corresponding to
        // spec positions and group places.
        shard_type s;
        s.data          = base + offset;
        s.size          = 0;
        s.capacity      = 0;
        s.global_offset = offset;
        s.place         = dplace;
        s.exec          = eplace;
        s.stream        = stream;
        arr.shards_.push_back(s);
        continue;
      }
      shard_type s;
      s.data          = base + offset;
      s.size          = size;
      s.capacity      = size;
      s.global_offset = offset;
      s.place         = dplace;
      s.exec          = eplace;
      s.stream        = stream;
      arr.shards_.push_back(s);
      offset += size;
    }
    arr.total_size_ = total;
    return arr;
  }

  /// @brief Contiguous allocation distributed evenly over a group's places.
  static sharded_array
  allocate_contiguous(place_group& group, size_t total_size, size_t lane_id = place_group::auto_lane_id)
  {
    const auto sizes            = split_evenly(total_size, group.size());
    const size_t effective_lane = (lane_id == place_group::auto_lane_id) ? group.next_lane_id() : lane_id;

    ::std::vector<shard_spec> specs;
    specs.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); i++)
    {
      const auto& place = group.place(i);
      specs.emplace_back(sizes[i], place.affine_data_place(), place, group.get_stream(i, effective_lane));
    }
    return allocate_contiguous(specs);
  }

  /// @brief True when the whole array is one contiguous VA range (`allocate_contiguous`).
  bool is_contiguous() const
  {
    return contiguous_backing_ != nullptr;
  }

  /// @brief Base pointer of the contiguous range (`nullptr` unless `allocate_contiguous`).
  _Tp* contiguous_data() const
  {
    return contiguous_backing_ ? static_cast<_Tp*>(contiguous_backing_->get_base_ptr()) : nullptr;
  }

  // ========== Convenience allocation ==========

  /// @brief Allocate evenly across the listed devices (owning).
  static sharded_array allocate_uniform(size_t total_size, const ::std::vector<int>& device_ids)
  {
    if (device_ids.empty() || total_size == 0)
    {
      return sharded_array();
    }

    const auto sizes = split_evenly(total_size, device_ids.size());
    ::std::vector<::std::pair<size_t, data_place>> specs;
    specs.reserve(device_ids.size());
    for (size_t i = 0; i < device_ids.size(); i++)
    {
      specs.emplace_back(sizes[i], data_place::device(device_ids[i]));
    }
    return allocate(specs);
  }

  /// @brief Allocate a single host (pinned) shard.
  static sharded_array allocate_host(size_t total_size)
  {
    return allocate(::std::vector<::std::pair<size_t, data_place>>{{total_size, data_place::host()}});
  }

  /// @brief Allocate a single managed-memory shard.
  static sharded_array allocate_managed(size_t total_size)
  {
    return allocate(::std::vector<::std::pair<size_t, data_place>>{{total_size, data_place::managed()}});
  }

  /**
   * @brief Allocate with the same shard layout (sizes, places, streams) as
   * another array, possibly with a different element type.
   */
  template <typename _Up>
  static sharded_array allocate_like(const sharded_array<_Up>& other)
  {
    if (other.empty())
    {
      return sharded_array();
    }
    ::std::vector<shard_spec> specs;
    specs.reserve(other.num_shards());
    for (size_t i = 0; i < other.num_shards(); i++)
    {
      const auto& s = other.shard(i);
      specs.emplace_back(s.size, s.place, s.exec, s.stream);
    }
    return allocate(specs);
  }

  // ========== Host transfer ==========

  /// @brief Allocate per specs and copy from contiguous host data (synchronous).
  static sharded_array from_host(const _Tp* host_data, const ::std::vector<::std::pair<size_t, data_place>>& specs)
  {
    auto arr = allocate(specs);
    arr.copy_from_host(host_data);
    return arr;
  }

  /// @brief Allocate evenly over a group's places and copy from host (synchronous).
  static sharded_array from_host(place_group& group, const _Tp* host_data, size_t total_size)
  {
    auto arr = allocate(group, total_size);
    arr.copy_from_host(host_data);
    return arr;
  }

  /**
   * @brief Copy contiguous host data into every shard (each shard reads from
   * `host_data + global_offset`). SYNCHRONOUS: blocks until all copies complete.
   */
  void copy_from_host(const _Tp* host_data)
  {
    check_not_capturing_any("sharded_array::copy_from_host");
    for (auto& s : shards_)
    {
      if (s.size == 0)
      {
        continue;
      }
      exec_place_scope scope(s.exec);
      if (s.stream)
      {
        cuda_safe_call(
          cudaMemcpyAsync(s.data, host_data + s.global_offset, s.size_bytes(), cudaMemcpyDefault, s.stream));
      }
      else
      {
        cuda_safe_call(cudaMemcpy(s.data, host_data + s.global_offset, s.size_bytes(), cudaMemcpyDefault));
      }
    }
    sync();
  }

  /**
   * @brief Copy every shard back to contiguous host memory (each shard writes
   * to `host_data + global_offset`). SYNCHRONOUS.
   */
  void copy_to_host(_Tp* host_data) const
  {
    check_not_capturing_any("sharded_array::copy_to_host");
    // Asynchronous per-shard copies + one join: shards copy concurrently
    // (mirrors copy_from_host; a pinned destination gets the overlap, a
    // pageable one degrades per-copy but stays correct and ordered).
    each_shard->*[host_data](const auto& s) {
      if (s.stream)
      {
        cuda_safe_call(
          cudaMemcpyAsync(host_data + s.global_offset, s.data, s.size_bytes(), cudaMemcpyDefault, s.stream));
      }
      else
      {
        cuda_safe_call(cudaMemcpy(host_data + s.global_offset, s.data, s.size_bytes(), cudaMemcpyDefault));
      }
    };
    sync(); // join all shard streams (the SYNCHRONOUS contract)
  }

  // ==========================================================================
  // Shard visitation
  // ==========================================================================
  //
  //   data.each_shard->*[](auto& s) { ... };            // per shard
  //   data.each_shard->*[](size_t i, auto& s) { ... };  // index-aware
  //
  // The shard's exec place is activated around each call; empty shards are
  // skipped. Work is enqueued in shard order; use per-shard streams for
  // overlap and `sync()` to wait.

  /// @brief Proxy for shard visitation (see above).
  class each_shard_visitor
  {
    sharded_array* parent_;
    friend class sharded_array;

    template <typename _Parent, typename _Fn>
    static void impl(_Parent& parent, _Fn&& func)
    {
      for (size_t i = 0; i < parent.num_shards(); ++i)
      {
        auto& s = parent.shard(i);
        if (s.size == 0)
        {
          continue;
        }
        exec_place_scope scope(s.exec);
        if constexpr (::cuda::std::is_invocable_v<_Fn, size_t, decltype(s)>)
        {
          func(i, s);
        }
        else
        {
          func(s);
        }
      }
    }

  public:
    /// @brief Visit shards with mutable access.
    template <typename _Fn>
    void operator->*(_Fn&& func)
    {
      impl(*parent_, ::std::forward<_Fn>(func));
    }

    /// @brief Visit shards with read-only access.
    template <typename _Fn>
    void operator->*(_Fn&& func) const
    {
      impl(::std::as_const(*parent_), ::std::forward<_Fn>(func));
    }
  };

  /// @brief Shard visitation entry point: `arr.each_shard->*functor`.
  each_shard_visitor each_shard;

  /// @brief Synchronize one shard's reference stream (in the shard's
  /// execution context). Prefer this over synchronizing the raw stream: it
  /// activates the shard's exec place the way every other shard operation
  /// does.
  /// @throws std::runtime_error under an active CUDA stream capture.
  void sync(size_t shard_idx) const
  {
    const auto& s = shard(shard_idx);
    places::check_not_capturing(nullptr, "sharded_array::sync");
    if (s.stream)
    {
      places::check_not_capturing(s.stream, "sharded_array::sync");
      exec_place_scope scope(s.exec);
      cuda_safe_call(cudaStreamSynchronize(s.stream));
    }
  }

  /// @brief Synchronize every shard's reference stream.
  /// @throws std::runtime_error under an active CUDA stream capture
  /// (synchronization cannot be recorded into a graph; the capture stays
  /// valid, so pass `blocking = false` to the elementwise algorithms and
  /// synchronize outside capture instead).
  void sync() const
  {
    check_not_capturing_any("sharded_array::sync");
    for (size_t i = 0; i < shards_.size(); i++)
    {
      sync(i);
    }
  }

  // ========== Stream ordering: composing with a caller stream ==========

  /**
   * @brief Declare that the shards' subsequent work depends on the work
   *        currently enqueued on @p stream (fork a caller stream out to the
   *        per-shard streams).
   *
   * ORDERING DECLARATION, NOT A SYNCHRONIZATION: one event is recorded on
   * @p stream and every shard stream waits on it. The host returns
   * immediately; nothing is synchronized and no work is awaited. Use it to
   * hand results produced on a caller stream to per-shard consumers without
   * a host round-trip:
   *
   * @code
   * producer<<<..., s>>>(...);   // writes the array's memory on stream s
   * data.fork_from(s);           // shard streams now depend on the producer
   * transform(group, data, op);  // per-shard work sees the produced values
   * @endcode
   *
   * Capture-safe: inside an active CUDA graph capture the event record/wait
   * pair becomes graph dependencies, making `fork_from`/`join_into` the
   * composition idiom between a captured caller stream and the shard streams.
   *
   * Events are drawn from a small pool owned by the container (lazily
   * created, reused across calls; see `reserved::fork_join_event_pool` for the
   * ownership rationale), so adopted arrays with foreign streams are fully
   * supported. Shards without a reference stream are skipped: their
   * operations are synchronous and need no ordering. Concurrent
   * `fork_from`/`join_into` calls on the same container reuse the pooled
   * events and must be ordered externally.
   */
  void fork_from(cudaStream_t stream) const
  {
    cudaEvent_t event = nullptr; // recorded once, on the first distinct shard stream
    for (const auto& s : shards_)
    {
      if (!s.stream || s.stream == stream)
      {
        continue;
      }
      if (!event)
      {
        int device                             = -1;
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (stream)
        {
          cuda_safe_call(cudaStreamIsCapturing(stream, &capture_status));
        }
        if (stream && capture_status == cudaStreamCaptureStatusNone)
        {
          // stream_ref::device() is version-portable (cudaStreamGetDevice
          // itself requires CUDA 12.8+); green-context streams report their
          // underlying device, which is exactly the event-pool key we want.
          device = ::cuda::stream_ref{stream}.device().get();
        }
        else
        {
          // Under capture, querying a stream's device is not permitted on
          // every driver (CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED); the event
          // recorded on a capturing stream only becomes a graph dependency
          // node, so the current device is the right home for it.
          cuda_safe_call(cudaGetDevice(&device));
        }
        event = fork_join_events_.fork_event(device);
        cuda_safe_call(cudaEventRecord(event, stream));
      }
      cuda_safe_call(cudaStreamWaitEvent(s.stream, event, 0));
    }
  }

  /**
   * @brief Declare that subsequent work on @p stream depends on the work
   *        currently enqueued on every shard stream (join the per-shard
   *        streams back into a caller stream).
   *
   * ORDERING DECLARATION, NOT A SYNCHRONIZATION: an event is recorded on each
   * shard stream and @p stream waits on all of them. The host returns
   * immediately; nothing is synchronized. The mirror image of `fork_from`:
   *
   * @code
   * transform(group, data, op);  // per-shard producers on the shard streams
   * data.join_into(s);           // stream s now depends on all of them
   * reader<<<..., s>>>(...);     // sees every shard's results
   * @endcode
   *
   * Capture-safe and pool-backed exactly like `fork_from` (see there for the
   * event-ownership and concurrency notes).
   */
  void join_into(cudaStream_t stream) const
  {
    for (size_t i = 0; i < shards_.size(); i++)
    {
      const auto& s = shards_[i];
      if (!s.stream || s.stream == stream)
      {
        continue;
      }
      cudaEvent_t event = nullptr;
      {
        // Create/record in the shard's context so the event matches its stream.
        exec_place_scope scope(s.exec);
        event = fork_join_events_.join_event(i);
        cuda_safe_call(cudaEventRecord(event, s.stream));
      }
      cuda_safe_call(cudaStreamWaitEvent(stream, event, 0));
    }
  }

  // ========== Move-only semantics ==========

  sharded_array(sharded_array&& other) noexcept
      : shards_(mv(other.shards_))
      , total_size_(other.total_size_)
      , ownership_(other.ownership_)
      , contiguous_backing_(mv(other.contiguous_backing_))
      , fork_join_events_(mv(other.fork_join_events_))
  {
    each_shard.parent_ = this;
    other.total_size_  = 0;
    other.ownership_   = ownership::view;
  }

  sharded_array& operator=(sharded_array&& other) noexcept
  {
    if (this != &other)
    {
      clear();
      shards_             = mv(other.shards_);
      total_size_         = other.total_size_;
      ownership_          = other.ownership_;
      contiguous_backing_ = mv(other.contiguous_backing_);
      fork_join_events_   = mv(other.fork_join_events_);
      other.total_size_   = 0;
      other.ownership_    = ownership::view;
      // each_shard.parent_ already points to this
    }
    return *this;
  }

  sharded_array(const sharded_array&)            = delete;
  sharded_array& operator=(const sharded_array&) = delete;

  ~sharded_array()
  {
    clear();
  }

  // ========== Size and shard access ==========

  size_t size() const
  {
    return total_size_;
  }
  size_t size_bytes() const
  {
    return total_size_ * sizeof(_Tp);
  }
  bool empty() const
  {
    return total_size_ == 0;
  }
  size_t num_shards() const
  {
    return shards_.size();
  }

  shard_type& shard(size_t idx)
  {
    _CCCL_ASSERT(idx < shards_.size(), "sharded_array: shard index out of range");
    return shards_[idx];
  }

  const shard_type& shard(size_t idx) const
  {
    _CCCL_ASSERT(idx < shards_.size(), "sharded_array: shard index out of range");
    return shards_[idx];
  }

  shard_type& operator[](size_t idx)
  {
    return shard(idx);
  }
  const shard_type& operator[](size_t idx) const
  {
    return shard(idx);
  }

  iterator begin()
  {
    return shards_.begin();
  }
  iterator end()
  {
    return shards_.end();
  }
  const_iterator begin() const
  {
    return shards_.begin();
  }
  const_iterator end() const
  {
    return shards_.end();
  }
  const_iterator cbegin() const
  {
    return shards_.cbegin();
  }
  const_iterator cend() const
  {
    return shards_.cend();
  }

  // ========== Slicing (non-owning views) ==========

  static constexpr size_t npos = static_cast<size_t>(-1);

  /**
   * @brief Non-owning view of elements [start, end) across shards.
   *
   * Preserves shard count and place correspondence: when the slice does not
   * overlap a shard, an empty shard (size 0) keeps the position, so the i-th
   * shard of the view still corresponds to the i-th place of the source.
   */
  sharded_array slice(size_t start, size_t end = npos) const
  {
    if (end == npos)
    {
      end = total_size_;
    }
    end   = ::std::min(end, total_size_);
    start = ::std::min(start, end);

    ::std::vector<shard_type> new_shards(shards_.size());
    size_t current_pos = 0;
    size_t new_offset  = 0;

    for (size_t i = 0; i < shards_.size(); i++)
    {
      const auto& src        = shards_[i];
      const size_t shard_end = current_pos + src.size;

      const size_t overlap_start = ::std::max(current_pos, start);
      const size_t overlap_end   = ::std::min(shard_end, end);

      shard_type& s   = new_shards[i];
      s.place         = src.place;
      s.exec          = src.exec;
      s.stream        = src.stream;
      s.global_offset = new_offset;

      if (overlap_start < overlap_end)
      {
        const size_t local_start = overlap_start - current_pos;
        const size_t count       = overlap_end - overlap_start;
        s.data                   = src.data + local_start;
        s.size                   = count;
        s.capacity               = count;
        new_offset += count;
      }
      else
      {
        s.data     = nullptr;
        s.size     = 0;
        s.capacity = 0;
      }

      current_pos = shard_end;
    }

    return sharded_array(mv(new_shards)); // non-owning
  }

  // ========== Ownership ==========

  ownership get_ownership() const
  {
    return ownership_;
  }
  bool is_owning() const
  {
    return ownership_ != ownership::view;
  }
  bool is_view() const
  {
    return ownership_ == ownership::view;
  }

  /// @brief Release ownership: the caller becomes responsible for the memory.
  ///
  /// Only meaningful for `ownership::owning_shards` (per-shard allocations
  /// the caller can free individually). A contiguous (VMM) backing cannot be
  /// handed over through raw pointers — the mapping dies with the backing —
  /// so releasing it is refused.
  ///
  /// @throws std::invalid_argument for `ownership::owning_backing`
  void release()
  {
    if (ownership_ == ownership::owning_backing)
    {
      _CCCL_THROW(::std::invalid_argument,
                  "sharded_array::release: a contiguous (VMM) backing cannot be released through "
                  "raw pointers; keep the array alive instead");
    }
    ownership_ = ownership::view;
  }

  // ========== Capacity bookkeeping ==========

  /// @brief Total capacity across all shards.
  size_t total_capacity() const
  {
    size_t cap = 0;
    for (const auto& s : shards_)
    {
      cap += s.capacity;
    }
    return cap;
  }

  /// @brief Reset every shard's size to its capacity (buffer reuse after
  /// shrinking operations) and rebuild the offsets.
  void reset_sizes_to_capacity()
  {
    for (auto& s : shards_)
    {
      s.size = s.capacity;
    }
    recalculate_offsets();
  }

  /**
   * @brief Atomically set every shard's logical size and re-tile the global
   * offsets: the owning structure's size-mutation verb.
   *
   * The view invariants (regions disjoint, ordered, exactly tiling
   * `[0, total_size())`) hold before and after; there is no observable
   * intermediate state. Capacities are unchanged; each new size must not
   * exceed the shard's capacity. Refused on contiguous backing (the flat
   * view's element order could not survive per-shard shrinkage).
   *
   * @throws std::invalid_argument on size/count mismatch, capacity overflow,
   *         or contiguous backing.
   */
  void commit_sizes(const ::std::vector<size_t>& new_sizes)
  {
    if (new_sizes.size() != shards_.size())
    {
      _CCCL_THROW(::std::invalid_argument, "sharded_array::commit_sizes: one size per shard required");
    }
    if (is_contiguous())
    {
      _CCCL_THROW(::std::invalid_argument, "sharded_array::commit_sizes: not supported on contiguous backing");
    }
    for (size_t i = 0; i < shards_.size(); i++)
    {
      if (new_sizes[i] > shards_[i].capacity)
      {
        _CCCL_THROW(::std::invalid_argument, "sharded_array::commit_sizes: new size exceeds shard capacity");
      }
    }
    for (size_t i = 0; i < shards_.size(); i++)
    {
      shards_[i].size = new_sizes[i];
    }
    recalculate_offsets();
  }

  /// @brief Recompute global offsets from current shard sizes. Call after an
  /// operation that legitimately updated shard sizes.
  void recalculate_offsets()
  {
    size_t offset = 0;
    for (auto& s : shards_)
    {
      s.global_offset = offset;
      offset += s.size;
    }
    total_size_ = offset;
  }

  // ========== Validation ==========

  /**
   * @brief Check shard-metadata consistency: sequential offsets, size <=
   * capacity, non-null data for non-empty shards, and total-size agreement.
   *
   * @param error_stream optional stream receiving a description of each
   *                     violation
   * @return true when consistent
   */
  bool validate(::std::ostream* error_stream = nullptr) const
  {
    size_t expected_offset = 0;
    size_t sum_sizes       = 0;
    bool valid             = true;

    for (size_t i = 0; i < shards_.size(); i++)
    {
      const auto& s = shards_[i];

      if (s.global_offset != expected_offset)
      {
        if (error_stream)
        {
          *error_stream
            << "shard " << i << ": expected offset " << expected_offset << " but got " << s.global_offset << "\n";
        }
        valid = false;
      }

      if (s.size > s.capacity)
      {
        if (error_stream)
        {
          *error_stream << "shard " << i << ": size (" << s.size << ") > capacity (" << s.capacity << ")\n";
        }
        valid = false;
      }

      if (s.size > 0 && s.data == nullptr)
      {
        if (error_stream)
        {
          *error_stream << "shard " << i << ": null data pointer for non-empty shard\n";
        }
        valid = false;
      }

      expected_offset += s.size;
      sum_sizes += s.size;
    }

    if (sum_sizes != total_size_)
    {
      if (error_stream)
      {
        *error_stream << "sum of shard sizes (" << sum_sizes << ") != total size (" << total_size_ << ")\n";
      }
      valid = false;
    }

    return valid;
  }

  /// @brief `validate()` that throws `std::runtime_error` with details on failure.
  void validate_or_throw(const ::std::string& context = "") const
  {
    ::std::ostringstream errors;
    if (!validate(&errors))
    {
      const ::std::string msg =
        context.empty() ? "sharded_array validation failed:\n" : context + " - sharded_array validation failed:\n";
      _CCCL_THROW(::std::runtime_error, msg + errors.str());
    }
  }

  // ========== Modification ==========

  /// @brief Drop all shards (freeing their memory when owning) and release
  /// the contiguous backing, if any.
  void clear()
  {
    if (ownership_ == ownership::owning_shards)
    {
      for (auto& s : shards_)
      {
        free_memory(s);
      }
    }
    shards_.clear();
    total_size_ = 0;
    contiguous_backing_.reset(); // unmaps + releases the VMM range, if any
  }

private:
  /// Refuse synchronous/host-side member operations under an active capture:
  /// probes a global-mode capture anywhere in the process (legacy stream) and
  /// each shard's reference stream. Throws without touching the capture.
  void check_not_capturing_any(const char* what) const
  {
    places::check_not_capturing(nullptr, what);
    for (const auto& s : shards_)
    {
      if (s.stream)
      {
        places::check_not_capturing(s.stream, what);
      }
    }
  }

  static ::std::vector<size_t> split_evenly(size_t total_size, size_t parts)
  {
    ::std::vector<size_t> sizes(parts, 0);
    if (parts == 0)
    {
      return sizes;
    }
    const size_t base      = total_size / parts;
    const size_t remainder = total_size % parts;
    for (size_t i = 0; i < parts; i++)
    {
      sizes[i] = base + (i < remainder ? 1 : 0);
    }
    return sizes;
  }

  void compute_total_size()
  {
    total_size_ = 0;
    for (const auto& s : shards_)
    {
      total_size_ += s.size;
    }
  }

  static _Tp* allocate_memory(size_t count, const data_place& place, cudaStream_t stream = nullptr)
  {
    if (count == 0)
    {
      return nullptr;
    }
    return static_cast<_Tp*>(place.allocate(static_cast<::std::ptrdiff_t>(count * sizeof(_Tp)), stream));
  }

  static void free_memory(shard_type& s)
  {
    if (!s.data)
    {
      return;
    }
    s.place.deallocate(s.data, s.capacity * sizeof(_Tp), s.stream);
    s.data = nullptr;
  }

  ::std::vector<shard_type> shards_;
  size_t total_size_   = 0;
  ownership ownership_ = ownership::view;
  // Set only by allocate_contiguous: the VMM backing (one VA range, physical
  // blocks owned per shard's place). Shards are then non-owning views.
  ::std::shared_ptr<places::localized_array> contiguous_backing_;
  // Pooled events for fork_from/join_into (lazily created; mutable because
  // the ordering declarations are const — they do not modify elements).
  mutable reserved::fork_join_event_pool fork_join_events_;
};

namespace reserved
{
/**
 * @brief Refuse a synchronous sharded algorithm under an active CUDA stream
 * capture: probes a global-mode capture anywhere in the process (legacy
 * stream) and every shard's reference stream (delegating to
 * `places::check_not_capturing`). Safe query; throwing leaves the capture
 * valid.
 */
template <typename _Tp>
void check_not_capturing(const sharded_array<_Tp>& data, const char* what)
{
  places::check_not_capturing(nullptr, what);
  for (const auto& s : data)
  {
    if (s.stream)
    {
      places::check_not_capturing(s.stream, what);
    }
  }
}
} // namespace reserved

// ============================================================================
// Compatibility checks
// ============================================================================

/// @brief Validate that an array has one shard per place of a group.
/// @throws std::invalid_argument on mismatch
template <typename _Tp>
void check_places(const sharded_array<_Tp>& arr, const place_group& group, const char* context = "operation")
{
  if (arr.num_shards() != group.size())
  {
    _CCCL_THROW(::std::invalid_argument,
                ::std::string(context) + ": shard count (" + ::std::to_string(arr.num_shards())
                  + ") doesn't match the number of places (" + ::std::to_string(group.size()) + ")");
  }
}

// ============================================================================
// copy_between: copy/redistribute between arbitrary shard layouts
// ============================================================================

/**
 * @brief Copy `src` to `dst`, handling arbitrary shard layouts (different
 * sizes, counts and placements). Peer access between the devices involved
 * must be available. SYNCHRONOUS.
 */
template <typename _Tp>
void copy_between(const sharded_array<_Tp>& src, sharded_array<_Tp>& dst)
{
  if (src.size() == 0 || dst.size() == 0)
  {
    return;
  }

  reserved::check_not_capturing(src, "sharded::copy_between");
  reserved::check_not_capturing(dst, "sharded::copy_between");

  for (auto& dst_shard : dst)
  {
    const size_t dst_start = dst_shard.global_offset;
    const size_t dst_end   = dst_start + dst_shard.size;

    exec_place_scope scope(dst_shard.exec);

    for (const auto& src_shard : src)
    {
      const size_t src_start = src_shard.global_offset;
      const size_t src_end   = src_start + src_shard.size;

      const size_t overlap_start = ::std::max(dst_start, src_start);
      const size_t overlap_end   = ::std::min(dst_end, src_end);

      if (overlap_start >= overlap_end)
      {
        continue;
      }

      const size_t copy_count = overlap_end - overlap_start;
      const _Tp* src_ptr      = src_shard.data + (overlap_start - src_shard.global_offset);
      _Tp* dst_ptr            = dst_shard.data + (overlap_start - dst_shard.global_offset);

      cuda_safe_call(cudaMemcpy(dst_ptr, src_ptr, copy_count * sizeof(_Tp), cudaMemcpyDefault));
    }
  }
}
} // namespace cuda::experimental::sharded
