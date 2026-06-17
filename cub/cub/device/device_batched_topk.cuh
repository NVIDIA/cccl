// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceBatchedTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
//! from many (small) segments of unordered data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/env_dispatch.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_common.cuh> // topk::select::{min, max}
#include <cub/util_type.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/argument>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @cond
//! Shared implementation for all cub::DeviceBatchedTopK entry points.
//!
//! Validates the requested execution requirements and argument annotations, resolves the (optionally tuned) policy
//! selector from the environment, and forwards to the internal batched top-k dispatch. The selection direction is
//! threaded through as a compile-time `cuda::args::constant<SelectDirection>` so the kernel only emits the
//! requested (max OR min) code path.
//!
//! All current API-surface constraints are surfaced here as `static_assert`s so the diagnostic appears at the
//! `cub::DeviceBatchedTopK` call site rather than deep inside the kernel/agent instantiation.
template <topk::select SelectDirection,
          typename KeyInputIteratorItT,
          typename KeyOutputIteratorItT,
          typename ValueInputIteratorItT,
          typename ValueOutputIteratorItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename NumSegmentsParameterT,
          typename EnvT>
CUB_RUNTIME_FUNCTION static cudaError_t dispatch_batched_topk(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputIteratorItT d_keys_in,
  KeyOutputIteratorItT d_keys_out,
  ValueInputIteratorItT d_values_in,
  ValueOutputIteratorItT d_values_out,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  NumSegmentsParameterT num_segments,
  EnvT env)
{
  // ---------------------------------------------------------------------------
  // Execution requirements (mirrors cub::DeviceTopK): non-deterministic + unsorted output only.
  // ---------------------------------------------------------------------------
  static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                "Determinism should be used inside requires to have an effect.");
  using requirements_t = ::cuda::std::execution::
    __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
  using requested_determinism_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::determinism::__get_determinism_t,
                                                ::cuda::execution::determinism::run_to_run_t>;
  using requested_order_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::output_ordering::__get_output_ordering_t,
                                                ::cuda::execution::output_ordering::sorted_t>;
  static_assert(::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::not_guaranteed_t>
                  && ::cuda::std::is_same_v<requested_order_t, ::cuda::execution::output_ordering::unsorted_t>,
                "cub::DeviceBatchedTopK only supports non-deterministic, unsorted output. Acknowledge this by "
                "passing cuda::execution::require(cuda::execution::determinism::not_guaranteed, "
                "cuda::execution::output_ordering::unsorted) in the environment.");

  // A tie-break requirement only constrains *which* of the elements that compare equal at the K-th position are
  // selected (per segment); it is meaningless without determinism, so it must be paired with a deterministic
  // requirement (run_to_run / gpu_to_gpu). Deterministic execution and tie-breaking are not implemented yet; this only
  // validates the requirement combination so the eventual behavior is wired up.
  using requested_tie_break_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::tie_break::__get_tie_break_t,
                                                ::cuda::execution::tie_break::unspecified_t>;
  static_assert(
    ::cuda::std::is_same_v<requested_tie_break_t, ::cuda::execution::tie_break::unspecified_t>
      || !::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::not_guaranteed_t>,
    "cub::DeviceBatchedTopK: a tie_break requirement (cuda::execution::tie_break::prefer_smaller_index or "
    "prefer_larger_index) requires a deterministic execution requirement "
    "(cuda::execution::determinism::run_to_run or gpu_to_gpu); it cannot be combined with "
    "cuda::execution::determinism::not_guaranteed.");

  // ---------------------------------------------------------------------------
  // Resolve the (optionally tuned) policy selector from the environment.
  // ---------------------------------------------------------------------------
  using key_t                     = it_value_t<it_value_t<KeyInputIteratorItT>>;
  using value_t                   = it_value_t<it_value_t<ValueInputIteratorItT>>;
  using default_policy_selector_t = batched_topk::
    policy_selector_from_types<key_t, value_t, ::cuda::std::int64_t, ::cuda::args::__traits<KParameterT>::highest>;
  using tuning_env_t =
    ::cuda::__call_result_or_t<::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>, EnvT>;
  using policy_selector_t = ::cuda::std::execution::
    __query_result_or_t<tuning_env_t, batched_topk::batched_topk_policy, default_policy_selector_t>;

  // ---------------------------------------------------------------------------
  // Argument-annotation constraints surfaced at the call site.
  // ---------------------------------------------------------------------------
  static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value,
                "cub::DeviceBatchedTopK currently requires a single (uniform) number of segments resolved on the "
                "host; pass num_segments as a single-value annotation (e.g. cuda::args::constant or "
                "cuda::args::immediate), not a per-segment sequence.");

  const auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env);

  // The total-number-of-items guarantee is intentionally not part of the initial public API surface. The dispatch
  // only uses its element type to size internal large-segment offsets (the value itself is unused), so we pass a
  // conservative 64-bit upper bound here.
  const auto total_num_items = ::cuda::args::immediate{::cuda::std::numeric_limits<::cuda::std::int64_t>::max()};

  return batched_topk::dispatch(
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    segment_sizes,
    k,
    ::cuda::args::constant<SelectDirection>{},
    num_segments,
    total_num_items,
    stream.get(),
    policy_selector_t{});
}
//! @endcond
} // namespace detail

//! @rst
//! DeviceBatchedTopK provides device-wide, parallel operations for finding the largest (or smallest) K items from
//! many segments of unordered data items residing within device-accessible memory.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! Given a batch of segments, ``DeviceBatchedTopK`` finds, independently for each segment, the K largest (or
//! smallest) items.
//!
//! Argument annotation framework
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The parameters ``segment_sizes``, ``k``, and ``num_segments`` are passed as **annotated arguments** from
//! ``cuda::args``. An annotation tells the algorithm everything you know about a parameter: where its value comes
//! from and how tightly it is bounded. The more you can tell the algorithm, and the more precisely (a
//! compile-time constant rather than a runtime value, a tight bound rather than a loose one), the more it can
//! specialize. For that reason, we encourage you to provide as much information as you have.
//!
//! **Where the value comes from.** The first three forms describe a single value shared by every segment, the last
//! describes a distinct value per segment:
//!
//! - ``cuda::args::constant<N>{}`` for a value fixed at compile time. ``N`` is both the value and its bound.
//! - ``cuda::args::immediate{value}`` for a single value known on the host at the call.
//! - ``cuda::args::deferred{pointer}`` for a single value read in stream order, for example one produced on the
//!   device by a preceding launch.
//! - ``cuda::args::deferred_sequence{iterator}`` for a distinct value per segment, also read in stream order.
//!
//! **How it is bounded.** A bound lets the algorithm reason about a value it does not know exactly:
//!
//! - A **compile-time** bound, ``cuda::args::bounds<lo, hi>()``, may accompany ``immediate``, ``deferred``, or
//!   ``deferred_sequence`` (a ``constant`` is already its own bound). The kernel specializes on this range and uses
//!   it to size temporary storage (see *Choosing argument bounds*), so prefer the tightest range you can prove.
//! - A **runtime** bound, ``cuda::args::bounds(lo, hi)``, may accompany ``deferred`` and ``deferred_sequence`` when
//!   the range is only known at runtime. A compile-time and a runtime bound can be combined, and the effective range
//!   is then their intersection.
//!
//! **Which form each parameter accepts.** ``segment_sizes`` and ``k`` accept all four forms. The kernel specializes
//! on their upper bound, so each carries a compile-time bound (a ``constant`` is its own, the other forms take an
//! explicit ``cuda::args::bounds<lo, hi>()``). ``num_segments`` is a single value, supplied as ``constant``,
//! ``immediate``, or ``deferred``, never as a per-segment sequence.
//!
//! .. code-block:: c++
//!
//!     // segment_sizes (k is analogous):
//!     cuda::args::constant<256>{};                                           // fixed at compile time
//!     cuda::args::immediate{n, cuda::args::bounds<1, 1024>()};               // host value, at most 1024
//!     cuda::args::deferred_sequence{d_sizes, cuda::args::bounds<1, 1024>()}; // per-segment, each at most 1024
//!
//!     // a single value produced on the device (e.g. num_segments), with a static cap and a tighter runtime cap:
//!     cuda::args::deferred{d_count, cuda::args::bounds<0, 4096>(), cuda::args::bounds(0, n)};
//!
//! Choosing argument bounds
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! Prefer **sharp (tight) upper bounds**, especially for the segment size. The statically-known *maximum* segment size
//! (the upper bound of the ``segment_sizes`` annotation) does more than select the kernel: it can also drive how much
//! temporary storage the algorithm requests. As a rough intuition, the temporary allocation may grow with the number
//! of segments times some factor of the *maximum* segment size, so an unnecessarily loose upper bound can inflate
//! temporary storage even when the actual segments are much smaller. The precise relationship is intentionally left
//! unspecified and may change across releases (temporary-storage handling is an implementation detail). Treat this
//! purely as guidance for choosing bounds rather than as a guarantee.
//!
//! Current constraints (initial API surface)
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! This is an initial, intentionally restricted API surface. The following constraints are enforced at compile time
//! (a ``static_assert`` fires if violated):
//!
//! - **Small segments only.** Every segment must be processable by a single thread block (one worker per segment).
//!   The *statically-known maximum* segment size (the upper bound of the ``segment_sizes`` annotation) must be small
//!   enough that such a block fits within the shared-memory limit. Both uniform (fixed) and variable segment sizes are
//!   supported as long as this maximum is honored.
//! - **Uniform number of segments.** ``num_segments`` must be a single value, never a per-segment sequence.
//! - **Explicit opt-out required for the output guarantees.** The deterministic, stable-sorted default contract
//!   described in *Determinism, tie-breaking, and output ordering* below (and in :ref:`cub-topk-requirements`) is not
//!   yet implemented. Like :cpp:struct:`cub::DeviceTopK`, the caller must currently request non-deterministic,
//!   unsorted output explicitly by passing ``cuda::execution::require(cuda::execution::determinism::not_guaranteed,
//!   cuda::execution::output_ordering::unsorted)`` in the environment.
//!
//! Determinism, tie-breaking, and output ordering
//! +++++++++++++++++++++++++++++++++++++++++++++++
//!
//! Like :cpp:struct:`cub::DeviceTopK`, the result of ``DeviceBatchedTopK`` is governed by two orthogonal execution
//! requirements: *which* items are selected per segment (``cuda::execution::determinism``, optionally refined by
//! ``cuda::execution::tie_break``) and the order in which they are written (``cuda::execution::output_ordering``).
//! When the caller does not opt out, the committed default is the most reproducible behavior: deterministic results
//! (``cuda::execution::determinism::run_to_run``), ties resolved toward the smaller (lower) source index
//! (``cuda::execution::tie_break::prefer_smaller_index``), and stable-sorted output
//! (``cuda::execution::output_ordering::stable_sorted``). Callers opt *out* of these guarantees to obtain faster
//! implementations.
//!
//! See :ref:`cub-topk-requirements` for the full requirement model, worked examples, and guidance on choosing
//! requirements.
//!
//! .. note::
//!
//!    **Current support.** This release only implements the fully opted-out configuration, which must be requested
//!    explicitly: ``cuda::execution::require(cuda::execution::determinism::not_guaranteed,
//!    cuda::execution::output_ordering::unsorted)``. Any other combination (including an empty, no-requirement
//!    environment) is rejected at compile time. In this configuration the per-segment output is unordered and may be
//!    non-deterministic: if multiple items tie at the K-th position, the subset of tied elements returned is not
//!    uniquely defined and may vary between runs.
//!
//! Usage Considerations
//! ++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceBatchedTopK}
//!
//! @endrst
struct DeviceBatchedTopK
{
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Finds, for each segment, the largest K keys from an unordered input sequence of keys.
  //!
  //! .. note::
  //!
  //!    The behavior is undefined if the input and output ranges overlap in any way.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-max-keys
  //!     :end-before: example-end batched-topk-max-keys
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Iterator such that `d_keys_in[i]` yields a random-access iterator to the keys of segment `i`
  //!
  //! @param[out] d_keys_out
  //!   Iterator such that `d_keys_out[i]` yields a random-access output iterator for the top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a small compile-time maximum.
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   Annotated argument providing the number of selected items per segment. Capped per segment to the segment size.
  //!
  //! @param[in] num_segments
  //!   Annotated argument providing the (uniform) number of segments
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `determinism::not_guaranteed` and
  //!   `output_ordering::unsorted`.
  //!   @endrst
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MaxKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceBatchedTopK::MaxKeys");
    return detail::dispatch_batched_topk<detail::topk::select::max>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      static_cast<NullType**>(nullptr),
      static_cast<NullType**>(nullptr),
      segment_sizes,
      k,
      num_segments,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Finds, for each segment, the largest K keys from an unordered input sequence of keys.
  //!
  //! This is an environment-based API that allocates and manages the required temporary storage internally using the
  //! memory resource queried from the environment.
  //!
  //! .. note::
  //!
  //!    The behavior is undefined if the input and output ranges overlap in any way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-max-keys-env
  //!     :end-before: example-end batched-topk-max-keys-env
  //!
  //! @endrst
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MaxKeys(
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MaxKeys");
    return detail::dispatch_with_env(
      env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, [[maybe_unused]] auto stream) {
        return detail::dispatch_batched_topk<detail::topk::select::max>(
          storage,
          bytes,
          d_keys_in,
          d_keys_out,
          static_cast<NullType**>(nullptr),
          static_cast<NullType**>(nullptr),
          segment_sizes,
          k,
          num_segments,
          env);
      });
  }

  //! @rst
  //! Finds, for each segment, the smallest K keys from an unordered input sequence of keys.
  //!
  //! .. note::
  //!
  //!    The behavior is undefined if the input and output ranges overlap in any way.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-min-keys
  //!     :end-before: example-end batched-topk-min-keys
  //!
  //! @endrst
  //!
  //! @copydetails MaxKeys
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MinKeys(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceBatchedTopK::MinKeys");
    return detail::dispatch_batched_topk<detail::topk::select::min>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      static_cast<NullType**>(nullptr),
      static_cast<NullType**>(nullptr),
      segment_sizes,
      k,
      num_segments,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Finds, for each segment, the smallest K keys from an unordered input sequence of keys. Environment-based overload
  //! that allocates temporary storage internally.
  //!
  //! .. note::
  //!
  //!    The behavior is undefined if the input and output ranges overlap in any way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-min-keys-env
  //!     :end-before: example-end batched-topk-min-keys-env
  //!
  //! @endrst
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MinKeys(
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MinKeys");
    return detail::dispatch_with_env(
      env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, [[maybe_unused]] auto stream) {
        return detail::dispatch_batched_topk<detail::topk::select::min>(
          storage,
          bytes,
          d_keys_in,
          d_keys_out,
          static_cast<NullType**>(nullptr),
          static_cast<NullType**>(nullptr),
          segment_sizes,
          k,
          num_segments,
          env);
      });
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Finds, for each segment, the largest K keys and their corresponding values from an unordered input sequence of
  //! key-value pairs.
  //!
  //! .. note::
  //!
  //!    The behavior is undefined if the input and output ranges overlap in any way.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-max-pairs
  //!     :end-before: example-end batched-topk-max-pairs
  //!
  //! @endrst
  //!
  //! @tparam ValueInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-input iterators @iterator
  //!
  //! @tparam ValueOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-output iterators @iterator
  //!
  //! @param[in] d_values_in
  //!   Iterator such that `d_values_in[i]` yields a random-access iterator to the values of segment `i`
  //!
  //! @param[out] d_values_out
  //!   Iterator such that `d_values_out[i]` yields a random-access output iterator for the values corresponding to the
  //!   top-k keys of segment `i`
  //!
  //! @copydetails MaxKeys
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename ValueInputIteratorItT,
            typename ValueOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MaxPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    ValueInputIteratorItT d_values_in,
    ValueOutputIteratorItT d_values_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceBatchedTopK::MaxPairs");
    return detail::dispatch_batched_topk<detail::topk::select::max>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      segment_sizes,
      k,
      num_segments,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Finds, for each segment, the largest K keys and their corresponding values. Environment-based overload that
  //! allocates temporary storage internally.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-max-pairs-env
  //!     :end-before: example-end batched-topk-max-pairs-env
  //!
  //! @endrst
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename ValueInputIteratorItT,
            typename ValueOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MaxPairs(
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    ValueInputIteratorItT d_values_in,
    ValueOutputIteratorItT d_values_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MaxPairs");
    return detail::dispatch_with_env(
      env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, [[maybe_unused]] auto stream) {
        return detail::dispatch_batched_topk<detail::topk::select::max>(
          storage, bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k, num_segments, env);
      });
  }

  //! @rst
  //! Finds, for each segment, the smallest K keys and their corresponding values from an unordered input sequence of
  //! key-value pairs.
  //!
  //! .. note::
  //!
  //!    The behavior is undefined if the input and output ranges overlap in any way.
  //!
  //! - @devicestorage
  //!
  //! A Simple Example
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-min-pairs
  //!     :end-before: example-end batched-topk-min-pairs
  //!
  //! @endrst
  //!
  //! @copydetails MaxPairs
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename ValueInputIteratorItT,
            typename ValueOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MinPairs(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    ValueInputIteratorItT d_values_in,
    ValueOutputIteratorItT d_values_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceBatchedTopK::MinPairs");
    return detail::dispatch_batched_topk<detail::topk::select::min>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      segment_sizes,
      k,
      num_segments,
      ::cuda::std::move(env));
  }

  //! @rst
  //! Finds, for each segment, the smallest K keys and their corresponding values. Environment-based overload that
  //! allocates temporary storage internally.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_batched_topk_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin batched-topk-min-pairs-env
  //!     :end-before: example-end batched-topk-min-pairs-env
  //!
  //! @endrst
  template <typename KeyInputIteratorItT,
            typename KeyOutputIteratorItT,
            typename ValueInputIteratorItT,
            typename ValueOutputIteratorItT,
            typename SegmentSizeParameterT,
            typename KParameterT,
            typename NumSegmentsParameterT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t MinPairs(
    KeyInputIteratorItT d_keys_in,
    KeyOutputIteratorItT d_keys_out,
    ValueInputIteratorItT d_values_in,
    ValueOutputIteratorItT d_values_out,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    NumSegmentsParameterT num_segments,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MinPairs");
    return detail::dispatch_with_env(
      env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, [[maybe_unused]] auto stream) {
        return detail::dispatch_batched_topk<detail::topk::select::min>(
          storage, bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k, num_segments, env);
      });
  }
};

CUB_NAMESPACE_END
