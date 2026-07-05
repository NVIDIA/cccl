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
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
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
  // Execution requirements.
  //
  // Two orthogonal concerns govern the result: *which* items are selected (possibly refined by
  // a tie-break preference) and the order in which they are written (output ordering). The committed default contract
  // is the most reproducible behavior (determinism::gpu_to_gpu + tie_break::prefer_smaller_index +
  // output_ordering::stable_sorted). Three rules are validated here:
  //   1. determinism and tie_break must be acknowledged together (both specified, or both omitted default)
  //   2. an explicit tie_break of prefer_smaller_index / prefer_larger_index fully pins the result set across GPUs and
  //      therefore requires determinism::gpu_to_gpu (it cannot be paired with run_to_run or not_guaranteed)
  //   3. only `output_ordering::unsorted` is implemented. Given rules 1/2, this admits the five implemented
  //      (determinism, tie_break) combinations -- (not_guaranteed, unspecified), (run_to_run, unspecified),
  //      (gpu_to_gpu, {unspecified, prefer_smaller_index, prefer_larger_index}) -- while `sorted` / `stable_sorted`
  //      (and therefore the empty-env default, which resolves to `stable_sorted`) remain rejected. Deterministic
  //      requests route to the cluster backend on SM 9.0+; the opt-out configuration uses the arch+size crossover.
  // ---------------------------------------------------------------------------
  static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                "Determinism should be used inside cuda::execution::require to have an effect.");
  static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::tie_break::__get_tie_break_t>,
                "Tie-break should be used inside cuda::execution::require to have an effect.");
  using requirements_t = ::cuda::std::execution::
    __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;

  constexpr bool determinism_specified =
    ::cuda::std::execution::__queryable_with<requirements_t, ::cuda::execution::determinism::__get_determinism_t>;
  constexpr bool tie_break_specified =
    ::cuda::std::execution::__queryable_with<requirements_t, ::cuda::execution::tie_break::__get_tie_break_t>;

  using requested_determinism_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::determinism::__get_determinism_t,
                                                ::cuda::execution::determinism::gpu_to_gpu_t>;
  using requested_tie_break_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::tie_break::__get_tie_break_t,
                                                ::cuda::execution::tie_break::prefer_smaller_index_t>;
  using requested_order_t =
    ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                ::cuda::execution::output_ordering::__get_output_ordering_t,
                                                ::cuda::execution::output_ordering::stable_sorted_t>;

  constexpr bool determinism_and_tie_break_paired = (determinism_specified == tie_break_specified);

  // Encodes rule 2 as the implication "a concrete tie-break requires gpu_to_gpu". The expression is the "or" form
  // and satisfied in the two cases that are allowed: the tie-break is unspecified or the determinism is already
  // gpu_to_gpu (which accepts any tie-break).
  constexpr bool tie_break_compatible_with_determinism =
    ::cuda::std::is_same_v<requested_tie_break_t, ::cuda::execution::tie_break::unspecified_t>
    || ::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>;
  constexpr bool is_unsorted_output =
    ::cuda::std::is_same_v<requested_order_t, ::cuda::execution::output_ordering::unsorted_t>;

  static_assert(determinism_and_tie_break_paired,
                "cub::DeviceBatchedTopK: determinism and tie_break requirements must be acknowledged together. Either "
                "omit both to accept the defaults (cuda::execution::determinism::gpu_to_gpu and "
                "cuda::execution::tie_break::prefer_smaller_index), or pass both explicitly inside "
                "cuda::execution::require(...).");
  static_assert(!determinism_and_tie_break_paired || tie_break_compatible_with_determinism,
                "cub::DeviceBatchedTopK: a tie_break of cuda::execution::tie_break::prefer_smaller_index or "
                "prefer_larger_index pins the result set across GPUs and therefore requires "
                "cuda::execution::determinism::gpu_to_gpu (it cannot be combined with run_to_run or not_guaranteed).");
  static_assert(
    !determinism_and_tie_break_paired || !tie_break_compatible_with_determinism || is_unsorted_output,
    "cub::DeviceBatchedTopK currently only implements cuda::execution::output_ordering::unsorted output "
    "(cuda::execution::output_ordering::sorted and stable_sorted are not yet implemented). Because the default "
    "output ordering is stable_sorted, an empty (no-requirement) environment is rejected: request unsorted output "
    "explicitly, e.g. cuda::execution::require(cuda::execution::determinism::not_guaranteed, "
    "cuda::execution::tie_break::unspecified, cuda::execution::output_ordering::unsorted).");

  // ---------------------------------------------------------------------------
  // Resolve the (optionally tuned) policy selector from the environment.
  // ---------------------------------------------------------------------------
  // A single tuning query on the whole `topk_policy`: a `tune`d selector returning `topk_policy` picks the backend
  // (baseline vs cluster) and both sub-policies in one shot. Absent by default (the sentinel `no_override`), in which
  // case the dispatch's automatic arch+size selector is used. Strip cv/ref so the override is a value type.
  using tuning_env_t =
    ::cuda::__call_result_or_t<::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>, EnvT>;
  using selector_override_t = ::cuda::std::remove_cvref_t<
    ::cuda::std::execution::__query_result_or_t<tuning_env_t, batched_topk::topk_policy, batched_topk::no_override>>;

  // ---------------------------------------------------------------------------
  // Argument-annotation constraints surfaced at the call site.
  // ---------------------------------------------------------------------------
  static_assert(::cuda::args::__traits<NumSegmentsParameterT>::is_single_value,
                "cub::DeviceBatchedTopK currently requires a single (uniform) number of segments resolved on the "
                "host; pass num_segments as a single-value annotation (e.g. cuda::args::constant or "
                "cuda::args::immediate), not a per-segment sequence.");
  static_assert(
    ::cuda::args::__is_wrapper_v<SegmentSizeParameterT> || ::cuda::std::is_integral_v<SegmentSizeParameterT>,
    "cub::DeviceBatchedTopK: segment_sizes must be a cuda::args annotation or a plain integral value "
    "(taken as a uniform immediate). A raw pointer or iterator is not interpreted as a sequence. Wrap "
    "per-segment sizes in cuda::args::deferred_sequence, or a single device-side value in "
    "cuda::args::deferred.");
  static_assert(::cuda::args::__is_wrapper_v<KParameterT> || ::cuda::std::is_integral_v<KParameterT>,
                "cub::DeviceBatchedTopK: k must be a cuda::args annotation or a plain integral value (taken as a "
                "uniform immediate). A raw pointer or iterator is not interpreted as a sequence. Wrap a per-segment k "
                "in cuda::args::deferred_sequence, or a single device-side value in cuda::args::deferred.");
  static_assert(
    ::cuda::args::__is_wrapper_v<NumSegmentsParameterT> || ::cuda::std::is_integral_v<NumSegmentsParameterT>,
    "cub::DeviceBatchedTopK: num_segments must be a cuda::args annotation or a plain integral value. A "
    "raw pointer or iterator is not accepted.");

  const auto stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, env);

  // The total-number-of-items guarantee is intentionally not part of the initial public API surface. The dispatch
  // only uses its element type to size internal large-segment offsets (the value itself is unused), so we pass a
  // conservative 64-bit upper bound here.
  constexpr auto total_num_items = ::cuda::args::immediate{::cuda::std::numeric_limits<::cuda::std::int64_t>::max()};

  return batched_topk::dispatch<requested_determinism_t::value,
                                requested_tie_break_t::value,
                                batched_topk::backend_mode::automatic,
                                selector_override_t>(
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
    stream.get());
}
//! @endcond
} // namespace detail

//! @rst
//! DeviceBatchedTopK provides device-wide, parallel operations for finding the largest (or smallest) K items from
//! many segments of unordered data items residing within device-accessible memory.
//!
//! .. versionadded:: 3.5.0
//!    First appears in CUDA Toolkit 13.5.
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
//! The parameters ``segment_sizes``, ``k``, and ``num_segments`` can be passed as **annotated arguments** from
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
//! - ``cuda::args::deferred{iterator}`` for a single value read in stream order through a pointer or iterator, for
//!   example one produced on the device by a preceding launch.
//! - ``cuda::args::deferred_sequence{iterator}`` for a distinct value per segment, also read in stream order.
//!
//! A plain integral value works too and is taken as a uniform ``immediate`` (no extra bounds). A pointer or iterator,
//! by contrast, must be wrapped explicitly in ``deferred`` (single value) or ``deferred_sequence`` (per segment).
//! Passing a raw pointer or iterator is rejected at compile time, because it would otherwise be misread as a single
//! value rather than a sequence.
//!
//! **How it is bounded.** A bound lets the algorithm reason about a value it does not know exactly:
//!
//! - A **compile-time** bound, ``cuda::args::bounds<lo, hi>()``, may accompany ``immediate``, ``deferred``, or
//!   ``deferred_sequence`` (a ``constant`` is already its own bound). The kernel specializes on this range and uses
//!   it to size temporary storage (see *Choosing argument bounds*), so prefer the tightest range you can prove.
//! - A **runtime** bound, ``cuda::args::bounds(lo, hi)``, may accompany ``deferred`` and ``deferred_sequence`` when
//!   the range is only known at runtime. When combined with a compile-time bound, the runtime bound must be at least
//!   as narrow, lying within the compile-time range and only tightening it further.
//!
//! **Which form each parameter accepts.** ``segment_sizes`` and ``k`` accept all four forms. ``num_segments`` must be
//! a single value (``constant``, ``immediate``, or a plain integral), never a per-segment sequence. ``segment_sizes``
//! must also carry a compile-time upper bound (a ``constant<N>`` or ``cuda::args::bounds<lo, hi>()``); the permitted
//! maximum is architecture-dependent (see *Current constraints* below), and tight bounds on every parameter are
//! encouraged.
//!
//! .. code-block:: c++
//!
//!     // segment_sizes (k is analogous):
//!     cuda::args::constant<256>{};                                           // fixed at compile time
//!     cuda::args::immediate{n, cuda::args::bounds<1, 1024>()};               // host value, at most 1024
//!     cuda::args::deferred_sequence{d_sizes, cuda::args::bounds<1, 1024>()}; // per-segment, each at most 1024
//!
//!     // a uniform segment size produced on the device, capped at compile time and narrowed at runtime:
//!     cuda::args::deferred{d_size, cuda::args::bounds<1, 1024>(), cuda::args::bounds(1, runtime_max)};
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
//! - **Segment size is architecture-dependent.** On pre-Hopper GPUs (compute capability < 9.0) every segment must be
//!   processable by a single thread block (one worker per segment): the *statically-known maximum* segment size (the
//!   upper bound of the ``segment_sizes`` annotation) must be small enough that such a block fits within the
//!   shared-memory limit. On Hopper and newer GPUs (compute capability >= 9.0) the thread-block-cluster backend also
//!   handles larger segments that exceed this per-block limit. Both uniform (fixed) and variable segment sizes are
//!   supported.
//! - **Uniform number of segments.** ``num_segments`` must be a single value, never a per-segment sequence.
//! - **Unsorted output required.** Only ``cuda::execution::output_ordering::unsorted`` is implemented; the sorted
//!   orderings of the default contract described in *Determinism, tie-breaking, and output ordering* below (and hence
//!   an empty, no-requirement environment, which defaults to ``stable_sorted``) are rejected at compile time. The
//!   supported ``determinism`` / ``tie_break`` requirements depend on the architecture -- see the *Current support*
//!   note below. ``determinism`` and ``tie_break`` must always be specified together, or both omitted to take the
//!   default.
//!
//! Determinism, tie-breaking, and output ordering
//! +++++++++++++++++++++++++++++++++++++++++++++++
//!
//! Like :cpp:struct:`cub::DeviceTopK`, the result of ``DeviceBatchedTopK`` is governed by two orthogonal execution
//! requirements: *which* items are selected per segment (``cuda::execution::determinism``, optionally refined by
//! ``cuda::execution::tie_break``) and the order in which they are written (``cuda::execution::output_ordering``).
//! When the caller does not opt out, the committed default is the most reproducible behavior: deterministic results
//! (``cuda::execution::determinism::gpu_to_gpu``), ties resolved toward the smaller (lower) source index
//! (``cuda::execution::tie_break::prefer_smaller_index``), and stable-sorted output
//! (``cuda::execution::output_ordering::stable_sorted``). Callers opt *out* of these guarantees to obtain faster
//! implementations. ``determinism`` and ``tie_break`` must always be specified together, or both omitted to take the
//! default. A specified ``tie_break`` of ``prefer_smaller_index`` or ``prefer_larger_index`` requires
//! ``determinism::gpu_to_gpu``.
//!
//! See :ref:`cub-topk-requirements` for the full requirement model, worked examples, and guidance on choosing
//! requirements.
//!
//! .. note::
//!
//!    **Current support.** Only the unsorted output ordering is implemented;
//!    ``cuda::execution::output_ordering::unsorted`` must be requested explicitly (``sorted`` / ``stable_sorted``, and
//!    thus an empty, no-requirement environment, are rejected at compile time). The supported selection requirements
//!    and segment sizes differ by architecture:
//!
//!    - **Pre-Hopper (compute capability < 9.0):** only the fully non-deterministic request
//!      ``(determinism::not_guaranteed, tie_break::unspecified)`` is supported, and every segment must fit a single
//!      thread block. Deterministic / tie-break requests and larger segments require the SM 9.0+ cluster backend and
//!      are diagnosed at compile time (or, in relaxed builds, at runtime as ``cudaErrorNotSupported``).
//!    - **Hopper and newer (compute capability >= 9.0):** all five acknowledged ``(determinism, tie_break)`` pairs are
//!      supported -- ``(not_guaranteed, unspecified)``, ``(run_to_run, unspecified)``, and ``(gpu_to_gpu,
//!      {unspecified, prefer_smaller_index, prefer_larger_index})`` -- and larger segments are handled by the cluster
//!      backend. Among non-deterministic requests the baseline vs cluster backend is chosen by an architecture /
//!      segment-size crossover.
//!
//!    When ``determinism::not_guaranteed`` is requested the per-segment output may be non-deterministic: if multiple
//!    items tie at the K-th position, the subset of tied elements returned is not uniquely defined and may vary between
//!    runs.
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
  //!    The behavior is undefined if an output range overlaps another output range or any input range.
  //!    Input ranges may overlap one another.
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
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
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
    const EnvT& env = {})
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
  //!    The behavior is undefined if an output range overlaps another output range or any input range.
  //!    Input ranges may overlap one another.
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
  //!
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Iterator such that `d_keys_in[i]` yields a random-access iterator to the keys of segment `i`
  //!
  //! @param[out] d_keys_out
  //!   Iterator such that `d_keys_out[i]` yields a random-access output iterator for the top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MaxKeys");
    return detail::dispatch_with_env(env, [&](auto /* tuning */, void* storage, size_t& bytes, auto /* stream */) {
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
  //!    The behavior is undefined if an output range overlaps another output range or any input range.
  //!    Input ranges may overlap one another.
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
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
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
  //!    The behavior is undefined if an output range overlaps another output range or any input range.
  //!    Input ranges may overlap one another.
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
  //!
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Iterator such that `d_keys_in[i]` yields a random-access iterator to the keys of segment `i`
  //!
  //! @param[out] d_keys_out
  //!   Iterator such that `d_keys_out[i]` yields a random-access output iterator for the top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MinKeys");
    return detail::dispatch_with_env(env, [&](auto /* tuning */, void* storage, size_t& bytes, auto /* stream */) {
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
  //!    The behavior is undefined if an output range overlaps another output range or any input range.
  //!    Input ranges may overlap one another.
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
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam ValueInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-input iterators @iterator
  //!
  //! @tparam ValueOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] d_values_in
  //!   Iterator such that `d_values_in[i]` yields a random-access iterator to the values of segment `i`
  //!
  //! @param[out] d_values_out
  //!   Iterator such that `d_values_out[i]` yields a random-access output iterator for the values corresponding to the
  //!   top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
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
  //!
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam ValueInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-input iterators @iterator
  //!
  //! @tparam ValueOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Iterator such that `d_keys_in[i]` yields a random-access iterator to the keys of segment `i`
  //!
  //! @param[out] d_keys_out
  //!   Iterator such that `d_keys_out[i]` yields a random-access output iterator for the top-k keys of segment `i`
  //!
  //! @param[in] d_values_in
  //!   Iterator such that `d_values_in[i]` yields a random-access iterator to the values of segment `i`
  //!
  //! @param[out] d_values_out
  //!   Iterator such that `d_values_out[i]` yields a random-access output iterator for the values corresponding to the
  //!   top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MaxPairs");
    return detail::dispatch_with_env(env, [&](auto /* tuning */, void* storage, size_t& bytes, auto /* stream */) {
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
  //!    The behavior is undefined if an output range overlaps another output range or any input range.
  //!    Input ranges may overlap one another.
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
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam ValueInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-input iterators @iterator
  //!
  //! @tparam ValueOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
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
  //! @param[in] d_values_in
  //!   Iterator such that `d_values_in[i]` yields a random-access iterator to the values of segment `i`
  //!
  //! @param[out] d_values_out
  //!   Iterator such that `d_values_out[i]` yields a random-access output iterator for the values corresponding to the
  //!   top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
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
  //!
  //! @tparam KeyInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-input iterators @iterator
  //!
  //! @tparam KeyOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment key-output iterators @iterator
  //!
  //! @tparam ValueInputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-input iterators @iterator
  //!
  //! @tparam ValueOutputIteratorItT
  //!   **[inferred]** Random-access input iterator over per-segment value-output iterators @iterator
  //!
  //! @tparam SegmentSizeParameterT
  //!   **[inferred]** Type of the ``segment_sizes`` argument
  //!
  //! @tparam KParameterT
  //!   **[inferred]** Type of the ``k`` argument
  //!
  //! @tparam NumSegmentsParameterT
  //!   **[inferred]** Type of the ``num_segments`` argument
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Iterator such that `d_keys_in[i]` yields a random-access iterator to the keys of segment `i`
  //!
  //! @param[out] d_keys_out
  //!   Iterator such that `d_keys_out[i]` yields a random-access output iterator for the top-k keys of segment `i`
  //!
  //! @param[in] d_values_in
  //!   Iterator such that `d_values_in[i]` yields a random-access iterator to the values of segment `i`
  //!
  //! @param[out] d_values_out
  //!   Iterator such that `d_values_out[i]` yields a random-access output iterator for the values corresponding to the
  //!   top-k keys of segment `i`
  //!
  //! @param[in] segment_sizes
  //!   Annotated argument providing the per-segment sizes (e.g. `cuda::args::constant<N>` for a uniform size,
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Must carry a compile-time upper bound; the
  //!   permitted maximum is architecture-dependent (see the *Current constraints* section).
  //!   Prefer a sharp (tight) upper bound, since a looser bound may increase temporary-storage usage (see the
  //!   *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] num_segments
  //!   The (uniform) number of segments, given as a `cuda::args` annotation or a plain integral value.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Must require `output_ordering::unsorted` (`sorted` / `stable_sorted`, and
  //!   thus an empty environment, are not yet supported). The selection requirements may be any acknowledged
  //!   `(determinism, tie_break)` pair: `(not_guaranteed, unspecified)`, `(run_to_run, unspecified)`, or `gpu_to_gpu`
  //!   with `unspecified` / `prefer_smaller_index` / `prefer_larger_index`. Deterministic requests require SM 9.0+.
  //!   @endrst
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
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceBatchedTopK::MinPairs");
    return detail::dispatch_with_env(env, [&](auto /* tuning */, void* storage, size_t& bytes, auto /* stream */) {
      return detail::dispatch_batched_topk<detail::topk::select::min>(
        storage, bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k, num_segments, env);
    });
  }
};

CUB_NAMESPACE_END
