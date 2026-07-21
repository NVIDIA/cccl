// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceBatchedTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
//! from many segments of unordered data items residing within device-accessible memory.

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
#include <cub/detail/segmented_params.cuh> // detail::params::__validate_uniform{,_or_per_segment}_integral_param
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
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#ifdef _CCCL_DOXYGEN_INVOKED // Only parse this during doxygen passes:

//! @def CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT
//!
//! Specific to cub::DeviceBatchedTopK (it has no effect on any other CUB algorithm).
//!
//! By default, cub::DeviceBatchedTopK fails at compile time (via `static_assert`) when the requested configuration
//! cannot be served on *every* compute capability the translation unit is being compiled for. Some requests (a
//! deterministic result, or a segment too large for the single-block backend) require the SM90+ cluster backend, so
//! they cannot compile when a pre-SM90 compute capability is among the targets.
//!
//! Define this macro (before including any CUB header) to suppress that compile-time check and defer the diagnosis to
//! runtime instead: on a device that cannot serve the request, dispatch returns `cudaErrorNotSupported`. This is
//! useful when a single translation unit must compile the full configuration space across a mix of compute
//! capabilities and decide what is runnable at runtime. CUB's own tests and benchmarks define it for this reason.
#  define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#endif // _CCCL_DOXYGEN_INVOKED

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
_CCCL_HOST_API static cudaError_t dispatch_batched_topk(
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
  //      (and therefore the empty-env default, which resolves to `stable_sorted`) remain rejected.
  //
  // Backend routing implied by the request (exact rules in `select_backend`, tuning_batched_topk.cuh): any request
  // beyond fully non-deterministic -- determinism stronger than not_guaranteed (run_to_run/gpu_to_gpu), or a concrete
  // tie-break -- is served only by the cluster backend (SM 9.0+). A fully non-deterministic request (not_guaranteed +
  // unspecified tie-break) leaves the backend open, chosen from the target architecture and the statically-known
  // maximum segment size: a segment too large for the single-block baseline forces the cluster backend, otherwise the
  // baseline runs unless the cluster is measured to win.
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
  // segment_sizes and k may be uniform or per-segment; num_segments must be a single value. Instantiating each
  // validation struct (via the `all_ok` reads below) runs its layered per-argument static_asserts -- one targeted
  // diagnostic per misuse. Three argument-specific constraints are then applied inline below (segment_sizes' maximum
  // size, k's element-type width, num_segments host-known).
  using segment_sizes_validation =
    detail::params::__validate_uniform_or_per_segment_integral_param<SegmentSizeParameterT>;
  using k_validation            = detail::params::__validate_uniform_or_per_segment_integral_param<KParameterT>;
  using num_segments_validation = detail::params::__validate_uniform_integral_param<NumSegmentsParameterT>;

  // Only the statically-known *maximum* segment size is constrained: it must not exceed 2^21 (about 2 million). Beyond
  // that the streaming cluster backend is not competitive; larger segments are future work (a WIP multi-CTA baseline
  // backend). So a segment-size type or bound whose maximum exceeds 2^21 (e.g. an un-annotated int32/uint32, or an
  // explicit bound above 2^21) must carry a tighter compile-time `cuda::args::bounds`. The minimum is left
  // unconstrained: a negative statically-known lower bound is accepted, and the kernel clamps any negative runtime size
  // up to 0 (see detail::params::__get_and_clamp_param_to_nonnegative). k carries no such maximum-value bound (on the
  // device an over-large k is clamped to the segment size and a negative k to 0), only the 64-bit element-type width
  // limit checked below.
  if constexpr (segment_sizes_validation::all_ok)
  {
    static_assert(
      ::cuda::std::cmp_less_equal(segment_sizes_validation::args_traits::highest, ::cuda::std::int64_t{1} << 21),
      "cub::DeviceBatchedTopK: the statically-known maximum segment size exceeds the maximum currently supported "
      "segment size (2^21, about 2 million). Give a segment-size type whose maximum fits, or a compile-time upper "
      "bound (cuda::args::bounds) not exceeding 2^21.");
  }

  // k's value is clamped to the segment size on the device through a 64-bit intermediate, so a wider element type
  // (e.g. __int128) could silently wrap. `sizeof` rather than a value comparison keeps character element types
  // compiling (the `cmp_*` comparators reject them).
  if constexpr (k_validation::all_ok)
  {
    static_assert(sizeof(typename k_validation::args_traits::element_type) <= sizeof(::cuda::std::uint64_t),
                  "cub::DeviceBatchedTopK: k's element type must be at most 64 bits wide.");
  }

  // num_segments must be known on the host: the launch configuration is computed from it before any kernel runs, so a
  // device-resident `deferred` value cannot be used (a per-segment sequence is already rejected by the validation
  // above).
  if constexpr (num_segments_validation::all_ok)
  {
    static_assert(!num_segments_validation::args_traits::is_deferred,
                  "cub::DeviceBatchedTopK: num_segments must be a host-known value (e.g. cuda::args::constant, "
                  "cuda::args::immediate, or a plain integral); a device-resident cuda::args::deferred value is not "
                  "supported because the launch configuration is computed on the host.");
  }

  // Only instantiate the dispatch once every argument passes its type/element checks above (and num_segments is
  // host-known).
  if constexpr (segment_sizes_validation::all_ok && k_validation::all_ok && num_segments_validation::all_ok
                && !num_segments_validation::args_traits::is_deferred)
  {
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
  else
  {
    return cudaErrorInvalidValue;
  }
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
//! value rather than a sequence. A plain integral (no bound) is still subject to each parameter's constraints: for
//! ``segment_sizes`` in particular, a plain signed integral such as ``int`` is rejected because its maximum exceeds
//! the supported maximum segment size of ``2^21`` (see *Which form each parameter accepts* and *Current constraints*
//! below).
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
//! a single, non-negative value known on the host (``constant``, ``immediate``, or a plain integral).
//! ``segment_sizes``
//! must have a statically-known *maximum* not exceeding the supported ``2^21`` (about 2 million; see *Current
//! constraints* below): a type whose maximum already fits (a narrow type such as ``uint8_t``, ``int16_t``, or
//! ``uint16_t``) is accepted without an explicit bound, while a type whose maximum exceeds ``2^21`` (e.g. ``int32_t``,
//! ``uint32_t``, or ``int64_t``) must carry a compile-time upper bound (a ``constant<N>`` or
//! ``cuda::args::bounds<lo, hi>()``). A negative statically-known lower bound is allowed: negative runtime sizes are
//! clamped to an empty segment (size 0). A non-negative lower bound is trusted -- passing an actual value outside its
//! declared bound (for instance a negative value under a non-negative bound) is a caller precondition violation
//! (undefined behavior). ``k`` has no algorithm-imposed maximum: a ``k`` larger than a segment's size selects that
//! whole segment. Its lower bound follows the same rule as ``segment_sizes`` above -- under a negative statically-known
//! lower bound a negative runtime ``k`` is clamped to 0 (selecting nothing), while under a non-negative lower bound a
//! negative value is a caller precondition violation (undefined behavior). Tight bounds on every parameter are
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
//! (a compile error is emitted if violated):
//!
//! - **Host-only.** Unlike most CUB algorithms, ``DeviceBatchedTopK`` does not support CUDA dynamic parallelism: its
//!   methods must be invoked from host code, not from device code.
//! - **Segment size is architecture-dependent.** On pre-Hopper GPUs (compute capability < 9.0) every segment must be
//!   processable by a single thread block (one worker per segment): the *statically-known maximum* segment size (the
//!   upper bound of the ``segment_sizes`` annotation) must be small enough that such a block fits within the
//!   shared-memory limit. On Hopper and newer GPUs (compute capability >= 9.0) the thread-block-cluster backend also
//!   handles larger segments that exceed this per-block limit. Both uniform (fixed) and variable segment sizes are
//!   supported. Independent of the architecture -- and independent of the integer type used for the ``segment_sizes``
//!   argument (a wider type such as ``int64_t`` does not raise it) -- an individual segment is currently limited to a
//!   maximum of ``2^21`` (about 2 million) items, enforced at compile time from the statically-known maximum segment
//!   size. Larger segments are future work. A type whose maximum already lies within it (a narrow type such as
//!   ``uint8_t``, ``int16_t``, or ``uint16_t``) is accepted un-annotated; a type whose maximum exceeds ``2^21`` (e.g.
//!   ``int32_t``, ``uint32_t``, or ``int64_t``) must carry a compile-time ``cuda::args::bounds`` whose upper end does
//!   not exceed ``2^21`` (see *Which form each parameter accepts* above for how the lower bound and out-of-bound values
//!   are handled).
//! - **k is at most 64 bits wide.** The element type of ``k`` may be no wider than 64 bits. The device clamps ``k`` to
//!   the segment size through a 64-bit intermediate, so a wider integer type (e.g. ``__int128``) is rejected to avoid a
//!   silent wrap. ``k`` itself has no algorithm-imposed maximum (see *Which form each parameter accepts* above).
//! - **Uniform number of segments.** ``num_segments`` must be a single value (``constant``, ``immediate``, or a plain
//!   integral) resolved on the host. A ``deferred`` (device-resident) count is not supported at this time.
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
//!      are diagnosed at compile time (or, when ``CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT`` is defined, deferred to
//!      runtime as ``cudaErrorNotSupported``).
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MaxKeys(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MaxKeys(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MinKeys(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MinKeys(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MaxPairs(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MaxPairs(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MinPairs(
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
  //!   or `cuda::args::deferred_sequence{...}` for variable sizes). Its statically-known maximum must lie within the
  //!   currently supported range (see the *Current constraints* section for the value and its architecture dependence):
  //!   narrow types (`int8_t`/`int16_t`/`uint16_t`) qualify unannotated, while wider types (`int32_t`/`uint32_t`/
  //!   `int64_t`) must carry a compile-time `cuda::args::bounds`. A negative lower bound is allowed; negative runtime
  //!   sizes clamp to an empty segment. Prefer a sharp (tight) upper bound, since a looser bound may increase
  //!   temporary-storage usage (see the *Choosing argument bounds* section).
  //!
  //! @param[in] k
  //!   The number of selected items per segment, given as a `cuda::args` annotation or a plain integral value. It has
  //!   no algorithm-imposed maximum; a `k` larger than a segment's size selects that whole segment. Like
  //!   `segment_sizes`, a negative lower bound is allowed and a negative runtime `k` is then clamped to 0 (selecting
  //!   nothing).
  //!
  //! @param[in] num_segments
  //!   The number of segments, given as a `cuda::args` annotation or a plain integral value. Must be
  //!   non-negative.
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
  [[nodiscard]] _CCCL_HOST_API static cudaError_t MinPairs(
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
