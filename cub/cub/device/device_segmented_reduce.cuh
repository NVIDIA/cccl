// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! cub::DeviceSegmentedReduce provides device-wide, parallel operations for computing a batched reduction across
//! multiple sequences of data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#ifndef CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <cub/device/device_segmented_reduce.cuh> is not supported when compiling with NVRTC. Include block-, warp-, or thread-level primitives instead (e.g. <cub/block/block_reduce.cuh>). You can define CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK to disable this warning."
#  endif // _CCCL_COMPILER(NVRTC)
#endif // CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/device_memory_resource.cuh>
#include <cub/detail/env_dispatch.cuh>
#include <cub/detail/it_traits.cuh>
#include <cub/detail/key_value_pair.cuh>
#include <cub/detail/non_void_value.cuh>
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/__functional/call_or.h>
#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceSegmentedReduce provides device-wide, parallel operations for
//! computing a reduction across multiple sequences of data items
//! residing within device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`_
//! (or *fold*) uses a binary combining operator to compute a single aggregate
//! from a sequence of input elements.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceSegmentedReduce}
//! @determinism{run_to_run}
//!
//! Determinism
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! ``cub::DeviceSegmentedReduce`` supports ``not_guaranteed`` and ``run_to_run`` (default ``run_to_run``).
//! ``gpu_to_gpu`` is not supported and is rejected at compile time. See the
//! :ref:`determinism guarantees <cccl-determinism>` for what each level means.
//!
//! Tuning
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! All algorithms in DeviceSegmentedReduce can be tuned by passing a custom :ref:`policy selector
//! <cub-policy-selectors>` that returns a :cpp:struct:`cub::SegmentedReducePolicy`, as shown in the example below:
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-reduce-sum-policy-selector
//!      :end-before: example-end segmented-reduce-sum-policy-selector
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin segmented-reduce-sum-tuning
//!      :end-before: example-end segmented-reduce-sum-tuning
//!
//! @endrst
struct DeviceSegmentedReduce
{
private:
  // Rejects determinism requirements of an environment that device segmented reductions do not support
  template <typename EnvT>
  static constexpr bool is_supported_determinism_v = !::cuda::std::is_same_v<
    ::cuda::std::execution::__query_result_or_t<
      ::cuda::std::execution::
        __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>,
      ::cuda::execution::determinism::__get_determinism_t,
      ::cuda::execution::determinism::run_to_run_t>,
    ::cuda::execution::determinism::gpu_to_gpu_t>;

  // TODO(bgruber): I would ideally like to have the logic of extracting the policy selector from the tuning environment
  // inside the dispatch function, but this will not work with CCCL.C, which needs to pass a stateful policy selector.
  // Refactor this once we have a host code JIT compiler.
  template <typename OverrideAccumT  = detail::use_default,
            typename OverrideOffsetT = detail::use_default,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename InitValueT,
            typename TuningEnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t select_tuning_and_dispatch_variable_size(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    const InitValueT& initial_value,
    cudaStream_t stream,
    TuningEnvT)
  {
    using offset_t =
      typename ::cuda::std::conditional_t<::cuda::std::is_same_v<OverrideOffsetT, detail::use_default>,
                                          detail::common_iterator_value<BeginOffsetIteratorT, EndOffsetIteratorT>,
                                          ::cuda::std::type_identity<OverrideOffsetT>>::type;
    static_assert(::cuda::std::is_integral_v<offset_t>, "Offset iterator value type should be integral.");
    if constexpr (::cuda::std::is_integral_v<offset_t>)
    {
      using accum_t =
        decltype(detail::segmented_reduce::select_segmented_accum_t<InputIteratorT, InitValueT, ReductionOpT>(
          static_cast<OverrideAccumT*>(nullptr)));
      using default_policy_selector_t =
        detail::segmented_reduce::policy_selector_from_types<accum_t, offset_t, ReductionOpT>;
      using policy_selector_t =
        ::cuda::std::execution::__query_result_or_t<TuningEnvT, SegmentedReducePolicy, default_policy_selector_t>;
      return detail::segmented_reduce::dispatch<OverrideAccumT, offset_t>(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        reduction_op,
        initial_value,
        0, // max_segment_size
        stream,
        policy_selector_t{});
    }
    _CCCL_UNREACHABLE();
  }

  template <typename OverrideAccumT = detail::use_default,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename OffsetT,
            typename ReductionOpT,
            typename InitValueT,
            typename TuningEnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t select_tuning_and_dispatch_fixed_size(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    OffsetT segment_size,
    ReductionOpT reduction_op,
    InitValueT initial_value,
    cudaStream_t stream,
    TuningEnvT)
  {
    using accum_t =
      decltype(detail::segmented_reduce::select_segmented_accum_t<InputIteratorT, InitValueT, ReductionOpT>(
        static_cast<OverrideAccumT*>(nullptr)));
    using default_policy_selector_t =
      detail::segmented_reduce::policy_selector_from_types<accum_t, OffsetT, ReductionOpT>;
    using policy_selector_t =
      ::cuda::std::execution::__query_result_or_t<TuningEnvT, SegmentedReducePolicy, default_policy_selector_t>;
    return detail::segmented_reduce::dispatch_fixed_size<OverrideAccumT>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      segment_size,
      reduction_op,
      initial_value,
      stream,
      policy_selector_t{});
  }

  template <typename ReductionOpT, typename InputIteratorT, typename OutputIteratorT, typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t fixed_size_arg_two_phase(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    const EnvT& env)
  {
    static_assert(is_supported_determinism_v<EnvT>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;

    using input_value_t  = cub::detail::it_value_t<InputIteratorT>;
    using output_tuple_t = cub::detail::non_void_value_t<OutputIteratorT, ::cuda::std::pair<offset_t, input_value_t>>;
    using accum_t        = output_tuple_t;
    using init_value_t   = detail::reduce::empty_problem_init_t<accum_t>;
    using output_key_t   = typename output_tuple_t::first_type;
    using output_value_t = typename output_tuple_t::second_type;

    static_assert(::cuda::std::is_same_v<int, output_key_t>, "Output key type must be int.");
    static_assert(::cuda::std::numeric_limits<input_value_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    auto d_indexed_in = THRUST_NS_QUALIFIER::make_transform_iterator(
      THRUST_NS_QUALIFIER::counting_iterator<::cuda::std::int64_t>{0},
      detail::segmented_reduce::generate_idx_value<InputIteratorT, output_value_t>(d_in, segment_size));

    constexpr bool is_min = ::cuda::std::is_same_v<ReductionOpT, cub::detail::arg_min>;
    auto sentinel =
      is_min ? ::cuda::std::numeric_limits<input_value_t>::max() : ::cuda::std::numeric_limits<input_value_t>::lowest();
    const init_value_t initial_value{accum_t(1, sentinel)};

    return detail::dispatch_with_env(
      d_temp_storage, temp_storage_bytes, env, [&](auto tuning_env, void* storage, size_t& bytes, cudaStream_t stream) {
        return select_tuning_and_dispatch_fixed_size<accum_t>(
          storage,
          bytes,
          d_indexed_in,
          d_out,
          num_segments,
          static_cast<offset_t>(segment_size),
          ReductionOpT(),
          initial_value,
          stream,
          tuning_env);
      });
  }

  template <typename ReductionOpT, typename InputIteratorT, typename OutputIteratorT, typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t fixed_size_arg_single_phase(
    InputIteratorT d_in, OutputIteratorT d_out, ::cuda::std::int64_t num_segments, int segment_size, const EnvT& env)
  {
    return detail::dispatch_with_env(
      env, [&](auto /* tuning */, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t /* stream */) {
        return fixed_size_arg_two_phase<ReductionOpT>(
          d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, segment_size, env);
      });
  }

  template <typename OverrideAccumT  = detail::use_default,
            typename OverrideOffsetT = detail::use_default,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename InitValueT,
            typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t variable_size_two_phase(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    const InitValueT& initial_value,
    const EnvT& env)
  {
    static_assert(is_supported_determinism_v<EnvT>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    return detail::dispatch_with_env(
      d_temp_storage, temp_storage_bytes, env, [&](auto tuning_env, void* storage, size_t& bytes, cudaStream_t stream) {
        return select_tuning_and_dispatch_variable_size<OverrideAccumT, OverrideOffsetT>(
          storage,
          bytes,
          d_in,
          d_out,
          num_segments,
          d_begin_offsets,
          d_end_offsets,
          reduction_op,
          initial_value,
          stream,
          tuning_env);
      });
  }

  template <typename OverrideAccumT  = detail::use_default,
            typename OverrideOffsetT = detail::use_default,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename InitValueT,
            typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t variable_size_single_phase(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitValueT initial_value,
    const EnvT& env)
  {
    return detail::dispatch_with_env(
      env, [&](auto /* tuning */, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t /* stream */) {
        return variable_size_two_phase<OverrideAccumT, OverrideOffsetT>(
          d_temp_storage,
          temp_storage_bytes,
          d_in,
          d_out,
          num_segments,
          d_begin_offsets,
          d_end_offsets,
          reduction_op,
          initial_value,
          env);
      });
  }

  template <typename OverrideAccumT = detail::use_default,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename InitValueT,
            typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t fixed_size_two_phase(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    ReductionOpT reduction_op,
    InitValueT initial_value,
    const EnvT& env)
  {
    static_assert(is_supported_determinism_v<EnvT>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;

    return detail::dispatch_with_env(
      d_temp_storage, temp_storage_bytes, env, [&](auto tuning_env, void* storage, size_t& bytes, cudaStream_t stream) {
        return select_tuning_and_dispatch_fixed_size<OverrideAccumT>(
          storage,
          bytes,
          d_in,
          d_out,
          num_segments,
          static_cast<offset_t>(segment_size),
          reduction_op,
          initial_value,
          stream,
          tuning_env);
      });
  }

  template <typename OverrideAccumT = detail::use_default,
            typename OffsetT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename InitValueT,
            typename EnvT>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t fixed_size_single_phase(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    OffsetT segment_size,
    ReductionOpT reduction_op,
    InitValueT initial_value,
    const EnvT& env)
  {
    return detail::dispatch_with_env(
      env, [&](auto /* tuning */, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t /* stream */) {
        return fixed_size_two_phase<OverrideAccumT>(
          d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, segment_size, reduction_op, initial_value, env);
      });
  }

public:
  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a custom min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-custommin
  //!     :end-before: example-end segmented-reduce-custommin
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-reduce
  //!     :end-before: example-end segmented-reduce-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename T,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    T initial_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Reduce");

    return variable_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      reduction_op,
      initial_value, // zero-initialize
      env);
  }

  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a custom min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-reduce-env
  //!     :end-before: example-end segmented-reduce-reduce-env
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-reduce-env-determinism
  //!     :end-before: example-end segmented-reduce-reduce-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename ReductionOpT,
            typename T,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    T initial_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Reduce");

    return variable_size_single_phase(
      d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, reduction_op, initial_value, env);
  }

  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor and a fixed segment size.
  //!
  //! .. versionadded:: 3.2.0
  //!    First appears in CUDA Toolkit 13.2.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a custom min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-custommin
  //!     :end-before: example-end segmented-reduce-custommin
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-reduce
  //!     :end-before: example-end fixed-size-segmented-reduce-reduce
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregates
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename T,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    ReductionOpT reduction_op,
    T initial_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Reduce");
    return fixed_size_two_phase(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, segment_size, reduction_op, initial_value, env);
  }

  //! @rst
  //! Computes a device-wide segmented reduction using the specified
  //! binary ``reduction_op`` functor and a fixed segment size.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-reduce-env
  //!     :end-before: example-end fixed-size-segmented-reduce-reduce-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregates
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] initial_value
  //!   Initial value of the reduction for each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename T,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    ReductionOpT reduction_op,
    T initial_value,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Reduce");

    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;

    return fixed_size_single_phase(
      d_in, d_out, num_segments, static_cast<offset_t>(segment_size), reduction_op, initial_value, env);
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-sum
  //!     :end-before: example-end segmented-reduce-sum
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>,
            typename      = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                                typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Sum");

    using OutputT      = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;
    using init_value_t = OutputT;

    return variable_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      ::cuda::std::plus<>{},
      init_value_t{}, // zero-initialize
      env);
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-sum-env
  //!     :end-before: example-end segmented-reduce-sum-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Sum");

    using OutputT      = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;
    using init_value_t = OutputT;
    using op_t         = ::cuda::std::plus<>;

    return variable_size_single_phase(
      d_in, d_out, num_segments, d_begin_offsets, d_end_offsets, op_t{}, init_value_t{}, env);
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator.
  //!
  //! .. versionadded:: 3.2.0
  //!    First appears in CUDA Toolkit 13.2.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-sum
  //!     :end-before: example-end fixed-size-segmented-reduce-sum
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      const EnvT& env = {})
  {
    static_assert(!::cuda::std::is_same_v<InputIteratorT, void*>,
                  "InputIteratorT must be a real iterator; void* has no iterator_traits::value_type.");
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Sum");
    using init_value_t = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;
    return fixed_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      segment_size,
      ::cuda::std::plus{},
      init_value_t{},
      env);
  }

  //! @rst
  //! Computes a device-wide segmented sum using the addition (``+``) operator
  //! and a fixed segment size.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Uses ``0`` as the initial value of the reduction for each segment.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-sum-env
  //!     :end-before: example-end fixed-size-segmented-reduce-sum-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Sum");

    using output_t = detail::non_void_value_t<OutputIteratorT, detail::it_value_t<InputIteratorT>>;

    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;
    using op_t     = ::cuda::std::plus<>;

    return fixed_size_single_phase(
      d_in, d_out, num_segments, static_cast<offset_t>(segment_size), op_t{}, output_t{}, env);
  }

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter is
  //!   specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-min
  //!     :end-before: example-end segmented-reduce-min
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>,
            typename      = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                                typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Min");

    using InputT       = detail::it_value_t<InputIteratorT>;
    using init_value_t = InputT;
    static_assert(::cuda::std::numeric_limits<init_value_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    return variable_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      ::cuda::minimum<>{},
      ::cuda::std::numeric_limits<init_value_t>::max(),
      env);
  }

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter is
  //!   specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-min-env
  //!     :end-before: example-end segmented-reduce-min-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Min(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Min");

    using InputT       = detail::it_value_t<InputIteratorT>;
    using init_value_t = InputT;
    using op_t         = ::cuda::minimum<>;

    static_assert(::cuda::std::numeric_limits<init_value_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    return variable_size_single_phase(
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      op_t{},
      ::cuda::std::numeric_limits<init_value_t>::max(),
      env);
  }

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator.
  //!
  //! .. versionadded:: 3.2.0
  //!    First appears in CUDA Toolkit 13.2.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-min
  //!     :end-before: example-end fixed-size-segmented-reduce-min
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Min");
    using input_t = detail::it_value_t<InputIteratorT>;
    static_assert(::cuda::std::numeric_limits<input_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");
    return fixed_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      segment_size,
      ::cuda::minimum<>{},
      ::cuda::std::numeric_limits<input_t>::max(),
      env);
  }

  //! @rst
  //! Computes a device-wide segmented minimum using the less-than (``<``) operator
  //! and a fixed segment size.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction for each segment.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-min-env
  //!     :end-before: example-end fixed-size-segmented-reduce-min-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Min(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Min");

    using input_t = detail::it_value_t<InputIteratorT>;

    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");

    static_assert(::cuda::std::numeric_limits<input_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;
    using op_t     = ::cuda::minimum<>;

    return fixed_size_single_phase(
      d_in,
      d_out,
      num_segments,
      static_cast<offset_t>(segment_size),
      op_t{},
      ::cuda::std::numeric_limits<input_t>::max(),
      env);
  }

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased for both
  //!   the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where the latter
  //!   is specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmin
  //!     :end-before: example-end segmented-reduce-argmin
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>,
            typename      = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                                typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMin");

    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OverrideOffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    using InputValueT    = detail::it_value_t<InputIteratorT>;
    using OutputTupleT   = detail::non_void_value_t<OutputIteratorT, KeyValuePair<OverrideOffsetT, InputValueT>>;
    using OutputKeyT     = typename OutputTupleT::Key;
    using OutputValueT   = typename OutputTupleT::Value;
    using OverrideAccumT = OutputTupleT;
    using init_value_t   = detail::reduce::empty_problem_init_t<OverrideAccumT>;

    static_assert(::cuda::std::is_same_v<int, OutputKeyT>, "Output key type must be int.");
    static_assert(::cuda::std::numeric_limits<InputValueT>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    // Wrapped input iterator to produce index-value <OverrideOffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OverrideOffsetT, OutputValueT>;
    const ArgIndexInputIteratorT d_indexed_in(d_in);

    const init_value_t initial_value{OverrideAccumT(1, ::cuda::std::numeric_limits<InputValueT>::max())};

    return variable_size_two_phase<OverrideAccumT, OverrideOffsetT>(
      d_temp_storage,
      temp_storage_bytes,
      d_indexed_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::ArgMin{},
      initial_value,
      env);
  }

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmin-env
  //!     :end-before: example-end segmented-reduce-argmin-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cub::KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::ArgMin");

    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OverrideOffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    using InputValueT    = detail::it_value_t<InputIteratorT>;
    using OutputTupleT   = detail::non_void_value_t<OutputIteratorT, KeyValuePair<OverrideOffsetT, InputValueT>>;
    using OutputKeyT     = typename OutputTupleT::Key;
    using OutputValueT   = typename OutputTupleT::Value;
    using OverrideAccumT = OutputTupleT;
    using init_value_t   = detail::reduce::empty_problem_init_t<OverrideAccumT>;

    static_assert(::cuda::std::is_same_v<int, OutputKeyT>, "Output key type must be int.");
    static_assert(::cuda::std::numeric_limits<InputValueT>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    // Wrapped input iterator to produce index-value <OverrideOffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OverrideOffsetT, OutputValueT>;
    const ArgIndexInputIteratorT d_indexed_in(d_in);

    const init_value_t initial_value{OverrideAccumT(1, ::cuda::std::numeric_limits<InputValueT>::max())};

    return variable_size_single_phase<OverrideAccumT, OverrideOffsetT>(
      d_indexed_in, d_out, num_segments, d_begin_offsets, d_end_offsets, cub::ArgMin{}, initial_value, env);
  }

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item.
  //!
  //! .. versionadded:: 3.2.0
  //!    First appears in CUDA Toolkit 13.2.
  //!
  //! - The output value type of ``d_out`` is ``::cuda::std::pair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].second`` and its offset in that segment is written to ``d_out[i].first``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-argmin
  //!     :end-before: example-end fixed-size-segmented-reduce-argmin
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cuda::std::pair<int, T>`) @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMin");
    return fixed_size_arg_two_phase<cub::detail::arg_min>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, segment_size, env);
  }

  //! @rst
  //! Finds the first device-wide minimum in each segment using the
  //! less-than (``<``) operator, also returning the in-segment index of that item,
  //! with a fixed segment size.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - The output value type of ``d_out`` is ``::cuda::std::pair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].second`` and its offset in that segment is written to ``d_out[i].first``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-argmin-env
  //!     :end-before: example-end fixed-size-segmented-reduce-argmin-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cuda::std::pair<int, T>`) @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ArgMin(InputIteratorT d_in,
         OutputIteratorT d_out,
         ::cuda::std::int64_t num_segments,
         int segment_size,
         const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::ArgMin");
    return fixed_size_arg_single_phase<cub::detail::arg_min>(d_in, d_out, num_segments, segment_size, env);
  }

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-max
  //!     :end-before: example-end segmented-reduce-max
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>,
            typename      = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                                typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Max");

    using InputT       = cub::detail::it_value_t<InputIteratorT>;
    using init_value_t = InputT;

    static_assert(::cuda::std::numeric_limits<init_value_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    return variable_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      ::cuda::maximum<>{},
      ::cuda::std::numeric_limits<init_value_t>::lowest(),
      env);
  }

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-max-env
  //!     :end-before: example-end segmented-reduce-max-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Max(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      BeginOffsetIteratorT d_begin_offsets,
      EndOffsetIteratorT d_end_offsets,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Max");

    using InputT       = cub::detail::it_value_t<InputIteratorT>;
    using init_value_t = InputT;
    using op_t         = ::cuda::maximum<>;

    static_assert(::cuda::std::numeric_limits<init_value_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    return variable_size_single_phase(
      d_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      op_t{},
      ::cuda::std::numeric_limits<init_value_t>::lowest(),
      env);
  }

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator.
  //!
  //! .. versionadded:: 3.2.0
  //!    First appears in CUDA Toolkit 13.2.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-max
  //!     :end-before: example-end fixed-size-segmented-reduce-max
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::Max");
    using input_t = detail::it_value_t<InputIteratorT>;
    static_assert(::cuda::std::numeric_limits<input_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");
    return fixed_size_two_phase(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      num_segments,
      segment_size,
      ::cuda::maximum<>{},
      ::cuda::std::numeric_limits<input_t>::lowest(),
      env);
  }

  //! @rst
  //! Computes a device-wide segmented maximum using the greater-than (``>``) operator
  //! and a fixed segment size.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Uses ``::cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-max-env
  //!     :end-before: example-end fixed-size-segmented-reduce-max-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Max(InputIteratorT d_in,
      OutputIteratorT d_out,
      ::cuda::std::int64_t num_segments,
      int segment_size,
      const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::Max");

    using input_t = detail::it_value_t<InputIteratorT>;

    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported for device segmented reductions ");
    static_assert(::cuda::std::numeric_limits<input_t>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    // `offset_t` a.k.a `SegmentSizeT` is fixed to `int` type now, but later can be changed to accept
    // integral constant or larger integral types
    using offset_t = int;
    using op_t     = ::cuda::maximum<>;
    return fixed_size_single_phase(
      d_in,
      d_out,
      num_segments,
      static_cast<offset_t>(segment_size),
      op_t{},
      ::cuda::std::numeric_limits<input_t>::lowest(),
      env);
  }

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmax
  //!     :end-before: example-end segmented-reduce-argmax
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment
  //!   ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length `num_segments`, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>,
            typename      = ::cuda::std::void_t<typename ::cuda::std::iterator_traits<BeginOffsetIteratorT>::value_type,
                                                typename ::cuda::std::iterator_traits<EndOffsetIteratorT>::value_type>>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMax");

    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OverrideOffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    using InputValueT    = cub::detail::it_value_t<InputIteratorT>;
    using OutputTupleT   = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OverrideOffsetT, InputValueT>>;
    using OverrideAccumT = OutputTupleT;
    using init_value_t   = detail::reduce::empty_problem_init_t<OverrideAccumT>;
    using OutputKeyT     = typename OutputTupleT::Key;
    using OutputValueT   = typename OutputTupleT::Value;

    static_assert(::cuda::std::is_same_v<int, OutputKeyT>, "Output key type must be int.");
    static_assert(::cuda::std::numeric_limits<InputValueT>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    // Wrapped input iterator to produce index-value <OverrideOffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OverrideOffsetT, OutputValueT>;
    const ArgIndexInputIteratorT d_indexed_in(d_in);

    const init_value_t initial_value{OverrideAccumT(1, ::cuda::std::numeric_limits<InputValueT>::lowest())};

    return variable_size_two_phase<OverrideAccumT, OverrideOffsetT>(
      d_temp_storage,
      temp_storage_bytes,
      d_indexed_in,
      d_out,
      num_segments,
      d_begin_offsets,
      d_end_offsets,
      cub::ArgMax{},
      initial_value,
      env);
  }

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].value`` and its offset in that segment is written to ``d_out[i].key``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! - When input a contiguous sequence of segments, a single sequence
  //!   ``segment_offsets`` (of length ``num_segments + 1``) can be aliased
  //!   for both the ``d_begin_offsets`` and ``d_end_offsets`` parameters (where
  //!   the latter is specified as ``segment_offsets + 1``).
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Let ``s`` be in ``[0, num_segments)``. The range
  //!   ``[d_out + d_begin_offsets[s], d_out + d_end_offsets[s])`` shall not
  //!   overlap ``[d_in + d_begin_offsets[s], d_in + d_end_offsets[s])``,
  //!   ``[d_begin_offsets, d_begin_offsets + num_segments)`` nor
  //!   ``[d_end_offsets, d_end_offsets + num_segments)``.
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-reduce-argmax-env
  //!     :end-before: example-end segmented-reduce-argmax-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cub::KeyValuePair<int, T>`) @iterator
  //!
  //! @tparam BeginOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment beginning offsets @iterator
  //!
  //! @tparam EndOffsetIteratorT
  //!   **[inferred]** Random-access input iterator type for reading segment ending offsets @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] d_begin_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of beginning offsets of
  //!   length ``num_segments``, such that ``d_begin_offsets[i]`` is the first
  //!   element of the *i*\ :sup:`th` data segment in ``d_in``
  //!   @endrst
  //!
  //! @param[in] d_end_offsets
  //!   @rst
  //!   Random-access input iterator to the sequence of ending offsets of length
  //!   ``num_segments``, such that ``d_end_offsets[i] - 1`` is the last element of
  //!   the *i*\ :sup:`th` data segment in ``d_in``.
  //!   If ``d_end_offsets[i] - 1 <= d_begin_offsets[i]``, the *i*\ :sup:`th` is considered empty.
  //!   @endrst
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::ArgMax");

    // Using common iterator value type is a breaking change, see:
    // https://github.com/NVIDIA/cccl/pull/414#discussion_r1330632615
    using OverrideOffsetT = int; // detail::common_iterator_value_t<BeginOffsetIteratorT, EndOffsetIteratorT>;

    using InputValueT    = cub::detail::it_value_t<InputIteratorT>;
    using OutputTupleT   = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OverrideOffsetT, InputValueT>>;
    using OverrideAccumT = OutputTupleT;
    using init_value_t   = detail::reduce::empty_problem_init_t<OverrideAccumT>;
    using OutputKeyT     = typename OutputTupleT::Key;
    using OutputValueT   = typename OutputTupleT::Value;

    static_assert(::cuda::std::is_same_v<int, OutputKeyT>, "Output key type must be int.");
    static_assert(::cuda::std::numeric_limits<InputValueT>::is_specialized,
                  "numeric_limits must be specialized for the input value type");

    // Wrapped input iterator to produce index-value <OverrideOffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OverrideOffsetT, OutputValueT>;
    const ArgIndexInputIteratorT d_indexed_in(d_in);

    const init_value_t initial_value{OverrideAccumT(1, ::cuda::std::numeric_limits<InputValueT>::lowest())};

    return variable_size_single_phase<OverrideAccumT, OverrideOffsetT>(
      d_indexed_in, d_out, num_segments, d_begin_offsets, d_end_offsets, cub::ArgMax{}, initial_value, env);
  }

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item
  //!
  //! .. versionadded:: 3.2.0
  //!    First appears in CUDA Toolkit 13.2.
  //!
  //! - The output value type of ``d_out`` is ``::cuda::std::pair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].second`` and its offset in that segment is written to ``d_out[i].first``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-argmax
  //!     :end-before: example-end fixed-size-segmented-reduce-argmax
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cuda::std::pair<int, T>`) @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    int segment_size,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSegmentedReduce::ArgMax");
    return fixed_size_arg_two_phase<detail::arg_max>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, segment_size, env);
  }

  //! @rst
  //! Finds the first device-wide maximum in each segment using the
  //! greater-than (``>``) operator, also returning the in-segment index of that item,
  //! with a fixed segment size.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - The output value type of ``d_out`` is ``::cuda::std::pair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum of the *i*\ :sup:`th` segment is written to
  //!     ``d_out[i].second`` and its offset in that segment is written to ``d_out[i].first``.
  //!   - The ``{1, ::cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_segmented_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin fixed-size-segmented-reduce-argmax-env
  //!     :end-before: example-end fixed-size-segmented-reduce-argmax-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cuda::std::pair<int, T>`) @iterator
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_segments
  //!   The number of segments that comprise the segmented reduction data
  //!
  //! @param[in] segment_size
  //!   The fixed segment size of each segment
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ArgMax(InputIteratorT d_in,
         OutputIteratorT d_out,
         ::cuda::std::int64_t num_segments,
         int segment_size,
         const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSegmentedReduce::ArgMax");
    return fixed_size_arg_single_phase<cub::detail::arg_max>(d_in, d_out, num_segments, segment_size, env);
  }
};

CUB_NAMESPACE_END
