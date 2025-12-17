// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! cub::DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data
//! items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

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
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/device/dispatch/dispatch_reduce_deterministic.cuh>
#include <cub/device/dispatch/dispatch_reduce_nondeterministic.cuh>
#include <cub/device/dispatch/dispatch_streaming_reduce.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_type.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/__functional/maximum.h>
#include <cuda/__functional/minimum.h>
#include <cuda/__iterator/tabulate_output_iterator.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename DeterminismT>
inline constexpr bool is_non_deterministic_v =
  ::cuda::std::is_same_v<DeterminismT, ::cuda::execution::determinism::not_guaranteed_t>;

namespace reduce
{
struct get_tuning_query_t
{};

template <class Derived>
struct tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_tuning_query_t&) const noexcept -> Derived
  {
    return static_cast<const Derived&>(*this);
  }
};

struct default_rfa_tuning : tuning<default_rfa_tuning>
{
  template <class AccumT, class Offset, class OpT>
  using fn = detail::rfa::policy_hub<AccumT, Offset, OpT>;
};

template <typename ExtremumOutIteratorT, typename IndexOutIteratorT>
struct unzip_and_write_arg_extremum_op
{
  ExtremumOutIteratorT result_out_it;
  IndexOutIteratorT index_out_it;

  template <typename IndexT, typename KeyValuePairT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(IndexT, KeyValuePairT reduced_result)
  {
    *result_out_it = reduced_result.value;
    *index_out_it  = reduced_result.key;
  }
};
} // namespace reduce
} // namespace detail

//! @rst
//! DeviceReduce provides device-wide, parallel operations for computing
//! a reduction across a sequence of data items residing within
//! device-accessible memory.
//!
//! .. image:: ../../img/reduce_logo.png
//!     :align: center
//!
//! Overview
//! ====================================
//!
//! A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`_
//! (or *fold*) uses a binary combining operator to compute a single aggregate
//! from a sequence of input elements.
//!
//! Usage Considerations
//! ====================================
//!
//! @cdp_class{DeviceReduce}
//!
//! Performance
//! ====================================
//!
//! @linear_performance{reduction, reduce-by-key, and run-length encode}
//!
//! @endrst
struct DeviceReduce
{
private:
  template <typename TuningEnvT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename TransformOpT,
            typename T,
            typename NumItemsT,
            ::cuda::execution::determinism::__determinism_t Determinism>
  CUB_RUNTIME_FUNCTION static cudaError_t reduce_impl(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    TransformOpT transform_op,
    T init,
    ::cuda::execution::determinism::__determinism_holder_t<Determinism>,
    cudaStream_t stream)
  {
    using offset_t = detail::choose_offset_t<NumItemsT>;
    using accum_t  = ::cuda::std::
      __accumulator_t<ReductionOpT, ::cuda::std::invoke_result_t<TransformOpT, detail::it_value_t<InputIteratorT>>, T>;
    using reduce_tuning_t = ::cuda::std::execution::__query_result_or_t<
      TuningEnvT,
      detail::reduce::get_tuning_query_t,
      detail::reduce::arch_policies_from_types<accum_t, offset_t, ReductionOpT>>;

    return detail::reduce::dispatch<accum_t>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<offset_t>(num_items),
      reduction_op,
      init,
      stream,
      transform_op,
      reduce_tuning_t{});
  }

  template <typename TuningEnvT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename TransformOpT,
            typename T,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t reduce_impl(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT,
    TransformOpT transform_op,
    T init,
    ::cuda::execution::determinism::gpu_to_gpu_t,
    cudaStream_t stream)
  {
    using offset_t = detail::choose_offset_t<NumItemsT>;

    using reduce_tuning_t = ::cuda::std::execution::
      __query_result_or_t<TuningEnvT, detail::reduce::get_tuning_query_t, detail::reduce::default_rfa_tuning>;

    using accum_t = ::cuda::std::
      __accumulator_t<ReductionOpT, ::cuda::std::invoke_result_t<TransformOpT, detail::it_value_t<InputIteratorT>>, T>;
    using policy_t = typename reduce_tuning_t::template fn<accum_t, offset_t, ReductionOpT>;
    using dispatch_t =
      detail::rfa::dispatch_t<InputIteratorT, OutputIteratorT, offset_t, T, TransformOpT, accum_t, policy_t>;

    return dispatch_t::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<offset_t>(num_items), init, stream, transform_op);
  }

  template <typename TuningEnvT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename TransformOpT,
            typename T,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t reduce_impl(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    TransformOpT transform_op,
    T init,
    ::cuda::execution::determinism::not_guaranteed_t,
    cudaStream_t stream)
  {
    using offset_t = detail::choose_offset_t<NumItemsT>;
    using accum_t  = ::cuda::std::__accumulator_t<ReductionOpT, detail::it_value_t<InputIteratorT>, T>;

    using reduce_tuning_t = ::cuda::std::execution::__query_result_or_t<
      TuningEnvT,
      detail::reduce::get_tuning_query_t,
      detail::reduce::arch_policies_from_types<accum_t, offset_t, ReductionOpT>>;

    return detail::reduce::dispatch_nondeterministic<accum_t>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      THRUST_NS_QUALIFIER::unwrap_contiguous_iterator(d_out),
      static_cast<offset_t>(num_items),
      reduction_op,
      init,
      stream,
      transform_op,
      reduce_tuning_t{});
  }

public:
  //! @rst
  //! Computes a device-wide reduction using the specified binary ``reduction_op`` functor and initial value ``init``.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined min-reduction of a
  //! device vector of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;  // e.g., 7
  //!    int          *d_in;      // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;     // e.g., [-]
  //!    CustomMin    min_op;
  //!    int          init;       // e.g., INT_MAX
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Reduce(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items, min_op, init);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run reduction
  //!    cub::DeviceReduce::Reduce(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items, min_op, init);
  //!
  //!    // d_out <-- [0]
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
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] init
  //!   Initial value of the reduction
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    T init,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Reduce");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::reduce::dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<OffsetT>(num_items), reduction_op, init, stream);
  }

  //! @rst
  //! Computes a device-wide reduction using the specified binary ``reduction_op`` functor and initial value ``init``.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - By default, provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //!   To request "gpu-to-gpu" determinism, pass ``cuda::execution::require(cuda::execution::determinism::gpu_to_gpu)``
  //!   as the `env` parameter.
  //!   To request "not-guaranteed" determinism, pass
  //!    ``cuda::execution::require(cuda::execution::determinism::not_guaranteed)`` as the `env` parameter.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined min-reduction of a
  //! device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin reduce-env-determinism
  //!     :end-before: example-end reduce-env-determinism
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
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
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
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] init
  //!   Initial value of the reduction
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename T,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items, ReductionOpT reduction_op, T init, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceReduce::Reduce");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using default_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    using accum_t = ::cuda::std::__accumulator_t<ReductionOpT, detail::it_value_t<InputIteratorT>, T>;

    constexpr auto gpu_gpu_determinism =
      ::cuda::std::is_same_v<default_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>;

    // integral types are always gpu-to-gpu deterministic if reduction operator is a simple cuda binary
    // operator, so fallback to run-to-run determinism
    constexpr auto integral_fallback =
      gpu_gpu_determinism && ::cuda::std::is_integral_v<accum_t> && (detail::is_cuda_binary_operator<ReductionOpT>);

    // use gpu-to-gpu determinism only for float and double types with ::cuda::std::plus operator
    constexpr auto float_double_plus =
      gpu_gpu_determinism && detail::is_one_of_v<accum_t, float, double> && detail::is_cuda_std_plus_v<ReductionOpT>;

    constexpr auto float_double_min_max_fallback =
      gpu_gpu_determinism
      && detail::is_one_of_v<accum_t, float, double> && detail::is_cuda_minimum_maximum_v<ReductionOpT>;

    constexpr auto supported =
      integral_fallback || float_double_plus || float_double_min_max_fallback || !gpu_gpu_determinism;

    // gpu_to_gpu determinism is only supported for integral types with cuda operators, or
    // float and double types with ::cuda::std::plus operator
    static_assert(supported, "gpu_to_gpu determinism is unsupported");

    if constexpr (!supported)
    {
      return cudaErrorNotSupported;
    }
    else
    {
      constexpr auto no_determinism = detail::is_non_deterministic_v<default_determinism_t>;

      // Certain conditions must be met to be able to use the non-deterministic
      // kernel. The output iterator must be a contiguous iterator and the
      // reduction operator must be plus (for now). Additionally, since atomics for types of
      // size < 4B are emulated, they perform poorly, so we fall back to the run-to-run
      // determinism.
      constexpr auto is_contiguous_fallback =
        !no_determinism || THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT>;
      constexpr auto is_plus_fallback = !no_determinism || detail::is_cuda_std_plus_v<ReductionOpT>;
      constexpr auto is_4b_or_greater = !no_determinism || sizeof(accum_t) >= 4;

      // If the conditions for gpu-to-gpu determinism or non-deterministic
      // reduction are not met, we fall back to run-to-run determinism.
      using determinism_t = ::cuda::std::conditional_t<
        (gpu_gpu_determinism && (integral_fallback || float_double_min_max_fallback))
          || (no_determinism && !(is_contiguous_fallback && is_plus_fallback && is_4b_or_greater)),
        ::cuda::execution::determinism::run_to_run_t,
        default_determinism_t>;

      // Dispatch with environment - handles all boilerplate
      return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
        using tuning_t = decltype(tuning);
        return reduce_impl<tuning_t>(
          storage, bytes, d_in, d_out, num_items, reduction_op, ::cuda::std::identity{}, init, determinism_t{}, stream);
      });
    }
  }

  //! @rst
  //! Computes a device-wide sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction.
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //!   To request "gpu-to-gpu" determinism, pass ``cuda::execution::require(cuda::execution::determinism::gpu_to_gpu)``
  //!   as the `env` parameter.
  //!   To request "not-guaranteed" determinism, pass
  //!    ``cuda::execution::require(cuda::execution::determinism::not_guaranteed)`` as the `env` parameter.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined min-reduction of a
  //! device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin sum-env-determinism
  //!     :end-before: example-end sum-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is `cuda::std::execution::env<>`.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is `cuda::std::execution::env{}`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceReduce::Sum");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using default_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    constexpr auto no_determinism = detail::is_non_deterministic_v<default_determinism_t>;

    // The output iterator must be a contiguous iterator or we fall back to run-to-run determinism.
    constexpr auto is_contiguous_fallback =
      !no_determinism || THRUST_NS_QUALIFIER::is_contiguous_iterator_v<OutputIteratorT>;

    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>;

    // Since atomics for types of size < 4B are emulated, they perform poorly, so we fall back to run-to-run
    // determinism.
    constexpr auto is_4b_or_greater = !no_determinism || sizeof(OutputT) >= 4;

    using determinism_t =
      ::cuda::std::conditional_t<no_determinism && !(is_contiguous_fallback && is_4b_or_greater),
                                 ::cuda::execution::determinism::run_to_run_t,
                                 default_determinism_t>;

    using InitT = OutputT;

    // Dispatch with environment - handles all boilerplate
    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return reduce_impl<tuning_t>(
        storage,
        bytes,
        d_in,
        d_out,
        num_items,
        ::cuda::std::plus<>{},
        ::cuda::std::identity{},
        InitT{},
        determinism_t{},
        stream);
    });
  }

  //! @rst
  //! Computes a device-wide sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction.
  //! - Does not support ``+`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum-reduction of a device vector
  //! of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [-]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Sum(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sum-reduction
  //!    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // d_out <-- [38]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      NumItemsT num_items,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Sum");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    // The output value type
    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>;

    using InitT = OutputT;

    return detail::reduce::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      ::cuda::std::plus<>{},
      InitT{}, // zero-initialize
      stream);
  }

  //! @rst
  //! Computes a device-wide minimum using the less-than (``<``) operator.
  //!
  //! - Uses ``cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction.
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [-]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Min(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run min-reduction
  //!    cub::DeviceReduce::Min(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // d_out <-- [0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      NumItemsT num_items,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Min");

    using OffsetT  = detail::choose_offset_t<NumItemsT>; // Signed integer type for global offsets
    using InputT   = detail::it_value_t<InputIteratorT>;
    using InitT    = InputT;
    using limits_t = ::cuda::std::numeric_limits<InitT>;
#ifndef CCCL_SUPPRESS_NUMERIC_LIMITS_CHECK_IN_CUB_DEVICE_REDUCE_MIN_MAX
    static_assert(limits_t::is_specialized,
                  "cub::DeviceReduce::Min uses cuda::std::numeric_limits<InputIteratorT::value_type>::max() as initial "
                  "value, but cuda::std::numeric_limits is not specialized for the iterator's value type. This is "
                  "probably a bug and you should specialize cuda::std::numeric_limits. Define "
                  "CCCL_SUPPRESS_NUMERIC_LIMITS_CHECK_IN_CUB_DEVICE_REDUCE_MIN_MAX to suppress this check.");
#endif // CCCL_SUPPRESS_NUMERIC_LIMITS_CHECK_IN_CUB_DEVICE_REDUCE_MIN_MAX

    return detail::reduce::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      ::cuda::minimum<>{},
      limits_t::max(),
      stream);
  }

  //! @rst
  //! Computes a device-wide minimum using the less-than (``<``) operator. The result is written to the output
  //! iterator.
  //!
  //! - Uses ``cuda::std::numeric_limits<T>::max()`` as the initial value of the reduction.
  //! - Provides determinism based on the environment's determinism requirements.
  //!   To request "run-to-run" determinism, pass ``cuda::execution::require(cuda::execution::determinism::run_to_run)``
  //!   as the `env` parameter.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin min-env-determinism
  //!     :end-before: example-end min-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is `cuda::std::execution::env<>`.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is `cuda::std::execution::env{}`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Min(InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceReduce::Min");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    // Static assert to reject gpu_to_gpu determinism since it's not properly implemented
    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported");

    // TODO(NaderAlAwar): Relax this once non-deterministic implementation for min / max is available
    using determinism_t = ::cuda::execution::determinism::run_to_run_t;

    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>;

    using InitT    = OutputT;
    using limits_t = ::cuda::std::numeric_limits<InitT>;
    // Dispatch with environment - handles all boilerplate
    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return reduce_impl<tuning_t>(
        storage,
        bytes,
        d_in,
        d_out,
        num_items,
        ::cuda::minimum<>{},
        ::cuda::std::identity{},
        limits_t::max(),
        determinism_t{},
        stream);
    });
  }

  //! @rst
  //! Finds the first device-wide minimum using the less-than (``<``) operator and also returns the index of that item.
  //!
  //! - The minimum is written to ``d_min_out``
  //! - The offset of the returned item is written to ``d_index_out``, the offset type being written is of type
  //!   ``cuda::std::int64_t``.
  //! - For zero-length inputs, ``cuda::std::numeric_limits<T>::max()}`` is written to ``d_min_out``  and the index
  //!   ``1`` is written to ``d_index_out``.
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_min_out`` nor ``d_index_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector
  //! of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_reduce.cuh>
  //!    #include <cuda/std/cstdint>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int                num_items;    // e.g., 7
  //!    int                *d_in;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int                *d_min_out;   // memory for the minimum value
  //!    cuda::std::int64_t *d_index_out; // memory for the index of the returned value
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_min_out, d_index_out,
  //!    num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run argmin-reduction
  //!    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_min_out, d_index_out,
  //!    num_items);
  //!
  //!    // d_min_out   <-- 0
  //!    // d_index_out <-- 5
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam ExtremumOutIteratorT
  //!   **[inferred]** Output iterator type for recording minimum value
  //!
  //! @tparam IndexOutIteratorT
  //!   **[inferred]** Output iterator type for recording index of the returned value
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Iterator to the input sequence of data items
  //!
  //! @param[out] d_min_out
  //!   Iterator to which the minimum value is written
  //!
  //! @param[out] d_index_out
  //!   Iterator to which the index of the returned value is written
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename ExtremumOutIteratorT, typename IndexOutIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    ExtremumOutIteratorT d_min_out,
    IndexOutIteratorT d_index_out,
    ::cuda::std::int64_t num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMin");

    // The input type
    using InputValueT = cub::detail::it_value_t<InputIteratorT>;

    // Offset type used within the kernel and to index within one partition
    using PerPartitionOffsetT = int;

    // Offset type used to index within the total input in the range [d_in, d_in + num_items)
    using GlobalOffsetT = ::cuda::std::int64_t;

    // The value type used for the extremum
    using OutputExtremumT = detail::non_void_value_t<ExtremumOutIteratorT, InputValueT>;
    using InitT           = OutputExtremumT;

    // Reduction operation
    using ReduceOpT = cub::ArgMin;

    // Initial value
    OutputExtremumT initial_value{::cuda::std::numeric_limits<InputValueT>::max()};

    // Tabulate output iterator that unzips the result and writes it to the user-provided output iterators
    auto out_it = ::cuda::make_tabulate_output_iterator(
      detail::reduce::unzip_and_write_arg_extremum_op<ExtremumOutIteratorT, IndexOutIteratorT>{d_min_out, d_index_out});

    return detail::reduce::dispatch_streaming_arg_reduce_t<
      InputIteratorT,
      decltype(out_it),
      PerPartitionOffsetT,
      GlobalOffsetT,
      ReduceOpT,
      InitT>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       out_it,
                       static_cast<GlobalOffsetT>(num_items),
                       ReduceOpT{},
                       initial_value,
                       stream);
  }

  //! @rst
  //! Finds the first device-wide minimum using the less-than (``<``) operator and also returns the index of that item.
  //!
  //! - The minimum is written to ``d_min_out``
  //! - The offset of the returned item is written to ``d_index_out``, the offset type being written is of type
  //!   ``cuda::std::int64_t``.
  //! - For zero-length inputs, ``cuda::std::numeric_limits<T>::max()}`` is written to ``d_min_out``  and the index
  //!   ``1`` is written to ``d_index_out``.
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Provides determinism based on the environment's determinism requirements.
  //!   To request "run-to-run" determinism, pass ``cuda::execution::require(cuda::execution::determinism::run_to_run)``
  //!   as the `env` parameter.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_min_out`` nor ``d_index_out``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin argmin-env-determinism
  //!     :end-before: example-end argmin-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam ExtremumOutIteratorT
  //!   **[inferred]** Output iterator type for recording minimum value
  //!
  //! @tparam IndexOutIteratorT
  //!   **[inferred]** Output iterator type for recording index of the returned value
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Iterator to the input sequence of data items
  //!
  //! @param[out] d_min_out
  //!   Iterator to which the minimum value is written
  //!
  //! @param[out] d_index_out
  //!   Iterator to which the index of the returned value is written
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename ExtremumOutIteratorT,
            typename IndexOutIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ArgMin(InputIteratorT d_in,
         ExtremumOutIteratorT d_min_out,
         IndexOutIteratorT d_index_out,
         ::cuda::std::int64_t num_items,
         EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceReduce::ArgMin");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    // Static assert to reject gpu_to_gpu determinism since it's not properly implemented
    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported");

    // Query relevant properties from the environment
    auto stream = ::cuda::std::execution::__query_or(env, ::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}});
    auto mr =
      ::cuda::std::execution::__query_or(env, ::cuda::mr::__get_memory_resource, detail::device_memory_resource{});

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    using tuning_t =
      ::cuda::std::execution::__query_result_or_t<EnvT, ::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>>;

    // Reduction operation
    using ReduceOpT           = cub::ArgMin;
    using InputValueT         = cub::detail::it_value_t<InputIteratorT>;
    using PerPartitionOffsetT = int;
    using GlobalOffsetT       = ::cuda::std::int64_t;

    using OutputExtremumT = detail::non_void_value_t<ExtremumOutIteratorT, InputValueT>;
    using InitT           = OutputExtremumT;

    // Initial value
    OutputExtremumT initial_value{::cuda::std::numeric_limits<InputValueT>::max()};

    // Tabulate output iterator that unzips the result and writes it to the user-provided output iterators
    auto out_it = ::cuda::make_tabulate_output_iterator(
      detail::reduce::unzip_and_write_arg_extremum_op<ExtremumOutIteratorT, IndexOutIteratorT>{d_min_out, d_index_out});

    // Query the required temporary storage size
    cudaError_t error = detail::reduce::dispatch_streaming_arg_reduce_t<
      InputIteratorT,
      decltype(out_it),
      PerPartitionOffsetT,
      GlobalOffsetT,
      ReduceOpT,
      InitT>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       out_it,
                       static_cast<GlobalOffsetT>(num_items),
                       ReduceOpT{},
                       initial_value,
                       stream.get());
    if (error != cudaSuccess)
    {
      return error;
    }

    // TODO(gevtushenko): use uninitialized buffer when it's available
    error = CubDebug(detail::temporary_storage::allocate(stream, d_temp_storage, temp_storage_bytes, mr));
    if (error != cudaSuccess)
    {
      return error;
    }

    // Run the algorithm
    error = detail::reduce::dispatch_streaming_arg_reduce_t<
      InputIteratorT,
      decltype(out_it),
      PerPartitionOffsetT,
      GlobalOffsetT,
      ReduceOpT,
      InitT>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       out_it,
                       static_cast<GlobalOffsetT>(num_items),
                       ReduceOpT{},
                       initial_value,
                       stream.get());

    // Try to deallocate regardless of the error to avoid memory leaks
    cudaError_t deallocate_error =
      CubDebug(detail::temporary_storage::deallocate(stream, d_temp_storage, temp_storage_bytes, mr));

    if (error != cudaSuccess)
    {
      // Reduction error takes precedence over deallocation error since it happens first
      return error;
    }

    return deallocate_error;
  }

  //! @rst
  //! Finds the first device-wide minimum using the less-than (``<``) operator, also returning the index of that item.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum is written to ``d_out.value`` and its offset in the input array is written to ``d_out.key``.
  //!   - The ``{1, cuda::std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector
  //! of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int                      num_items;      // e.g., 7
  //!    int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    KeyValuePair<int, int>   *d_argmin;      // e.g., [{-,-}]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run argmin-reduction
  //!    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
  //!
  //!    // d_argmin <-- [{5, 0}]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type ``cub::KeyValuePair<int, T>``) @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CCCL_DEPRECATED_BECAUSE("CUB has superseded this interface in favor of the ArgMin interface that takes two separate "
                          "iterators: one iterator to which the extremum is written and another iterator to which the "
                          "index of the found extremum is written. ") CUB_RUNTIME_FUNCTION static cudaError_t
  ArgMin(void* d_temp_storage,
         size_t& temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         int num_items,
         cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMin");

    // Signed integer type for global offsets
    using OffsetT = int;

    // The input type
    using InputValueT = cub::detail::it_value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;

    using AccumT = OutputTupleT;

    using InitT = detail::reduce::empty_problem_init_t<AccumT>;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value
    InitT initial_value{AccumT(1, ::cuda::std::numeric_limits<InputValueT>::max())};

    return detail::reduce::dispatch<AccumT>(
      d_temp_storage, temp_storage_bytes, d_indexed_in, d_out, OffsetT{num_items}, cub::ArgMin(), initial_value, stream);
  }

  //! @rst
  //! Computes a device-wide maximum using the greater-than (``>``) operator.
  //!
  //! - Uses ``cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_max;         // e.g., [-]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run max-reduction
  //!    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
  //!
  //!    // d_max <-- [9]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      NumItemsT num_items,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Max");

    // Signed integer type for global offsets
    using OffsetT  = detail::choose_offset_t<NumItemsT>;
    using InputT   = detail::it_value_t<InputIteratorT>;
    using InitT    = InputT;
    using limits_t = ::cuda::std::numeric_limits<InitT>;
#ifndef CCCL_SUPPRESS_NUMERIC_LIMITS_CHECK_IN_CUB_DEVICE_REDUCE_MIN_MAX
    static_assert(limits_t::is_specialized,
                  "cub::DeviceReduce::Max uses cuda::std::numeric_limits<InputIteratorT::value_type>::lowest() as "
                  "initial value, but cuda::std::numeric_limits is not specialized for the iterator's value type. This "
                  "is probably a bug and you should specialize cuda::std::numeric_limits. Define "
                  "CCCL_SUPPRESS_NUMERIC_LIMITS_CHECK_IN_CUB_DEVICE_REDUCE_MIN_MAX to suppress this check.");
#endif // CCCL_SUPPRESS_NUMERIC_LIMITS_CHECK_IN_CUB_DEVICE_REDUCE_MIN_MAX

    return detail::reduce::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      ::cuda::maximum<>{},
      limits_t::lowest(),
      stream);
  }

  //! @rst
  //! Computes a device-wide maximum using the greater-than (``>``) operator. The result is written to the output
  //! iterator.
  //!
  //! - Uses ``cuda::std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - Provides determinism based on the environment's determinism requirements.
  //!   To request "run-to-run" determinism, pass ``cuda::execution::require(cuda::execution::determinism::run_to_run)``
  //!   as the `env` parameter.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin max-env-determinism
  //!     :end-before: example-end max-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is `cuda::std::execution::env<>`.
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  Max(InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceReduce::Max");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    // Static assert to reject gpu_to_gpu determinism since it's not properly implemented
    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported");

    // TODO(NaderAlAwar): Relax this once non-deterministic implementation for min / max is available
    using determinism_t = ::cuda::execution::determinism::run_to_run_t;

    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>;

    using InitT    = OutputT;
    using limits_t = ::cuda::std::numeric_limits<InitT>;

    // Dispatch with environment - handles all boilerplate
    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return reduce_impl<tuning_t>(
        storage,
        bytes,
        d_in,
        d_out,
        num_items,
        ::cuda::maximum<>{},
        ::cuda::std::identity{},
        limits_t::lowest(),
        determinism_t{},
        stream);
    });
  }

  //! @rst
  //! Finds the first device-wide maximum using the greater-than (``>``) operator and also returns the index of that
  //! item.
  //!
  //! - The maximum is written to ``d_max_out``
  //! - The offset of the returned item is written to ``d_index_out``, the offset type being written is of type
  //!   ``cuda::std::int64_t``.
  //! - For zero-length inputs, ``cuda::std::numeric_limits<T>::max()}`` is written to ``d_max_out``  and the index
  //!   ``1`` is written to ``d_index_out``.
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_reduce.cuh>
  //!    #include <cuda/std/cstdint>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int                num_items;    // e.g., 7
  //!    int                *d_in;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int                *d_max_out;   // memory for the maximum value
  //!    cuda::std::int64_t *d_index_out; // memory for the index of the returned value
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ArgMax(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_max_out, d_index_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run argmax-reduction
  //!    cub::DeviceReduce::ArgMax(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_max_out, d_index_out, num_items);
  //!
  //!    // d_max_out   <-- 9
  //!    // d_index_out <-- 6
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam ExtremumOutIteratorT
  //!   **[inferred]** Output iterator type for recording maximum value
  //!
  //! @tparam IndexOutIteratorT
  //!   **[inferred]** Output iterator type for recording index of the returned value
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_max_out
  //!   Iterator to which the maximum value is written
  //!
  //! @param[out] d_index_out
  //!   Iterator to which the index of the returned value is written
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename ExtremumOutIteratorT, typename IndexOutIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    ExtremumOutIteratorT d_max_out,
    IndexOutIteratorT d_index_out,
    ::cuda::std::int64_t num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMax");

    // The input type
    using InputValueT = cub::detail::it_value_t<InputIteratorT>;

    // Offset type used within the kernel and to index within one partition
    using PerPartitionOffsetT = int;

    // Offset type used to index within the total input in the range [d_in, d_in + num_items)
    using GlobalOffsetT = ::cuda::std::int64_t;

    // The value type used for the extremum
    using OutputExtremumT = detail::non_void_value_t<ExtremumOutIteratorT, InputValueT>;
    using InitT           = OutputExtremumT;

    // Reduction operation
    using ReduceOpT = cub::ArgMax;

    // Initial value
    OutputExtremumT initial_value{::cuda::std::numeric_limits<InputValueT>::lowest()};

    // Tabulate output iterator that unzips the result and writes it to the user-provided output iterators
    auto out_it = ::cuda::make_tabulate_output_iterator(
      detail::reduce::unzip_and_write_arg_extremum_op<ExtremumOutIteratorT, IndexOutIteratorT>{d_max_out, d_index_out});

    return detail::reduce::dispatch_streaming_arg_reduce_t<
      InputIteratorT,
      decltype(out_it),
      PerPartitionOffsetT,
      GlobalOffsetT,
      ReduceOpT,
      InitT>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       out_it,
                       static_cast<GlobalOffsetT>(num_items),
                       ReduceOpT{},
                       initial_value,
                       stream);
  }

  //! @rst
  //! Finds the first device-wide maximum using the greater-than (``>``)
  //! operator, also returning the index of that item
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum is written to ``d_out.value`` and its offset in the input
  //!     array is written to ``d_out.key``.
  //!   - The ``{1, cuda::std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int                      num_items;      // e.g., 7
  //!    int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    KeyValuePair<int, int>   *d_argmax;      // e.g., [{-,-}]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ArgMax(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run argmax-reduction
  //!    cub::DeviceReduce::ArgMax(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
  //!
  //!    // d_argmax <-- [{6, 9}]
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
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CCCL_DEPRECATED_BECAUSE("CUB has superseded this interface in favor of the ArgMax interface that takes two separate "
                          "iterators: one iterator to which the extremum is written and another iterator to which the "
                          "index of the found extremum is written. ") CUB_RUNTIME_FUNCTION static cudaError_t
  ArgMax(void* d_temp_storage,
         size_t& temp_storage_bytes,
         InputIteratorT d_in,
         OutputIteratorT d_out,
         int num_items,
         cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMax");

    // Signed integer type for global offsets
    using OffsetT = int;

    // The input type
    using InputValueT = cub::detail::it_value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;

    using AccumT = OutputTupleT;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    using InitT = detail::reduce::empty_problem_init_t<AccumT>;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value
    InitT initial_value{AccumT(1, ::cuda::std::numeric_limits<InputValueT>::lowest())};

    return detail::reduce::dispatch<AccumT>(
      d_temp_storage, temp_storage_bytes, d_indexed_in, d_out, OffsetT{num_items}, cub::ArgMax(), initial_value, stream);
  }

  //! @rst
  //! Finds the first device-wide maximum using the greater-than (``>``) operator and also returns the index of that
  //! item.
  //!
  //! - The maximum is written to ``d_max_out``
  //! - The offset of the returned item is written to ``d_index_out``, the offset type being written is of type
  //!   ``cuda::std::int64_t``.
  //! - For zero-length inputs, ``cuda::std::numeric_limits<T>::lowest()}`` is written to ``d_max_out``  and the index
  //!   ``1`` is written to ``d_index_out``.
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Provides determinism based on the environment's determinism requirements.
  //!   To request "run-to-run" determinism, pass ``cuda::execution::require(cuda::execution::determinism::run_to_run)``
  //!   as the `env` parameter.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_max_out`` nor ``d_index_out``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector of ``int`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_reduce_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin argmax-env-determinism
  //!     :end-before: example-end argmax-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam ExtremumOutIteratorT
  //!   **[inferred]** Output iterator type for recording maximum value
  //!
  //! @tparam IndexOutIteratorT
  //!   **[inferred]** Output iterator type for recording index of the returned value
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Iterator to the input sequence of data items
  //!
  //! @param[out] d_max_out
  //!   Iterator to which the maximum value is written
  //!
  //! @param[out] d_index_out
  //!   Iterator to which the index of the returned value is written
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename ExtremumOutIteratorT,
            typename IndexOutIteratorT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ArgMax(InputIteratorT d_in,
         ExtremumOutIteratorT d_max_out,
         IndexOutIteratorT d_index_out,
         ::cuda::std::int64_t num_items,
         EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceReduce::ArgMax");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::run_to_run_t>;

    // Static assert to reject gpu_to_gpu determinism since it's not properly implemented
    static_assert(!::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::gpu_to_gpu_t>,
                  "gpu_to_gpu determinism is not supported");

    // Query relevant properties from the environment
    auto stream = ::cuda::std::execution::__query_or(env, ::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}});
    auto mr =
      ::cuda::std::execution::__query_or(env, ::cuda::mr::__get_memory_resource, detail::device_memory_resource{});

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    using tuning_t =
      ::cuda::std::execution::__query_result_or_t<EnvT, ::cuda::execution::__get_tuning_t, ::cuda::std::execution::env<>>;

    // Reduction operation
    using ReduceOpT           = cub::ArgMax;
    using InputValueT         = cub::detail::it_value_t<InputIteratorT>;
    using PerPartitionOffsetT = int;
    using GlobalOffsetT       = ::cuda::std::int64_t;

    using OutputExtremumT = detail::non_void_value_t<ExtremumOutIteratorT, InputValueT>;
    using InitT           = OutputExtremumT;

    // Initial value
    OutputExtremumT initial_value{::cuda::std::numeric_limits<InputValueT>::lowest()};

    // Tabulate output iterator that unzips the result and writes it to the user-provided output iterators
    auto out_it = ::cuda::make_tabulate_output_iterator(
      detail::reduce::unzip_and_write_arg_extremum_op<ExtremumOutIteratorT, IndexOutIteratorT>{d_max_out, d_index_out});

    // Query the required temporary storage size
    cudaError_t error = detail::reduce::dispatch_streaming_arg_reduce_t<
      InputIteratorT,
      decltype(out_it),
      PerPartitionOffsetT,
      GlobalOffsetT,
      ReduceOpT,
      InitT>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       out_it,
                       static_cast<GlobalOffsetT>(num_items),
                       ReduceOpT{},
                       initial_value,
                       stream.get());
    if (error != cudaSuccess)
    {
      return error;
    }

    // TODO(gevtushenko): use uninitialized buffer when it's available
    error = CubDebug(detail::temporary_storage::allocate(stream, d_temp_storage, temp_storage_bytes, mr));
    if (error != cudaSuccess)
    {
      return error;
    }

    // Run the algorithm
    error = detail::reduce::dispatch_streaming_arg_reduce_t<
      InputIteratorT,
      decltype(out_it),
      PerPartitionOffsetT,
      GlobalOffsetT,
      ReduceOpT,
      InitT>::Dispatch(d_temp_storage,
                       temp_storage_bytes,
                       d_in,
                       out_it,
                       static_cast<GlobalOffsetT>(num_items),
                       ReduceOpT{},
                       initial_value,
                       stream.get());

    // Try to deallocate regardless of the error to avoid memory leaks
    cudaError_t deallocate_error =
      CubDebug(detail::temporary_storage::deallocate(stream, d_temp_storage, temp_storage_bytes, mr));

    if (error != cudaSuccess)
    {
      // Reduction error takes precedence over deallocation error since it happens first
      return error;
    }

    return deallocate_error;
  }

  //! @rst
  //! Fuses transform and reduce operations
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined min-reduction of a
  //! device vector of `int` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    thrust::device_vector<int> in = { 1, 2, 3, 4 };
  //!    thrust::device_vector<int> out(1);
  //!
  //!    size_t temp_storage_bytes = 0;
  //!    uint8_t *d_temp_storage = nullptr;
  //!
  //!    const int init = 42;
  //!
  //!    cub::DeviceReduce::TransformReduce(
  //!      d_temp_storage,
  //!      temp_storage_bytes,
  //!      in.begin(),
  //!      out.begin(),
  //!      in.size(),
  //!      cuda::std::plus<>{},
  //!      square_t{},
  //!      init);
  //!
  //!    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  //!    d_temp_storage = temp_storage.data().get();
  //!
  //!    cub::DeviceReduce::TransformReduce(
  //!      d_temp_storage,
  //!      temp_storage_bytes,
  //!      in.begin(),
  //!      out.begin(),
  //!      in.size(),
  //!      cuda::std::plus<>{},
  //!      square_t{},
  //!      init);
  //!
  //!    // out[0] <-- 72
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
  //! @tparam TransformOpT
  //!   **[inferred]** Unary reduction functor type having member `auto operator()(const T &a)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of ``d_in``)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] transform_op
  //!   Unary transform functor
  //!
  //! @param[in] init
  //!   Initial value of the reduction
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename TransformOpT,
            typename T,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    TransformOpT transform_op,
    T init,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::TransformReduce");

    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::reduce::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      reduction_op,
      init,
      stream,
      transform_op);
  }

  //! @rst
  //! Reduces segments of values, where segments are demarcated by corresponding runs of identical keys.
  //!
  //! This operation computes segmented reductions within ``d_values_in`` using the specified binary ``reduction_op``
  //! functor. The segments are identified by "runs" of corresponding keys in `d_keys_in`, where runs are maximal
  //! ranges of consecutive, identical keys. For the *i*\ :sup:`th` run encountered, the first key of the run and
  //! the corresponding value aggregate of that run are written to ``d_unique_out[i]`` and ``d_aggregates_out[i]``,
  //! respectively. The total number of runs encountered is written to ``d_num_runs_out``.
  //!
  //! - The ``==`` equality operator is used to determine whether keys are equivalent
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - Let ``out`` be any of
  //!   ``[d_unique_out, d_unique_out + *d_num_runs_out)``
  //!   ``[d_aggregates_out, d_aggregates_out + *d_num_runs_out)``
  //!   ``d_num_runs_out``. The ranges represented by ``out`` shall not overlap
  //!   ``[d_keys_in, d_keys_in + num_items)``,
  //!   ``[d_values_in, d_values_in + num_items)`` nor ``out`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the segmented reduction of ``int`` values grouped by runs of
  //! associated ``int`` keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int          num_items;          // e.g., 8
  //!    int          *d_keys_in;         // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int          *d_values_in;       // e.g., [0, 7, 1, 6, 2, 5, 3, 4]
  //!    int          *d_unique_out;      // e.g., [-, -, -, -, -, -, -, -]
  //!    int          *d_aggregates_out;  // e.g., [-, -, -, -, -, -, -, -]
  //!    int          *d_num_runs_out;    // e.g., [-]
  //!    CustomMin    reduction_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ReduceByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_unique_out, d_values_in,
  //!      d_aggregates_out, d_num_runs_out, reduction_op, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run reduce-by-key
  //!    cub::DeviceReduce::ReduceByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_unique_out, d_values_in,
  //!      d_aggregates_out, d_num_runs_out, reduction_op, num_items);
  //!
  //!    // d_unique_out      <-- [0, 2, 9, 5, 8]
  //!    // d_aggregates_out  <-- [0, 1, 6, 2, 4]
  //!    // d_num_runs_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam UniqueOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing unique output keys @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam AggregatesOutputIterator
  //!   **[inferred]** Random-access output iterator type for writing output value aggregates @iterator
  //!
  //! @tparam NumRunsOutputIteratorT
  //!   **[inferred]** Output iterator type for recording the number of runs encountered @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of ``d_temp_storage`` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input sequence of keys
  //!
  //! @param[out] d_unique_out
  //!   Pointer to the output sequence of unique keys (one key per run)
  //!
  //! @param[in] d_values_in
  //!   Pointer to the input sequence of corresponding values
  //!
  //! @param[out] d_aggregates_out
  //!   Pointer to the output sequence of value aggregates
  //!   (one aggregate per run)
  //!
  //! @param[out] d_num_runs_out
  //!   Pointer to total number of runs encountered
  //!   (i.e., the length of ``d_unique_out``)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] num_items
  //!   Total number of associated key+value pairs
  //!   (i.e., the length of ``d_in_keys`` and ``d_in_values``)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT,
            typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename ReductionOpT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t ReduceByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    ReductionOpT reduction_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ReduceByKey");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    // FlagT iterator type (not used)

    // Selection op (not used)

    // Default == operator
    using EqualityOp = ::cuda::std::equal_to<>;

    return DispatchReduceByKey<
      KeysInputIteratorT,
      UniqueOutputIteratorT,
      ValuesInputIteratorT,
      AggregatesOutputIteratorT,
      NumRunsOutputIteratorT,
      EqualityOp,
      ReductionOpT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_unique_out,
                         d_values_in,
                         d_aggregates_out,
                         d_num_runs_out,
                         EqualityOp(),
                         reduction_op,
                         static_cast<OffsetT>(num_items),
                         stream);
  }
};
CUB_NAMESPACE_END
