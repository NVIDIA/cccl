// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data
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
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/dispatch_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tune.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_null_pointer.h>
#include <cuda/std/__type_traits/is_same.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceScan provides device-wide, parallel operations for computing a
//! prefix scan across a sequence of data items residing within
//! device-accessible memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! Given a sequence of input elements and a binary reduction operator, a
//! `prefix scan <http://en.wikipedia.org/wiki/Prefix_sum>`_ produces an output
//! sequence where each element is computed to be the reduction of the elements
//! occurring earlier in the input sequence. *Prefix sum* connotes a prefix scan
//! with the addition operator. The term *inclusive* indicates that the
//! *i*\ :sup:`th` output reduction incorporates the *i*\ :sup:`th` input.
//! The term *exclusive* indicates the *i*\ :sup:`th` input is not
//! incorporated into the *i*\ :sup:`th` output reduction. When the input and
//! output sequences are the same, the scan is performed in-place.
//!
//! In order to provide an efficient parallel implementation, the binary reduction operator must be associative. That
//! is, ``op(op(a, b), c)`` must be equivalent to ``op(a, op(b, c))`` for any input values ``a``, ``b``, and ``c``.
//!
//! As of CUB 1.0.1 (2013), CUB's device-wide scan APIs have implemented our
//! *"decoupled look-back"* algorithm for performing global prefix scan with
//! only a single pass through the input data, as described in our 2016 technical
//! report [1]_. The central idea is to leverage a small, constant factor of
//! redundant work in order to overlap the latencies of global prefix
//! propagation with local computation. As such, our algorithm requires only
//! ``~2*n*`` data movement (``n`` inputs are read, ``n`` outputs are written), and
//! typically proceeds at "memcpy" speeds. Our algorithm supports inplace operations.
//!
//! .. [1] Duane Merrill and Michael Garland. `Single-pass Parallel Prefix Scan with Decoupled Look-back
//!    <https://research.nvidia.com/publication/single-pass-parallel-prefix-scan-decoupled-look-back>`_,
//!    *NVIDIA Technical Report NVR-2016-002*, 2016.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceScan}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{prefix scan}
//!
//! @endrst
struct DeviceScan
{
  //! @cond
  template <typename TuningEnvT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename NumItemsT,
            ::cuda::execution::determinism::__determinism_t Determinism,
            ForceInclusive EnforceInclusive = ForceInclusive::No>
  CUB_RUNTIME_FUNCTION static cudaError_t scan_impl_determinism(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init,
    NumItemsT num_items,
    ::cuda::execution::determinism::__determinism_holder_t<Determinism>,
    cudaStream_t stream)
  {
    // Unsigned integer type for global offsets
    using offset_t = detail::choose_offset_t<NumItemsT>;

    using accum_t =
      ::cuda::std::__accumulator_t<ScanOpT,
                                   cub::detail::it_value_t<InputIteratorT>,
                                   ::cuda::std::_If<::cuda::std::is_same_v<InitValueT, NullType>,
                                                    cub::detail::it_value_t<InputIteratorT>,
                                                    typename InitValueT::value_type>>;

    using default_policy_selector_t =
      detail::scan::policy_selector_from_types<InputIteratorT, OutputIteratorT, accum_t, offset_t, ScanOpT>;

    using policy_selector_t =
      ::cuda::std::execution::__query_result_or_t<TuningEnvT, detail::scan::scan_policy, default_policy_selector_t>;

    return detail::scan::dispatch<EnforceInclusive>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      init,
      static_cast<offset_t>(num_items),
      stream,
      policy_selector_t{});
  }

  template <ForceInclusive EnforceInclusive = ForceInclusive::No,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename NumItemsT,
            typename EnvT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t scan_impl_env(
    InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT init, NumItemsT num_items, EnvT env)
  {
    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");

    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;

    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t,
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::not_guaranteed_t>;

    using accum_t =
      ::cuda::std::__accumulator_t<ScanOpT,
                                   cub::detail::it_value_t<InputIteratorT>,
                                   ::cuda::std::_If<::cuda::std::is_same_v<InitValueT, NullType>,
                                                    cub::detail::it_value_t<InputIteratorT>,
                                                    typename InitValueT::value_type>>;

    constexpr bool is_determinism_required =
      !::cuda::std::is_same_v<requested_determinism_t, ::cuda::execution::determinism::not_guaranteed_t>;
    constexpr bool is_safe_integral_op =
      ::cuda::std::is_integral_v<accum_t> && detail::is_cuda_binary_operator<ScanOpT>;

    // Logic: If determinism is required, we must have a safe integral operator.
    static_assert(!is_determinism_required || is_safe_integral_op,
                  "run_to_run or gpu_to_gpu is only supported for integral types with known operators");

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return scan_impl_determinism<
        tuning_t,
        InputIteratorT,
        OutputIteratorT,
        ScanOpT,
        InitValueT,
        NumItemsT,
        ::cuda::execution::determinism::__determinism_t(requested_determinism_t::value),
        EnforceInclusive>(storage, bytes, d_in, d_out, scan_op, init, num_items, requested_determinism_t{}, stream);
    });
  }

  template <typename TuningEnvT,
            typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT,
            typename ScanOpT,
            typename InitValueT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t scan_by_key_impl(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    EqualityOpT equality_op,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream)
  {
    using offset_t = detail::choose_offset_t<NumItemsT>;
    using accum_t  = ::cuda::std::__accumulator_t<
       ScanOpT,
       cub::detail::it_value_t<ValuesInputIteratorT>,
       ::cuda::std::
         _If<::cuda::std::is_same_v<InitValueT, NullType>, cub::detail::it_value_t<ValuesInputIteratorT>, InitValueT>>;

    using default_policy_selector_t =
      detail::scan_by_key::policy_selector_from_types<detail::it_value_t<KeysInputIteratorT>,
                                                      accum_t,
                                                      cub::detail::it_value_t<ValuesInputIteratorT>,
                                                      ScanOpT>;

    using policy_selector_t = ::cuda::std::execution::
      __query_result_or_t<TuningEnvT, detail::scan_by_key::scan_by_key_policy, default_policy_selector_t>;

    return detail::scan_by_key::dispatch<
      KeysInputIteratorT,
      ValuesInputIteratorT,
      ValuesOutputIteratorT,
      EqualityOpT,
      ScanOpT,
      InitValueT,
      offset_t,
      accum_t>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_values_out,
      equality_op,
      scan_op,
      init_value,
      static_cast<offset_t>(num_items),
      stream,
      policy_selector_t{});
  }
  //! @endcond

  //! @name Exclusive scans
  //! @{

  //! @rst
  //! Computes a device-wide exclusive prefix sum.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // d_out <-- [0, 8, 14, 21, 26, 29, 29]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveSum");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using InitT   = cub::detail::it_value_t<InputIteratorT>;

    // Initial value
    InitT init_value{};

    return detail::scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      ::cuda::std::plus<>{},
      detail::InputValue<InitT>(init_value),
      static_cast<OffsetT>(num_items),
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_out``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined exclusive-scan of a
  //! device vector of ``float`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-sum-env-determinism
  //!     :end-before: example-end exclusive-sum-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<::cuda::std::is_integral_v<NumItemsT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ExclusiveSum(InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::ExclusiveSum");

    using init_t = cub::detail::it_value_t<InputIteratorT>;
    init_t init_value{};

    return scan_impl_env(d_in, d_out, ::cuda::std::plus<>{}, detail::InputValue<init_t>(init_value), num_items, env);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum in-place.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_data``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum of an ``int``
  //! device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // d_data <-- [0, 8, 14, 21, 26, 29, 29]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading scan inputs and wrigin scan outputs
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_data`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    return ExclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum in-place.
  //! The value of ``0`` is applied as the initial value, and is assigned to ``*d_data``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an in-place exclusive prefix sum of an
  //! ``int`` device vector using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-sum-inplace-env
  //!     :end-before: example-end exclusive-sum-inplace-env
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing scan data
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_data`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename IteratorT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<!::cuda::std::is_integral_v<EnvT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ExclusiveSum(IteratorT d_data, NumItemsT num_items, EnvT env = {})
  {
    return ExclusiveSum(d_data, d_data, num_items, env);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, (int) INT_MAX, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, (int) INT_MAX, num_items);
  //!
  //!    // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      static_cast<OffsetT>(num_items),
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_out``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined exclusive-scan of a
  //! device vector of ``float`` data elements.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-scan-env-determinism
  //!     :end-before: example-end exclusive-scan-env-determinism
  //!
  //! The code snippet below illustrates an exclusive-scan using a custom stream
  //! and ``not_guaranteed`` determinism.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-scan-env-stream
  //!     :end-before: example-end exclusive-scan-env-stream
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<::cuda::std::is_integral_v<NumItemsT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::ExclusiveScan");

    return scan_impl_env(d_in, d_out, scan_op, detail::InputValue<InitValueT>(init_value), num_items, env);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_data``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an
  //! ``int`` device vector:
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, (int) INT_MAX, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, (int) INT_MAX, num_items);
  //!
  //!    // d_data <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, init_value, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan in-place using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is applied as
  //! the initial value, and is assigned to ``*d_data``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an in-place exclusive prefix scan of an
  //! ``int`` device vector with a user-supplied initial value and a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-scan-inplace-env
  //!     :end-before: example-end exclusive-scan-inplace-env
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing scan data
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!   **[inferred]** Type of the ``init_value``
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_data`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename IteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<!::cuda::std::is_integral_v<EnvT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  ExclusiveScan(IteratorT d_data, ScanOpT scan_op, InitValueT init_value, NumItemsT num_items, EnvT env = {})
  {
    return ExclusiveScan(d_data, d_data, scan_op, init_value, num_items, env);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is provided as a future value.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    int          *d_init_iter;   // e.g., INT_MAX
  //!    CustomMin    min_op;
  //!
  //!    auto future_init_value =
  //!      cub::FutureValue<InitialValueT, IterT>(d_init_iter);
  //!
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, future_init_value, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, future_init_value, num_items);
  //!
  //!    // d_out <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      static_cast<OffsetT>(num_items),
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The ``init_value`` value is provided as a future value.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_init_iter;   // e.g., INT_MAX
  //!    CustomMin    min_op;
  //!
  //!    auto future_init_value =
  //!      cub::FutureValue<InitialValueT, IterT>(d_init_iter);
  //!
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, future_init_value, num_items);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, future_init_value, num_items);
  //!
  //!    // d_data <-- [2147483647, 8, 6, 6, 5, 3, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Pointer to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    return ExclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, init_value, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan in-place using the specified
  //! binary associative ``scan_op`` functor. The ``init_value`` value is provided as a future value.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an in-place exclusive prefix scan of an
  //! ``int`` device vector with a future initial value and a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-scan-future-inplace-env
  //!     :end-before: example-end exclusive-scan-future-inplace-env
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing scan data
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!   **[inferred]** Type of the ``init_value``
  //!
  //! @tparam InitValueIterT
  //!   **[inferred]** Iterator type for the future value
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan, provided as a future value
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_data`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename IteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT = InitValueT*,
            typename NumItemsT      = int,
            typename EnvT           = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<!::cuda::std::is_integral_v<EnvT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    IteratorT d_data,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    EnvT env = {})
  {
    return ExclusiveScan(d_data, d_data, scan_op, init_value, num_items, env);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The ``init_value`` value is provided as a future value.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Can use a specific stream or cuda memory resource through the ``env`` parameter.
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan of an ``int`` device vector
  //! using a future value for the initial value.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-scan-future-env
  //!     :end-before: example-end exclusive-scan-future-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam InitValueIterT
  //!  **[inferred]** Random-access iterator type used to access the initial value on device
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to `*d_out`),
  //!   provided as a future value
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename InitValueIterT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<::cuda::std::is_integral_v<NumItemsT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScan(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    FutureValue<InitValueT, InitValueIterT> init_value,
    NumItemsT num_items,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::ExclusiveScan");

    return scan_impl_env(d_in, d_out, scan_op, detail::InputValue<InitValueT>(init_value), num_items, env);
  }

  //! @}

  //! @name Inclusive scans
  //! @{

  //! @rst
  //! Computes a device-wide inclusive prefix sum.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items);
  //!
  //!    // d_out <-- [8, 14, 21, 26, 29, 29, 38]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveSum");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::scan::dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      ::cuda::std::plus<>{},
      NullType{},
      static_cast<OffsetT>(num_items),
      stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum in-place.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSum(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items);
  //!
  //!    // d_data <-- [8, 14, 21, 26, 29, 29, 38]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSum(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    return InclusiveSum(d_temp_storage, temp_storage_bytes, d_data, d_data, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum in-place.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an in-place inclusive prefix sum of an
  //! ``int`` device vector using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-sum-inplace-env
  //!     :end-before: example-end inclusive-sum-inplace-env
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing scan data
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_data`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename IteratorT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<!::cuda::std::is_integral_v<EnvT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  InclusiveSum(IteratorT d_data, NumItemsT num_items, EnvT env = {})
  {
    return InclusiveSum(d_data, d_data, num_items, env);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place.
  //!   The range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-sum-env-determinism
  //!     :end-before: example-end inclusive-sum-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<::cuda::std::is_integral_v<NumItemsT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  InclusiveSum(InputIteratorT d_in, OutputIteratorT d_out, NumItemsT num_items, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::InclusiveSum");

    return scan_impl_env(d_in, d_out, ::cuda::std::plus<>{}, NullType{}, num_items, env);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // d_out <-- [8, 6, 6, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in]
  //!   d_temp_storage Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScan");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::scan::dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, NullType(), static_cast<OffsetT>(num_items), stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and ``*d_in``
  //! is assigned to ``*d_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive max-scan of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin device-inclusive-scan
  //!     :end-before: example-end device-inclusive-scan
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to the size in bytes of the `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the inclusive scan (`scan_op(init_value, d_in[0])`
  //!   is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   CUDA stream to launch kernels within.
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanInit(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScanInit");

    // Unsigned integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::scan::dispatch<ForceInclusive::Yes>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      scan_op,
      detail::InputValue<InitValueT>(init_value),
      static_cast<OffsetT>(num_items),
      stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_data;        // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    CustomMin    min_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive
  //!    // prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, min_op, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScan(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, min_op, num_items);
  //!
  //!    // d_data <-- [8, 6, 6, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs and writing scan outputs
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in]
  //!   d_temp_storage Device-accessible allocation of temporary storage.
  //!   When `nullptr`, the required allocation size is written to
  //!   `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScan(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = nullptr)
  {
    return InclusiveScan(d_temp_storage, temp_storage_bytes, d_data, d_data, scan_op, num_items, stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan in-place using the specified
  //! binary associative ``scan_op`` functor.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates an in-place inclusive prefix scan of an
  //! ``int`` device vector using a stream environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-scan-inplace-env
  //!     :end-before: example-end inclusive-scan-inplace-env
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing scan data
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in,out] d_data
  //!   Random-access iterator to the sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_data`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename IteratorT,
            typename ScanOpT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<!::cuda::std::is_integral_v<EnvT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  InclusiveScan(IteratorT d_data, ScanOpT scan_op, NumItemsT num_items, EnvT env = {})
  {
    return InclusiveScan(d_data, d_data, scan_op, num_items, env);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-scan-env
  //!     :end-before: example-end inclusive-scan-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            ,
            ::cuda::std::enable_if_t<::cuda::std::is_integral_v<NumItemsT>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  InclusiveScan(InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, NumItemsT num_items, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::InclusiveScan");

    return scan_impl_env(d_in, d_out, scan_op, NullType{}, num_items, env);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan using the specified binary associative ``scan_op`` functor.
  //! The result of applying the ``scan_op`` binary operator to ``init_value`` value and ``*d_in``
  //! is assigned to ``*d_out``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - When ``d_in`` and ``d_out`` are equal, the scan is performed in-place. The
  //!   range ``[d_in, d_in + num_items)`` and ``[d_out, d_out + num_items)``
  //!   shall not overlap in any other way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum of an ``int`` device vector
  //! with an initial value.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-scan-init-env
  //!     :end-before: example-end inclusive-scan-init-env
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan inputs @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!  **[inferred]** Type of the `init_value`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_in
  //!   Random-access iterator to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Random-access iterator to the output sequence of data items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the inclusive scan (`scan_op(init_value, d_in[0])`
  //!   is assigned to `*d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename NumItemsT,
            typename EnvT = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void
#else
            ::cuda::std::execution::env<>
#endif
            >
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanInit(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::InclusiveScanInit");

    return scan_impl_env<ForceInclusive::Yes>(
      d_in, d_out, scan_op, detail::InputValue<InitValueT>(init_value), num_items, env);
  }

  //! @}

  //! @name Scans by key
  //! @{

  //! @rst
  //! Computes a device-wide exclusive prefix sum-by-key with key equality
  //! defined by ``equality_op``. The value of ``0`` is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int num_items;      // e.g., 7
  //!    int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix sum
  //!    cub::DeviceScan::ExclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // d_values_out <-- [0, 8, 0, 7, 12, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSumByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveSumByKey");
    using init_t = cub::detail::it_value_t<ValuesInputIteratorT>;
    init_t init_value{};
    return scan_by_key_impl<::cuda::std::execution::env<>>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_values_out,
      equality_op,
      ::cuda::std::plus<>{},
      init_value,
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan-by-key using the
  //! specified binary associative ``scan_op`` functor. The key equality is defined by
  //! ``equality_op``.  The ``init_value`` value is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the exclusive prefix min-scan-by-key of an ``int`` device vector
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // CustomEqual functor
  //!    struct CustomEqual
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return a == b;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    CustomEqual  equality_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for exclusive
  //!    // prefix scan
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::ExclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op,
  //!      (int) INT_MAX, num_items, equality_op);
  //!
  //!    // Allocate temporary storage for exclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run exclusive prefix min-scan
  //!    cub::DeviceScan::ExclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op,
  //!      (int) INT_MAX, num_items, equality_op);
  //!
  //!    // d_values_out <-- [2147483647, 8, 2147483647, 7, 5, 2147483647, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!   **[inferred]** Type of the `init_value`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage. When `nullptr`, the
  //!    required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] scan_op
  //!    Binary associative scan functor
  //!
  //!  @param[in] init_value
  //!    Initial value to seed the exclusive scan (and is assigned to the
  //!    beginning of each segment in `d_values_out`)
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and
  //!    `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::ExclusiveScanByKey");
    return scan_by_key_impl<::cuda::std::execution::env<>>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_values_out,
      equality_op,
      scan_op,
      init_value,
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum-by-key with key equality defined by ``equality_op``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_scan.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int num_items;      // e.g., 7
  //!    int *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive prefix sum
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // Allocate temporary storage for inclusive prefix sum
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix sum
  //!    cub::DeviceScan::InclusiveSumByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, num_items);
  //!
  //!    // d_out <-- [8, 14, 7, 12, 15, 0, 9]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage.
  //!    When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSumByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveSumByKey");
    return scan_by_key_impl<::cuda::std::execution::env<>>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_values_out,
      equality_op,
      ::cuda::std::plus<>{},
      NullType{},
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan-by-key using the
  //! specified binary associative ``scan_op`` functor. The key equality is defined by ``equality_op``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the inclusive prefix min-scan-by-key of an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>      // or equivalently <cub/device/device_scan.cuh>
  //!    #include <cuda/std/climits> // for INT_MAX
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // CustomEqual functor
  //!    struct CustomEqual
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return a == b;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;      // e.g., 7
  //!    int          *d_keys_in;     // e.g., [0, 0, 1, 1, 1, 2, 2]
  //!    int          *d_values_in;   // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_values_out;  // e.g., [ ,  ,  ,  ,  ,  ,  ]
  //!    CustomMin    min_op;
  //!    CustomEqual  equality_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements for inclusive prefix scan
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceScan::InclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
  //!
  //!    // Allocate temporary storage for inclusive prefix scan
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run inclusive prefix min-scan
  //!    cub::DeviceScan::InclusiveScanByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in, d_values_out, min_op, num_items, equality_op);
  //!
  //!    // d_out <-- [8, 6, 7, 5, 3, 0, 0]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //!  @param[in] d_temp_storage
  //!    Device-accessible allocation of temporary storage.
  //!    When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //!  @param[in,out] temp_storage_bytes
  //!    Reference to size in bytes of `d_temp_storage` allocation
  //!
  //!  @param[in] d_keys_in
  //!    Random-access input iterator to the input sequence of key items
  //!
  //!  @param[in] d_values_in
  //!    Random-access input iterator to the input sequence of value items
  //!
  //!  @param[out] d_values_out
  //!    Random-access output iterator to the output sequence of value items
  //!
  //!  @param[in] scan_op
  //!    Binary associative scan functor
  //!
  //!  @param[in] num_items
  //!    Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //!  @param[in] equality_op
  //!    Binary functor that defines the equality of keys.
  //!    Default is cuda::std::equal_to<>{}.
  //!
  //!  @param[in] stream
  //!    @rst
  //!    **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!    @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t>
  CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    cudaStream_t stream     = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceScan::InclusiveScanByKey");
    return scan_by_key_impl<::cuda::std::execution::env<>>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_values_out,
      equality_op,
      scan_op,
      NullType{},
      num_items,
      stream);
  }

  //! @rst
  //! Computes a device-wide exclusive prefix sum-by-key with key equality
  //! defined by ``equality_op``. The value of ``0`` is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //!
  //! Snippet
  //!
  //! The code snippet below illustrates the exclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_by_key_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-sum-by-key-env
  //!     :end-before: example-end exclusive-sum-by-key-env
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t,
            typename EnvT        = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void,
#else
            ::cuda::std::execution::env<>,
#endif
            ::cuda::std::enable_if_t<
              !::cuda::std::is_same_v<KeysInputIteratorT, void*> && !::cuda::std::is_null_pointer_v<KeysInputIteratorT>
                && !::cuda::std::is_same_v<ValuesInputIteratorT, size_t>,
              int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveSumByKey(
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    EnvT env                = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::ExclusiveSumByKey");

    using init_t = cub::detail::it_value_t<ValuesInputIteratorT>;
    init_t init_value{};

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using offset_t = detail::choose_offset_t<NumItemsT>;
      using tuning_t = decltype(tuning);
      using accum_t =
        ::cuda::std::__accumulator_t<::cuda::std::plus<>, cub::detail::it_value_t<ValuesInputIteratorT>, init_t>;
      using default_policy_selector_t =
        detail::scan_by_key::policy_selector_from_types<detail::it_value_t<KeysInputIteratorT>,
                                                        accum_t,
                                                        cub::detail::it_value_t<ValuesInputIteratorT>,
                                                        ::cuda::std::plus<>>;
      using policy_selector_t = ::cuda::std::execution::
        __query_result_or_t<tuning_t, detail::scan_by_key::scan_by_key_policy, default_policy_selector_t>;

      return detail::scan_by_key::dispatch<
        KeysInputIteratorT,
        ValuesInputIteratorT,
        ValuesOutputIteratorT,
        EqualityOpT,
        ::cuda::std::plus<>,
        init_t,
        offset_t,
        accum_t>(
        storage,
        bytes,
        d_keys_in,
        d_values_in,
        d_values_out,
        equality_op,
        ::cuda::std::plus<>{},
        init_value,
        static_cast<offset_t>(num_items),
        stream,
        policy_selector_t{});
    });
  }

  //! @rst
  //! Computes a device-wide exclusive prefix scan-by-key using the
  //! specified binary associative ``scan_op`` functor. The key equality is defined by
  //! ``equality_op``.  The ``init_value`` value is applied as the initial
  //! value, and is assigned to the beginning of each segment in ``d_values_out``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //!
  //! Snippet
  //!
  //! The code snippet below illustrates the exclusive prefix scan-by-key of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_by_key_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin exclusive-scan-by-key-env
  //!     :end-before: example-end exclusive-scan-by-key-env
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam InitValueT
  //!   **[inferred]** Type of the `init_value`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] init_value
  //!   Initial value to seed the exclusive scan (and is assigned to the
  //!   beginning of each segment in `d_values_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename InitValueT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t,
            typename EnvT        = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void,
#else
            ::cuda::std::execution::env<>,
#endif
            ::cuda::std::enable_if_t<
              !::cuda::std::is_same_v<KeysInputIteratorT, void*> && !::cuda::std::is_null_pointer_v<KeysInputIteratorT>
                && !::cuda::std::is_same_v<ValuesInputIteratorT, size_t>,
              int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t ExclusiveScanByKey(
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    InitValueT init_value,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    EnvT env                = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::ExclusiveScanByKey");

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return scan_by_key_impl<tuning_t>(
        storage, bytes, d_keys_in, d_values_in, d_values_out, equality_op, scan_op, init_value, num_items, stream);
    });
  }

  //! @rst
  //! Computes a device-wide inclusive prefix sum-by-key with key equality defined by ``equality_op``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative sum operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //!
  //! Snippet
  //!
  //! The code snippet below illustrates the inclusive prefix sum-by-key of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_by_key_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-sum-by-key-env
  //!     :end-before: example-end inclusive-sum-by-key-env
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t,
            typename EnvT        = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void,
#else
            ::cuda::std::execution::env<>,
#endif
            ::cuda::std::enable_if_t<
              !::cuda::std::is_same_v<KeysInputIteratorT, void*> && !::cuda::std::is_null_pointer_v<KeysInputIteratorT>
                && !::cuda::std::is_same_v<ValuesInputIteratorT, size_t>,
              int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveSumByKey(
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    EnvT env                = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::InclusiveSumByKey");

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return scan_by_key_impl<tuning_t>(
        storage,
        bytes,
        d_keys_in,
        d_values_in,
        d_values_out,
        equality_op,
        ::cuda::std::plus<>{},
        NullType{},
        num_items,
        stream);
    });
  }

  //! @rst
  //! Computes a device-wide inclusive prefix scan-by-key using the
  //! specified binary associative ``scan_op`` functor. The key equality is defined by ``equality_op``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! - Supports non-commutative scan operators.
  //! - Results are not deterministic for pseudo-associative operators (e.g.,
  //!   addition of floating-point types). Results for pseudo-associative
  //!   operators may vary from run to run. Additional details can be found in
  //!   the @lookback description.
  //! - ``d_keys_in`` may equal ``d_values_out`` but the range
  //!   ``[d_keys_in, d_keys_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //! - ``d_values_in`` may equal ``d_values_out`` but the range
  //!   ``[d_values_in, d_values_in + num_items)`` and the range
  //!   ``[d_values_out, d_values_out + num_items)`` shall not overlap otherwise.
  //!
  //! Snippet
  //!
  //! The code snippet below illustrates the inclusive prefix scan-by-key of an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_scan_by_key_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin inclusive-scan-by-key-env
  //!     :end-before: example-end inclusive-scan-by-key-env
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan keys inputs @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading scan values inputs @iterator
  //!
  //! @tparam ValuesOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing scan values outputs @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Binary associative scan functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Functor type having member
  //!   `T operator()(const T &a, const T &b)` for binary operations that defines the equality of keys
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** An integral type representing the number of input elements
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type providing stream, memory resource,
  //!   or determinism requirements. Default is ``cuda::std::execution::env<>``.
  //!
  //! @param[in] d_keys_in
  //!   Random-access input iterator to the input sequence of key items
  //!
  //! @param[in] d_values_in
  //!   Random-access input iterator to the input sequence of value items
  //!
  //! @param[out] d_values_out
  //!   Random-access output iterator to the output sequence of value items
  //!
  //! @param[in] scan_op
  //!   Binary associative scan functor
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_keys_in` and `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary functor that defines the equality of keys.
  //!   Default is cuda::std::equal_to<>{}.
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename ValuesInputIteratorT,
            typename ValuesOutputIteratorT,
            typename ScanOpT,
            typename EqualityOpT = ::cuda::std::equal_to<>,
            typename NumItemsT   = uint32_t,
            typename EnvT        = // Doxygen cannot resolve ::cuda::std::execution::env
#ifdef _CCCL_DOXYGEN_INVOKED
            void,
#else
            ::cuda::std::execution::env<>,
#endif
            ::cuda::std::enable_if_t<
              !::cuda::std::is_same_v<KeysInputIteratorT, void*> && !::cuda::std::is_null_pointer_v<KeysInputIteratorT>
                && !::cuda::std::is_same_v<ValuesInputIteratorT, size_t>,
              int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t InclusiveScanByKey(
    KeysInputIteratorT d_keys_in,
    ValuesInputIteratorT d_values_in,
    ValuesOutputIteratorT d_values_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    EqualityOpT equality_op = EqualityOpT(),
    EnvT env                = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceScan::InclusiveScanByKey");

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return scan_by_key_impl<tuning_t>(
        storage, bytes, d_keys_in, d_values_in, d_values_out, equality_op, scan_op, NullType{}, num_items, stream);
    });
  }

  //! @}
};

CUB_NAMESPACE_END
