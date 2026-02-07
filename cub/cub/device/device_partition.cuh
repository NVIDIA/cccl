// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! cub::DevicePartition provides device-wide, parallel operations for partitioning sequences of data items residing
//! within device-accessible memory.

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
#include <cub/detail/env_dispatch.cuh>
#include <cub/device/dispatch/dispatch_select_if.cuh>
#include <cub/device/dispatch/dispatch_three_way_partition.cuh>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::partition
{
struct get_tuning_query_t
{};

template <class Derived>
struct tuning
{
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto query(const get_tuning_query_t&) const noexcept -> Derived
  {
    return static_cast<const Derived&>(*this);
  }
};

struct default_tuning : tuning<default_tuning>
{
  template <class InputT, class FlagT, class OffsetT, bool DistinctPartitions, SelectImpl Impl>
  using fn = detail::select::policy_hub<InputT, FlagT, OffsetT, DistinctPartitions, Impl>;
};
} // namespace detail::partition

//! @rst
//! DevicePartition provides device-wide, parallel operations for
//! partitioning sequences of data items residing within device-accessible memory.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! These operations apply a selection criterion to construct a partitioned
//! output sequence from items selected/unselected from a specified input
//! sequence.
//!
//! Usage Considerations
//! ++++++++++++++++++++++++++
//!
//! @cdp_class{DevicePartition}
//!
//! Performance
//! ++++++++++++++++++++++++++
//!
//! @linear_performance{partition}
//!
//! @endrst
struct DevicePartition
{
private:
  template <typename TuningEnvT,
            typename InputIteratorT,
            typename FlagIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOpT,
            typename OffsetT,
            ::cuda::execution::determinism::__determinism_t Determinism>
  CUB_RUNTIME_FUNCTION static cudaError_t partition_impl(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagIteratorT d_flags,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    OffsetT num_items,
    SelectOpT select_op,
    ::cuda::execution::determinism::__determinism_holder_t<Determinism> determinism_holder_arg,
    cudaStream_t stream)
  {
    (void) determinism_holder_arg; // determinism is of no use in DevicePartition at the moment
    using partition_tuning_t = ::cuda::std::execution::
      __query_result_or_t<TuningEnvT, detail::partition::get_tuning_query_t, detail::partition::default_tuning>;

    using flag_t = detail::it_value_t<FlagIteratorT>;

    using policy_t = typename partition_tuning_t::
      template fn<detail::it_value_t<InputIteratorT>, flag_t, OffsetT, true, SelectImpl::Partition>;

    using EqualityOp = NullType;

    using dispatch_t =
      DispatchSelectIf<InputIteratorT,
                       FlagIteratorT,
                       OutputIteratorT,
                       NumSelectedIteratorT,
                       SelectOpT,
                       EqualityOp,
                       OffsetT,
                       SelectImpl::Partition,
                       policy_t>;

    return dispatch_t::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_flags,
      d_out,
      d_num_selected_out,
      select_op,
      EqualityOp{},
      num_items,
      stream);
  }

public:
  //! @rst
  //! Uses the ``d_flags`` sequence to split the corresponding items from
  //! ``d_in`` into a partitioned sequence ``d_out``.
  //! The total number of items copied into the first partition is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The value type of ``d_flags`` must be castable to ``bool`` (e.g., ``bool``, ``char``, ``int``, etc.).
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering, however copies of the unselected
  //!   items are compacted into the rear of ``d_out`` in reverse order.
  //! - The range ``[d_out, d_out + num_items)`` shall not overlap
  //!   ``[d_in, d_in + num_items)`` nor ``[d_flags, d_flags + num_items)`` in any way.
  //!   The range ``[d_in, d_in + num_items)`` may overlap ``[d_flags, d_flags + num_items)``.
  //! - @devicestorage
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_partition.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input, flags, and output
  //!    int  num_items;              // e.g., 8
  //!    int  *d_in;                  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  //!    char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
  //!    int  *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_num_selected_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DevicePartition::Flagged(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_flags, d_out, d_num_selected_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DevicePartition::Flagged(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_flags, d_out, d_num_selected_out, num_items);
  //!
  //!    // d_out                 <-- [1, 4, 6, 7, 8, 5, 3, 2]
  //!    // d_num_selected_out    <-- [4]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam FlagIterator
  //!   **[inferred]** Random-access input iterator type for reading selection flags @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
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
  //! @param[in] d_flags
  //!   Pointer to the input sequence of selection flags
  //!
  //! @param[out] d_out
  //!   Pointer to the output sequence of partitioned data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected (i.e., the
  //!   offset of the unselected partition)
  //!
  //! @param[in] num_items
  //!   Total number of items to select from
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename FlagIterator,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Flagged(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagIterator d_flags,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DevicePartition::Flagged");
    using ChooseOffsetT = detail::choose_signed_offset<NumItemsT>;
    using OffsetT       = typename ChooseOffsetT::type; // Signed integer type for global offsets
    using SelectOp      = NullType; // Selection op (not used)
    using EqualityOp    = NullType; // Equality operator (not used)
    using DispatchSelectIfT =
      DispatchSelectIf<InputIteratorT,
                       FlagIterator,
                       OutputIteratorT,
                       NumSelectedIteratorT,
                       SelectOp,
                       EqualityOp,
                       OffsetT,
                       SelectImpl::Partition>;

    // Check if the number of items exceeds the range covered by the selected signed offset type
    cudaError_t error = ChooseOffsetT::is_exceeding_offset_type(num_items);
    if (error)
    {
      return error;
    }

    return DispatchSelectIfT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_flags,
      d_out,
      d_num_selected_out,
      SelectOp{},
      EqualityOp{},
      num_items,
      stream);
  }

  //! @rst
  //! Uses the ``d_flags`` sequence to split the corresponding items from
  //! ``d_in`` into a partitioned sequence ``d_out``.
  //! The total number of items copied into the first partition is written to ``d_num_selected_out``.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - The value type of ``d_flags`` must be castable to ``bool`` (e.g., ``bool``, ``char``, ``int``, etc.).
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering, however copies of the unselected
  //!   items are compacted into the rear of ``d_out`` in reverse order.
  //! - The range ``[d_out, d_out + num_items)`` shall not overlap
  //!   ``[d_in, d_in + num_items)`` nor ``[d_flags, d_flags + num_items)`` in any way.
  //!   The range ``[d_in, d_in + num_items)`` may overlap ``[d_flags, d_flags + num_items)``.
  //!
  //! Determinism
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! DevicePartition is inherently ``gpu_to_gpu`` deterministic because it uses integer prefix sums,
  //! which are truly associative. The stability and determinism guarantees hold provided that:
  //!
  //! - The ``d_flags`` sequence produces the same values when read multiple times during the operation.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the partitioning of flagged items from an ``int`` device vector
  //! using determinism requirements:
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_partition_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin partition-flagged-env-determinism
  //!     :end-before: example-end partition-flagged-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam FlagIterator
  //!   **[inferred]** Random-access input iterator type for reading selection flags @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., `cuda::std::execution::env<...>`)
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[in] d_flags
  //!   Pointer to the input sequence of selection flags
  //!
  //! @param[out] d_out
  //!   Pointer to the output sequence of partitioned data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected (i.e., the
  //!   offset of the unselected partition)
  //!
  //! @param[in] num_items
  //!   Total number of items to select from
  //!
  //! @param[in] env
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename FlagIterator,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>,
            typename ::cuda::std::enable_if_t<
              ::cuda::std::is_integral_v<NumItemsT> && !::cuda::std::is_same_v<InputIteratorT, void*>
                && !::cuda::std::is_same_v<FlagIterator, size_t&>,
              int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Flagged(
    InputIteratorT d_in,
    FlagIterator d_flags,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    NumItemsT num_items,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DevicePartition::Flagged");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");

    using offset_t = detail::choose_offset_t<NumItemsT>;

    // Extract determinism from environment, defaulting to run_to_run
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::gpu_to_gpu_t>;

    using determinism_t = ::cuda::execution::determinism::gpu_to_gpu_t;

    // Dispatch with environment - handles all boilerplate
    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return partition_impl<tuning_t,
                            InputIteratorT,
                            FlagIterator,
                            OutputIteratorT,
                            NumSelectedIteratorT,
                            NullType,
                            offset_t,
                            determinism_t::value>(
        storage,
        bytes,
        d_in,
        d_flags,
        d_out,
        d_num_selected_out,
        static_cast<offset_t>(num_items),
        NullType{},
        determinism_t{},
        stream);
    });
  }

  //! @rst
  //! Uses the ``select_op`` functor to split the corresponding items from ``d_in`` into
  //! a partitioned sequence ``d_out``. The total number of items copied into the first partition is written
  //! to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering, however copies of the unselected
  //!   items are compacted into the rear of ``d_out`` in reverse order.
  //! - The range ``[d_out, d_out + num_items)`` shall not overlap
  //!   ``[d_in, d_in + num_items)`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_partition.cuh>
  //!
  //!    // Functor type for selecting values less than some criteria
  //!    struct LessThan
  //!    {
  //!        int compare;
  //!
  //!        CUB_RUNTIME_FUNCTION __forceinline__
  //!        explicit LessThan(int compare) : compare(compare) {}
  //!
  //!        CUB_RUNTIME_FUNCTION __forceinline__
  //!        bool operator()(const int &a) const
  //!        {
  //!            return (a < compare);
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int      num_items;              // e.g., 8
  //!    int      *d_in;                  // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
  //!    int      *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int      *d_num_selected_out;    // e.g., [ ]
  //!    LessThan select_op(7);
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DevicePartition::If(
  //!    d_temp_storage, temp_storage_bytes,
  //!    d_in, d_out, d_num_selected_out, num_items, select_op);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DevicePartition::If(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_num_selected_out, num_items, select_op);
  //!
  //!    // d_out                 <-- [0, 2, 3, 5, 2, 8, 81, 9]
  //!    // d_num_selected_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam SelectOp
  //!   **[inferred]** Selection functor type having member `bool operator()(const T &a)`
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
  //!   Pointer to the output sequence of partitioned data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected (i.e., the offset of the unselected partition)
  //!
  //! @param[in] num_items
  //!   Total number of items to select from
  //!
  //! @param[in] select_op
  //!   Unary selection operator
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  If(void* d_temp_storage,
     size_t& temp_storage_bytes,
     InputIteratorT d_in,
     OutputIteratorT d_out,
     NumSelectedIteratorT d_num_selected_out,
     NumItemsT num_items,
     SelectOp select_op,
     cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DevicePartition::If");
    using ChooseOffsetT = detail::choose_signed_offset<NumItemsT>;
    using OffsetT       = typename ChooseOffsetT::type; // Signed integer type for global offsets
    using FlagIterator  = NullType*; // FlagT iterator type (not used)
    using EqualityOp    = NullType; // Equality operator (not used)

    // Check if the number of items exceeds the range covered by the selected signed offset type
    cudaError_t error = ChooseOffsetT::is_exceeding_offset_type(num_items);
    if (error)
    {
      return error;
    }

    using DispatchSelectIfT =
      DispatchSelectIf<InputIteratorT,
                       FlagIterator,
                       OutputIteratorT,
                       NumSelectedIteratorT,
                       SelectOp,
                       EqualityOp,
                       OffsetT,
                       SelectImpl::Partition>;

    return DispatchSelectIfT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      nullptr,
      d_out,
      d_num_selected_out,
      select_op,
      EqualityOp{},
      num_items,
      stream);
  }

  //! @rst
  //! Uses the ``select_op`` functor to split the corresponding items from ``d_in`` into
  //! a partitioned sequence ``d_out``. The total number of items copied into the first partition is written
  //! to ``d_num_selected_out``.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering, however copies of the unselected
  //!   items are compacted into the rear of ``d_out`` in reverse order.
  //! - The range ``[d_out, d_out + num_items)`` shall not overlap
  //!   ``[d_in, d_in + num_items)`` in any way.
  //!
  //! Determinism
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! DevicePartition is inherently ``gpu_to_gpu`` deterministic because it uses integer prefix sums,
  //! which are truly associative. The stability and determinism guarantees hold provided that
  //! ``select_op`` is a **pure function**, meaning:
  //!
  //! 1. **Referentially transparent**: For the same input value, it always returns the same result.
  //! 2. **Side-effect free**: It has no observable side effects.
  //!
  //! Violations of purity that break guarantees include:
  //!
  //! - Reading thread-varying state (e.g., ``threadIdx``, ``clock()``, uninitialized memory)
  //! - Reading or writing shared mutable state (e.g., global variables, atomics)
  //! - Behavior that depends on evaluation order or timing
  //! - Any operation that causes undefined behavior
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the partitioning of items selected from an ``int`` device vector
  //! using determinism requirements:
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_partition_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin partition-if-env-determinism
  //!     :end-before: example-end partition-if-env-determinism
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam SelectOp
  //!   **[inferred]** Selection functor type having member `bool operator()(const T &a)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., `cuda::std::execution::env<...>`)
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output sequence of partitioned data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected (i.e., the offset of the unselected partition)
  //!
  //! @param[in] num_items
  //!   Total number of items to select from
  //!
  //! @param[in] select_op
  //!   Unary selection operator
  //!
  //! @param[in] env
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <
    typename InputIteratorT,
    typename OutputIteratorT,
    typename NumSelectedIteratorT,
    typename SelectOp,
    typename NumItemsT,
    typename EnvT = ::cuda::std::execution::env<>,
    typename ::cuda::std::
      enable_if_t<::cuda::std::is_integral_v<NumItemsT> && !::cuda::std::is_same_v<InputIteratorT, void*>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  If(InputIteratorT d_in,
     OutputIteratorT d_out,
     NumSelectedIteratorT d_num_selected_out,
     NumItemsT num_items,
     SelectOp select_op,
     EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DevicePartition::If");

    static_assert(!::cuda::std::execution::__queryable_with<EnvT, ::cuda::execution::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");

    using offset_t = detail::choose_offset_t<NumItemsT>;

    // Extract determinism from environment, defaulting to run_to_run
    using requirements_t = ::cuda::std::execution::
      __query_result_or_t<EnvT, ::cuda::execution::__get_requirements_t, ::cuda::std::execution::env<>>;
    using requested_determinism_t =
      ::cuda::std::execution::__query_result_or_t<requirements_t, //
                                                  ::cuda::execution::determinism::__get_determinism_t,
                                                  ::cuda::execution::determinism::gpu_to_gpu_t>;

    using determinism_t = ::cuda::execution::determinism::gpu_to_gpu_t;

    // Dispatch with environment - handles all boilerplate
    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      using tuning_t = decltype(tuning);
      return partition_impl<tuning_t,
                            InputIteratorT,
                            NullType*,
                            OutputIteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            offset_t,
                            determinism_t::value>(
        storage,
        bytes,
        d_in,
        nullptr,
        d_out,
        d_num_selected_out,
        static_cast<offset_t>(num_items),
        select_op,
        determinism_t{},
        stream);
    });
  }

private:
  template <SortOrder Order,
            typename KeyT,
            typename ValueT,
            typename OffsetT,
            typename BeginOffsetIteratorT,
            typename EndOffsetIteratorT,
            typename PolicyHub,
            typename KernelSource,
            typename KernelLauncherFactory,
            typename PartitionPolicyHub,
            typename PartitionKernelSource>
  friend class DispatchSegmentedSort;

  // Internal version without NVTX range
  template <typename InputIteratorT,
            typename FirstOutputIteratorT,
            typename SecondOutputIteratorT,
            typename UnselectedOutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectFirstPartOp,
            typename SelectSecondPartOp,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t IfNoNVTX(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FirstOutputIteratorT d_first_part_out,
    SecondOutputIteratorT d_second_part_out,
    UnselectedOutputIteratorT d_unselected_out,
    NumSelectedIteratorT d_num_selected_out,
    NumItemsT num_items,
    SelectFirstPartOp select_first_part_op,
    SelectSecondPartOp select_second_part_op,
    cudaStream_t stream = 0)
  {
    using ChooseOffsetT                = detail::choose_signed_offset<NumItemsT>;
    using OffsetT                      = typename ChooseOffsetT::type;
    using DispatchThreeWayPartitionIfT = DispatchThreeWayPartitionIf<
      InputIteratorT,
      FirstOutputIteratorT,
      SecondOutputIteratorT,
      UnselectedOutputIteratorT,
      NumSelectedIteratorT,
      SelectFirstPartOp,
      SelectSecondPartOp,
      OffsetT>;

    // Signed integer type for global offsets
    // Check if the number of items exceeds the range covered by the selected signed offset type
    cudaError_t error = ChooseOffsetT::is_exceeding_offset_type(num_items);
    if (error)
    {
      return error;
    }

    return DispatchThreeWayPartitionIfT::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_first_part_out,
      d_second_part_out,
      d_unselected_out,
      d_num_selected_out,
      select_first_part_op,
      select_second_part_op,
      num_items,
      stream);
  }

public:
  //! @rst
  //! Uses two functors to split the corresponding items from ``d_in`` into a three partitioned sequences
  //! ``d_first_part_out``, ``d_second_part_out``, and ``d_unselected_out``.
  //! The total number of items copied into the first partition is written
  //! to ``d_num_selected_out[0]``, while the total number of items copied into the second partition is written
  //! to ``d_num_selected_out[1]``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Copies of the items selected by ``select_first_part_op`` are compacted
  //!   into ``d_first_part_out`` and maintain their original relative ordering.
  //! - Copies of the items selected by ``select_second_part_op`` are compacted
  //!   into ``d_second_part_out`` and maintain their original relative ordering.
  //! - Copies of the unselected items are compacted into the ``d_unselected_out`` in reverse order.
  //! - The ranges ``[d_out, d_out + num_items)``,
  //!   ``[d_first_part_out, d_first_part_out + d_num_selected_out[0])``,
  //!   ``[d_second_part_out, d_second_part_out + d_num_selected_out[1])``,
  //!   ``[d_unselected_out, d_unselected_out + num_items - d_num_selected_out[0] - d_num_selected_out[1])``,
  //!   shall not overlap in any way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how this algorithm can partition an
  //! input vector into small, medium, and large items so that the relative
  //! order of items remain deterministic.
  //!
  //! Let's consider any value that doesn't exceed six a small one. On the
  //! other hand, any value that exceeds 50 will be considered a large one.
  //! Since the value used to define a small part doesn't match one that
  //! defines the large part, the intermediate segment is implied.
  //!
  //! These definitions partition a value space into three categories. We want
  //! to preserve the order of items in which they appear in the input vector.
  //! Since the algorithm provides stable partitioning, this is possible.
  //!
  //! Since the number of items in each category is unknown beforehand, we need
  //! three output arrays of num_items elements each. To reduce the memory
  //! requirements, we can combine the output storage for two categories.
  //!
  //! Since each value falls precisely in one category, it's safe to add
  //! "large" values into the head of the shared output vector and the "middle"
  //! values into its tail. To add items into the tail of the output array, we
  //! can use ``cuda::std::reverse_iterator``.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_partition.cuh>
  //!
  //!    // Functor type for selecting values less than some criteria
  //!    struct LessThan
  //!    {
  //!        int compare;
  //!
  //!        __host__ __device__ __forceinline__
  //!        explicit LessThan(int compare) : compare(compare) {}
  //!
  //!        __host__ __device__ __forceinline__
  //!        bool operator()(const int &a) const
  //!        {
  //!            return a < compare;
  //!        }
  //!    };
  //!
  //!    // Functor type for selecting values greater than some criteria
  //!    struct GreaterThan
  //!    {
  //!        int compare;
  //!
  //!        __host__ __device__ __forceinline__
  //!        explicit GreaterThan(int compare) : compare(compare) {}
  //!
  //!        __host__ __device__ __forceinline__
  //!        bool operator()(const int &a) const
  //!        {
  //!            return a > compare;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int      num_items;                   // e.g., 8
  //!    int      *d_in;                       // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
  //!    int      *d_large_and_unselected_out; // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int      *d_small_out;                // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int      *d_num_selected_out;         // e.g., [ , ]
  //!    cud::std::reverse_iterator<T> unselected_out(d_large_and_unselected_out + num_items);
  //!    LessThan small_items_selector(7);
  //!    GreaterThan large_items_selector(50);
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DevicePartition::If(
  //!         d_temp_storage, temp_storage_bytes,
  //!         d_in, d_large_and_medium_out, d_small_out, unselected_out,
  //!         d_num_selected_out, num_items,
  //!         large_items_selector, small_items_selector);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DevicePartition::If(
  //!         d_temp_storage, temp_storage_bytes,
  //!         d_in, d_large_and_medium_out, d_small_out, unselected_out,
  //!         d_num_selected_out, num_items,
  //!         large_items_selector, small_items_selector);
  //!
  //!    // d_large_and_unselected_out  <-- [ 81,  ,  ,  ,  ,  , 8, 9 ]
  //!    // d_small_out                 <-- [  0, 2, 3, 5, 2,  ,  ,   ]
  //!    // d_num_selected_out          <-- [  1, 5 ]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam FirstOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output
  //!   items selected by first operator @iterator
  //!
  //! @tparam SecondOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output
  //!   items selected by second operator @iterator
  //!
  //! @tparam UnselectedOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing
  //!   unselected items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items
  //!   selected @iterator
  //!
  //! @tparam SelectFirstPartOp
  //!   **[inferred]** Selection functor type having member `bool operator()(const T &a)`
  //!
  //! @tparam SelectSecondPartOp
  //!   **[inferred]** Selection functor type having member `bool operator()(const T &a)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
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
  //! @param[out] d_first_part_out
  //!   Pointer to the output sequence of data items selected by `select_first_part_op`
  //!
  //! @param[out] d_second_part_out
  //!   Pointer to the output sequence of data items selected by `select_second_part_op`
  //!
  //! @param[out] d_unselected_out
  //!   Pointer to the output sequence of unselected data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output array with two elements, where total number of
  //!   items selected by `select_first_part_op` is stored as
  //!   `d_num_selected_out[0]` and total number of items selected by
  //!   `select_second_part_op` is stored as `d_num_selected_out[1]`,
  //!   respectively
  //!
  //! @param[in] num_items
  //!   Total number of items to select from
  //!
  //! @param[in] select_first_part_op
  //!   Unary selection operator to select `d_first_part_out`
  //!
  //! @param[in] select_second_part_op
  //!   Unary selection operator to select `d_second_part_out`
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename FirstOutputIteratorT,
            typename SecondOutputIteratorT,
            typename UnselectedOutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectFirstPartOp,
            typename SelectSecondPartOp,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  If(void* d_temp_storage,
     size_t& temp_storage_bytes,
     InputIteratorT d_in,
     FirstOutputIteratorT d_first_part_out,
     SecondOutputIteratorT d_second_part_out,
     UnselectedOutputIteratorT d_unselected_out,
     NumSelectedIteratorT d_num_selected_out,
     NumItemsT num_items,
     SelectFirstPartOp select_first_part_op,
     SelectSecondPartOp select_second_part_op,
     cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DevicePartition::If");
    return IfNoNVTX(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_first_part_out,
      d_second_part_out,
      d_unselected_out,
      d_num_selected_out,
      num_items,
      select_first_part_op,
      select_second_part_op,
      stream);
  }
};

CUB_NAMESPACE_END
