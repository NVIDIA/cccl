// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! cub::DeviceSelect provides device-wide, parallel operations for compacting selected items from sequences of data
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
#include <cub/detail/temporary_storage.cuh>
#include <cub/device/device_reduce.cuh> // lift it to a top common header
#include <cub/device/dispatch/dispatch_select_if.cuh>
#include <cub/device/dispatch/dispatch_unique_by_key.cuh>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace select
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
  using fn = policy_hub<InputT, FlagT, OffsetT, DistinctPartitions, Impl>;
};
} // namespace select
} // namespace detail

//! @rst
//! DeviceSelect provides device-wide, parallel operations for compacting selected items from sequences of data items
//! residing within device-accessible memory. It is similar to DevicePartition, except that non-selected items are
//! discarded, whereas DevicePartition retains them.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! These operations apply a selection criterion to selectively copy
//! items from a specified input sequence to a compact output sequence.
//!
//! Usage Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @cdp_class{DeviceSelect}
//!
//! Performance
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @linear_performance{select-flagged, select-if, and select-unique}
//!
//! @endrst
struct DeviceSelect
{
private:
  template <typename TuningEnvT,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOpT_Functor,
            typename EqualityOpT,
            typename OffsetT,
            SelectImpl SelectionMode,
            ::cuda::execution::determinism::__determinism_t Determinism>
  CUB_RUNTIME_FUNCTION static cudaError_t select_impl(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    OffsetT num_items,
    SelectOpT_Functor user_select_op_instance,
    ::cuda::execution::determinism::__determinism_holder_t<Determinism> determinism_holder_arg,
    cudaStream_t stream)
  {
    using select_tuning_t = ::cuda::std::execution::
      __query_result_or_t<TuningEnvT, detail::select::get_tuning_query_t, detail::select::default_tuning>;

    using policy_t =
      typename select_tuning_t::template fn<detail::it_value_t<InputIteratorT>, NullType, OffsetT, false, SelectionMode>;

    using dispatch_t =
      DispatchSelectIf<InputIteratorT,
                       NullType*,
                       OutputIteratorT,
                       NumSelectedIteratorT,
                       SelectOpT_Functor,
                       NullType,
                       OffsetT,
                       SelectionMode,
                       policy_t>;

    return dispatch_t::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      nullptr,
      d_out,
      d_num_selected_out,
      user_select_op_instance,
      EqualityOpT{},
      num_items,
      stream);
  }

  //! @rst
  //! Uses the ``d_flags`` sequence to selectively copy the corresponding items from ``d_in`` into ``d_out``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The value type of ``d_flags`` must be castable to ``bool`` (e.g., ``bool``, ``char``, ``int``, etc.).
  //! - Copies of the selected items are compacted into ``d_out`` and maintain their original relative ordering.
  //! - | The range ``[d_out, d_out + *d_num_selected_out)`` shall not overlap ``[d_in, d_in + num_items)``,
  //!   | ``[d_flags, d_flags + num_items)`` nor ``d_num_selected_out`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>  // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for input,
  //!    // flags, and output
  //!    int  num_items;              // e.g., 8
  //!    int  *d_in;                  // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  //!    char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
  //!    int  *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_num_selected_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::Flagged(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_flags, d_out, d_num_selected_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::Flagged(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_flags, d_out, d_num_selected_out, num_items);
  //!
  //!    // d_out                 <-- [1, 4, 6, 7]
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
  //!   **[inferred]** Random-access output iterator type for writing selected items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
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
  //!   Pointer to the output sequence of selected data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected (i.e., length of `d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename FlagIterator, typename OutputIteratorT, typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Flagged(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagIterator d_flags,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    ::cuda::std::int64_t num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::Flagged");

    using OffsetT    = ::cuda::std::int64_t; // Signed integer type for global offsets
    using SelectOp   = NullType; // Selection op (not used)
    using EqualityOp = NullType; // Equality operator (not used)

    return DispatchSelectIf<
      InputIteratorT,
      FlagIterator,
      OutputIteratorT,
      NumSelectedIteratorT,
      SelectOp,
      EqualityOp,
      OffsetT,
      SelectImpl::Select>::Dispatch(d_temp_storage,
                                    temp_storage_bytes,
                                    d_in,
                                    d_flags,
                                    d_out,
                                    d_num_selected_out,
                                    SelectOp(),
                                    EqualityOp(),
                                    num_items,
                                    stream);
  }

public:
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  If(InputIteratorT d_in,
     OutputIteratorT d_out,
     NumSelectedIteratorT d_num_selected_out,
     ::cuda::std::int64_t num_items,
     SelectOp select_op,
     EnvT env = EnvT{})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceSelect::If");

    static_assert(!_CUDA_STD_EXEC::__queryable_with<EnvT, _CUDA_EXEC::determinism::__get_determinism_t>,
                  "Determinism should be used inside requires to have an effect.");
    using requirements_t =
      _CUDA_STD_EXEC::__query_result_or_t<EnvT, _CUDA_EXEC::__get_requirements_t, _CUDA_STD_EXEC::env<>>;

    // This `determinism_concrete_type` is the actual enum class type (e.g., run_to_run_t)
    using determinism_concrete_type =
      _CUDA_STD_EXEC::__query_result_or_t<requirements_t, //
                                          _CUDA_EXEC::determinism::__get_determinism_t,
                                          _CUDA_EXEC::determinism::run_to_run_t>;

    // Query relevant properties from the environment
    auto stream = _CUDA_STD_EXEC::__query_or(env, ::cuda::get_stream, ::cuda::stream_ref{});
    auto mr     = _CUDA_STD_EXEC::__query_or(env, ::cuda::mr::__get_memory_resource, detail::device_memory_resource{});

    void* d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    using tuning_t = _CUDA_STD_EXEC::__query_result_or_t<EnvT, _CUDA_EXEC::__get_tuning_t, _CUDA_STD_EXEC::env<>>;

    using OffsetT      = ::cuda::std::int64_t; // Signed integer type for global offsets
    using FlagIterator = NullType*; // FlagT iterator type (not used)
    using EqualityOp   = NullType; // Equality operator (not used)

    // Construct the actual determinism holder object to pass as a function argument.
    // The `operator()` of `__get_determinism_t` returns a `__determinism_holder_t` by value.
    // So, `_CUDA_STD_EXEC::__query_or(env, ::cuda::get_determinism, ...) ` would yield `__determinism_holder_t<VAL>`.
    // However, the `select_impl` expects a *bare* `__determinism_holder_t<VAL>` for its argument.
    // The `determinism_concrete_type{}` syntax default-constructs `run_to_run_t`, which IS a
    // `__determinism_holder_t<__determinism_t::__run_to_run>`. So, simply passing `determinism_concrete_type{}` should
    // work for the function argument.

    cudaError_t error =
      select_impl<tuning_t,
                  InputIteratorT,
                  OutputIteratorT,
                  NumSelectedIteratorT,
                  SelectOp,
                  EqualityOp,
                  ::cuda::std::int64_t,
                  SelectImpl::Select,
                  determinism_concrete_type::value>(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        d_num_selected_out,
        num_items,
        select_op,
        determinism_concrete_type{},
        static_cast<cudaStream_t>(stream.get()));

    error = CubDebug(detail::temporary_storage::allocate_async(d_temp_storage, temp_storage_bytes, mr, stream));
    if (error != cudaSuccess)
    {
      return error;
    }

    error = select_impl<tuning_t,
                        InputIteratorT,
                        OutputIteratorT,
                        NumSelectedIteratorT,
                        SelectOp,
                        EqualityOp,
                        ::cuda::std::int64_t,
                        SelectImpl::Select,
                        determinism_concrete_type::value>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      d_num_selected_out,
      num_items,
      select_op,
      determinism_concrete_type{},
      static_cast<cudaStream_t>(stream.get()));

    // Try to deallocate regardless of the error to avoid memory leaks
    cudaError_t deallocate_error =
      CubDebug(detail::temporary_storage::deallocate_async(d_temp_storage, temp_storage_bytes, mr, stream));

    if (error != cudaSuccess)
    {
      // Reduction error takes precedence over deallocation error since it happens first
      return error;
    }

    return deallocate_error;
  }

  //! @rst
  //! Uses the ``d_flags`` sequence to selectively compact the items in `d_data``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The value type of ``d_flags`` must be castable to ``bool`` (e.g., ``bool``, ``char``, ``int``, etc.).
  //! - Copies of the selected items are compacted in-place and maintain their original relative ordering.
  //! - | The ``d_data`` may equal ``d_flags``. The range ``[d_data, d_data + num_items)`` shall not overlap
  //!   | ``[d_flags, d_flags + num_items)`` in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>  // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for input,
  //!    // flags, and output
  //!    int  num_items;              // e.g., 8
  //!    int  *d_data;                // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  //!    char *d_flags;               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]
  //!    int  *d_num_selected_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::Flagged(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_flags, d_num_selected_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::Flagged(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_flags, d_num_selected_out, num_items);
  //!
  //!    // d_data                <-- [1, 4, 6, 7]
  //!    // d_num_selected_out    <-- [4]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing selected items @iterator
  //!
  //! @tparam FlagIterator
  //!   **[inferred]** Random-access input iterator type for reading selection flags @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
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
  //! @param[in] d_flags
  //!   Pointer to the input sequence of selection flags
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_data`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename FlagIterator, typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Flagged(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    FlagIterator d_flags,
    NumSelectedIteratorT d_num_selected_out,
    ::cuda::std::int64_t num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::Flagged");

    using OffsetT    = ::cuda::std::int64_t; // Signed integer type for global offsets
    using SelectOp   = NullType; // Selection op (not used)
    using EqualityOp = NullType; // Equality operator (not used)

    return DispatchSelectIf<IteratorT,
                            FlagIterator,
                            IteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            SelectImpl::SelectPotentiallyInPlace>::
      Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_data, // in
        d_flags,
        d_data, // out
        d_num_selected_out,
        SelectOp(),
        EqualityOp(),
        num_items,
        stream);
  }

  //! @rst
  //! Uses the ``select_op`` functor to selectively copy items from ``d_in`` into ``d_out``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering.
  //! - | The range ``[d_out, d_out + *d_num_selected_out)`` shall not overlap
  //!   | ``[d_in, d_in + num_items)`` nor ``d_num_selected_out`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Functor type for selecting values less than some criteria
  //!    struct LessThan
  //!    {
  //!        int compare;
  //!
  //!        __host__ __device__ __forceinline__
  //!        LessThan(int compare) : compare(compare) {}
  //!
  //!        __host__ __device__ __forceinline__
  //!        bool operator()(const int &a) const {
  //!            return (a < compare);
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int      num_items;              // e.g., 8
  //!    int      *d_in;                  // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
  //!    int      *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int      *d_num_selected_out;    // e.g., [ ]
  //!    LessThan select_op(7);
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::If(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_num_selected_out, num_items, select_op);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::If(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_num_selected_out, num_items, select_op);
  //!
  //!    // d_out                 <-- [0, 2, 3, 5, 2]
  //!    // d_num_selected_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing selected items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam SelectOp
  //!   **[inferred]** Selection operator type having member `bool operator()(const T &a)`
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
  //!   Pointer to the output sequence of selected data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected
  //!   (i.e., length of `d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] select_op
  //!   Unary selection operator
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT, typename SelectOp>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  If(void* d_temp_storage,
     size_t& temp_storage_bytes,
     InputIteratorT d_in,
     OutputIteratorT d_out,
     NumSelectedIteratorT d_num_selected_out,
     ::cuda::std::int64_t num_items,
     SelectOp select_op,
     cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::If");

    using OffsetT      = ::cuda::std::int64_t; // Signed integer type for global offsets
    using FlagIterator = NullType*; // FlagT iterator type (not used)
    using EqualityOp   = NullType; // Equality operator (not used)

    return DispatchSelectIf<
      InputIteratorT,
      FlagIterator,
      OutputIteratorT,
      NumSelectedIteratorT,
      SelectOp,
      EqualityOp,
      OffsetT,
      SelectImpl::Select>::Dispatch(d_temp_storage,
                                    temp_storage_bytes,
                                    d_in,
                                    nullptr,
                                    d_out,
                                    d_num_selected_out,
                                    select_op,
                                    EqualityOp(),
                                    num_items,
                                    stream);
  }

  //! @rst
  //! Uses the ``select_op`` functor to selectively compact items in ``d_data``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - | Copies of the selected items are compacted in ``d_data`` and maintain
  //!   | their original relative ordering.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Functor type for selecting values less than some criteria
  //!    struct LessThan
  //!    {
  //!        int compare;
  //!
  //!        __host__ __device__ __forceinline__
  //!        LessThan(int compare) : compare(compare) {}
  //!
  //!        __host__ __device__ __forceinline__
  //!        bool operator()(const int &a) const {
  //!            return (a < compare);
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int      num_items;              // e.g., 8
  //!    int      *d_data;                // e.g., [0, 2, 3, 9, 5, 2, 81, 8]
  //!    int      *d_num_selected_out;    // e.g., [ ]
  //!    LessThan select_op(7);
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::If(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, d_num_selected_out, num_items, select_op);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::If(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, d_num_selected_out, num_items, select_op);
  //!
  //!    // d_data                <-- [0, 2, 3, 5, 2]
  //!    // d_num_selected_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access input iterator type for reading and writing items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam SelectOp
  //!   **[inferred]** Selection operator type having member `bool operator()(const T &a)`
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
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_data`)
  //!
  //! @param[in] select_op
  //!   Unary selection operator
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename NumSelectedIteratorT, typename SelectOp>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  If(void* d_temp_storage,
     size_t& temp_storage_bytes,
     IteratorT d_data,
     NumSelectedIteratorT d_num_selected_out,
     ::cuda::std::int64_t num_items,
     SelectOp select_op,
     cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::If");

    using OffsetT      = ::cuda::std::int64_t; // Signed integer type for global offsets
    using FlagIterator = NullType*; // FlagT iterator type (not used)
    using EqualityOp   = NullType; // Equality operator (not used)

    return DispatchSelectIf<IteratorT,
                            FlagIterator,
                            IteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            SelectImpl::SelectPotentiallyInPlace>::
      Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_data, // in
        nullptr,
        d_data, // out
        d_num_selected_out,
        select_op,
        EqualityOp(),
        num_items,
        stream);
  }

  //! @rst
  //! Uses the ``select_op`` functor applied to ``d_flags`` to selectively copy the
  //! corresponding items from ``d_in`` into ``d_out``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The expression ``select_op(flag)`` must be convertible to ``bool``,
  //!   where the type of ``flag`` corresponds to the value type of ``FlagIterator``.
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering.
  //! - | The range ``[d_out, d_out + *d_num_selected_out)`` shall not overlap
  //!   | ``[d_in, d_in + num_items)`` nor ``d_num_selected_out`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_select_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-select-iseven
  //!     :end-before: example-end segmented-select-iseven
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_select_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-select-flaggedif
  //!     :end-before: example-end segmented-select-flaggedif
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
  //!   **[inferred]** Random-access output iterator type for writing selected items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam SelectOp
  //!   **[inferred]** Selection operator type having member `bool operator()(const T &a)`
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
  //!   Pointer to the output sequence of selected data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected
  //!   (i.e., length of `d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] select_op
  //!   Unary selection operator
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename FlagIterator,
            typename OutputIteratorT,
            typename NumSelectedIteratorT,
            typename SelectOp>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t FlaggedIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    FlagIterator d_flags,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    ::cuda::std::int64_t num_items,
    SelectOp select_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::FlaggedIf");

    using OffsetT    = ::cuda::std::int64_t; // Signed integer type for global offsets
    using EqualityOp = NullType; // Equality operator (not used)

    return DispatchSelectIf<
      InputIteratorT,
      FlagIterator,
      OutputIteratorT,
      NumSelectedIteratorT,
      SelectOp,
      EqualityOp,
      OffsetT,
      SelectImpl::Select>::Dispatch(d_temp_storage,
                                    temp_storage_bytes,
                                    d_in,
                                    d_flags,
                                    d_out,
                                    d_num_selected_out,
                                    select_op,
                                    EqualityOp(),
                                    num_items,
                                    stream);
  }

  //! @rst
  //! Uses the ``select_op`` functor applied to ``d_flags`` to selectively compact the
  //! corresponding items in ``d_data``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The expression ``select_op(flag)`` must be convertible to ``bool``,
  //!   where the type of ``flag`` corresponds to the value type of ``FlagIterator``.
  //! - Copies of the selected items are compacted in-place and maintain their original relative ordering.
  //! - | The ``d_data`` may equal ``d_flags``. The range ``[d_data, d_data + num_items)`` shall not overlap
  //!   | ``[d_flags, d_flags + num_items)`` in any other way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_select_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-select-iseven
  //!     :end-before: example-end segmented-select-iseven
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_select_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin segmented-select-flaggedif-inplace
  //!     :end-before: example-end segmented-select-flaggedif-inplace
  //!
  //! @endrst
  //!
  //! @tparam IteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing selected items @iterator
  //!
  //! @tparam FlagIterator
  //!   **[inferred]** Random-access input iterator type for reading selection flags @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam SelectOp
  //!   **[inferred]** Selection operator type having member `bool operator()(const T &a)`
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
  //! @param[in] d_flags
  //!   Pointer to the input sequence of selection flags
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_data`)
  //!
  //! @param[in] select_op
  //!   Unary selection operator
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename IteratorT, typename FlagIterator, typename NumSelectedIteratorT, typename SelectOp>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t FlaggedIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    IteratorT d_data,
    FlagIterator d_flags,
    NumSelectedIteratorT d_num_selected_out,
    ::cuda::std::int64_t num_items,
    SelectOp select_op,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::FlaggedIf");

    using OffsetT    = ::cuda::std::int64_t; // Signed integer type for global offsets
    using EqualityOp = NullType; // Equality operator (not used)

    return DispatchSelectIf<IteratorT,
                            FlagIterator,
                            IteratorT,
                            NumSelectedIteratorT,
                            SelectOp,
                            EqualityOp,
                            OffsetT,
                            SelectImpl::SelectPotentiallyInPlace>::
      Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_data, // in
        d_flags,
        d_data, // out
        d_num_selected_out,
        select_op,
        EqualityOp(),
        num_items,
        stream);
  }

  //! @rst
  //! Given an input sequence ``d_in`` having runs of consecutive equal-valued keys,
  //! only the first key from each run is selectively copied to ``d_out``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The ``==`` equality operator is used to determine whether keys are equivalent
  //! - Copies of the selected items are compacted into ``d_out`` and maintain their original relative ordering.
  //! - | The range ``[d_out, d_out + *d_num_selected_out)`` shall not overlap
  //!   | ``[d_in, d_in + num_items)`` nor ``d_num_selected_out`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;              // e.g., 8
  //!    int  *d_in;                  // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int  *d_out;                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_num_selected_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::Unique(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_num_selected_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::Unique(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, d_num_selected_out, num_items);
  //!
  //!    // d_out                 <-- [0, 2, 9, 5, 8]
  //!    // d_num_selected_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing selected items @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
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
  //!   Pointer to the output sequence of selected data items
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the output total number of items selected
  //!   (i.e., length of `d_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Unique(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumSelectedIteratorT d_num_selected_out,
    ::cuda::std::int64_t num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::Unique");

    using OffsetT      = ::cuda::std::int64_t;
    using FlagIterator = NullType*; // FlagT iterator type (not used)
    using SelectOp     = NullType; // Selection op (not used)
    using EqualityOp   = ::cuda::std::equal_to<>; // Default == operator

    return DispatchSelectIf<
      InputIteratorT,
      FlagIterator,
      OutputIteratorT,
      NumSelectedIteratorT,
      SelectOp,
      EqualityOp,
      OffsetT,
      SelectImpl::Select>::Dispatch(d_temp_storage,
                                    temp_storage_bytes,
                                    d_in,
                                    nullptr,
                                    d_out,
                                    d_num_selected_out,
                                    SelectOp(),
                                    EqualityOp(),
                                    num_items,
                                    stream);
  }

  //! @rst
  //! Given an input sequence ``d_keys_in`` and ``d_values_in`` with runs of key-value pairs with consecutive
  //! equal-valued keys, only the first key and its value from each run is selectively copied
  //! to ``d_keys_out`` and ``d_values_out``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The user-provided equality operator, `equality_op`, is used to determine whether keys are equivalent
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering.
  //! - In-place operations are not supported. There must be no overlap between
  //!   any of the provided ranges:
  //!
  //!   - ``[d_keys_in,          d_keys_in    + num_items)``
  //!   - ``[d_keys_out,         d_keys_out   + *d_num_selected_out)``
  //!   - ``[d_values_in,        d_values_in  + num_items)``
  //!   - ``[d_values_out,       d_values_out + *d_num_selected_out)``
  //!   - ``[d_num_selected_out, d_num_selected_out + 1)``
  //!
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;              // e.g., 8
  //!    int  *d_keys_in;             // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int  *d_values_in;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  //!    int  *d_keys_out;            // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_values_out;          // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_num_selected_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::UniqueByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in,
  //!      d_keys_out, d_values_out, d_num_selected_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::UniqueByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in,
  //!      d_keys_out, d_values_out, d_num_selected_out, num_items);
  //!
  //!    // d_keys_out            <-- [0, 2, 9, 5, 8]
  //!    // d_values_out          <-- [1, 2, 4, 5, 8]
  //!    // d_num_selected_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam ValueInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing selected keys @iterator
  //!
  //! @tparam ValueOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing selected values @iterator
  //!
  //! @tparam NumSelectedIteratorT
  //!   **[inferred]** Output iterator type for recording the number of items selected @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EqualityOpT
  //!   **[inferred]** Type of equality_op
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input sequence of keys
  //!
  //! @param[in] d_values_in
  //!   Pointer to the input sequence of values
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the output sequence of selected keys
  //!
  //! @param[out] d_values_out
  //!   Pointer to the output sequence of selected values
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the total number of items selected (i.e., length of `d_keys_out` or `d_values_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_keys_in` or `d_values_in`)
  //!
  //! @param[in] equality_op
  //!   Binary predicate to determine equality
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueOutputIteratorT,
            typename NumSelectedIteratorT,
            typename NumItemsT,
            typename EqualityOpT>
  CUB_RUNTIME_FUNCTION __forceinline__ static //
    ::cuda::std::enable_if_t< //
      !::cuda::std::is_convertible_v<EqualityOpT, cudaStream_t>, //
      cudaError_t>
    UniqueByKey(
      void* d_temp_storage,
      size_t& temp_storage_bytes,
      KeyInputIteratorT d_keys_in,
      ValueInputIteratorT d_values_in,
      KeyOutputIteratorT d_keys_out,
      ValueOutputIteratorT d_values_out,
      NumSelectedIteratorT d_num_selected_out,
      NumItemsT num_items,
      EqualityOpT equality_op,
      cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceSelect::UniqueByKey");

    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchUniqueByKey<
      KeyInputIteratorT,
      ValueInputIteratorT,
      KeyOutputIteratorT,
      ValueOutputIteratorT,
      NumSelectedIteratorT,
      EqualityOpT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_values_in,
                         d_keys_out,
                         d_values_out,
                         d_num_selected_out,
                         equality_op,
                         static_cast<OffsetT>(num_items),
                         stream);
  }

  //! @rst
  //! Given an input sequence ``d_keys_in`` and ``d_values_in`` with runs of key-value pairs with consecutive
  //! equal-valued keys, only the first key and its value from each run is selectively copied
  //! to ``d_keys_out`` and ``d_values_out``.
  //! The total number of items selected is written to ``d_num_selected_out``.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! - The ``==`` equality operator is used to determine whether keys are equivalent
  //! - Copies of the selected items are compacted into ``d_out`` and maintain
  //!   their original relative ordering.
  //! - In-place operations are not supported. There must be no overlap between
  //!   any of the provided ranges:
  //!
  //!   - ``[d_keys_in,          d_keys_in    + num_items)``
  //!   - ``[d_keys_out,         d_keys_out   + *d_num_selected_out)``
  //!   - ``[d_values_in,        d_values_in  + num_items)``
  //!   - ``[d_values_out,       d_values_out + *d_num_selected_out)``
  //!   - ``[d_num_selected_out, d_num_selected_out + 1)``
  //!
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the compaction of items selected from an ``int`` device vector.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>   // or equivalently <cub/device/device_select.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;              // e.g., 8
  //!    int  *d_keys_in;             // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int  *d_values_in;           // e.g., [1, 2, 3, 4, 5, 6, 7, 8]
  //!    int  *d_keys_out;            // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_values_out;          // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
  //!    int  *d_num_selected_out;    // e.g., [ ]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceSelect::UniqueByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in,
  //!      d_keys_out, d_values_out, d_num_selected_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run selection
  //!    cub::DeviceSelect::UniqueByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_values_in,
  //!      d_keys_out, d_values_out, d_num_selected_out, num_items);
  //!
  //!    // d_keys_out            <-- [0, 2, 9, 5, 8]
  //!    // d_values_out          <-- [1, 2, 4, 5, 8]
  //!    // d_num_selected_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam KeyInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam ValueInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam KeyOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing selected keys @iterator
  //!
  //! @tparam ValueOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing selected values @iterator
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
  //! @param[in] d_keys_in
  //!   Pointer to the input sequence of keys
  //!
  //! @param[in] d_values_in
  //!   Pointer to the input sequence of values
  //!
  //! @param[out] d_keys_out
  //!   Pointer to the output sequence of selected keys
  //!
  //! @param[out] d_values_out
  //!   Pointer to the output sequence of selected values
  //!
  //! @param[out] d_num_selected_out
  //!   Pointer to the total number of items selected (i.e., length of `d_keys_out` or `d_values_out`)
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_keys_in` or `d_values_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeyInputIteratorT,
            typename ValueInputIteratorT,
            typename KeyOutputIteratorT,
            typename ValueOutputIteratorT,
            typename NumSelectedIteratorT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION __forceinline__ static cudaError_t UniqueByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeyInputIteratorT d_keys_in,
    ValueInputIteratorT d_values_in,
    KeyOutputIteratorT d_keys_out,
    ValueOutputIteratorT d_values_out,
    NumSelectedIteratorT d_num_selected_out,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    return UniqueByKey(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_values_in,
      d_keys_out,
      d_values_out,
      d_num_selected_out,
      num_items,
      ::cuda::std::equal_to<>{},
      stream);
  }
};

CUB_NAMESPACE_END
