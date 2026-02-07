// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/binary_search_helpers.cuh>
#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_for.cuh>
#include <cub/device/dispatch/dispatch_find.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__nvtx/nvtx.h>

CUB_NAMESPACE_BEGIN

struct DeviceFind
{
  //! @rst
  //! Finds the first element in the input sequence that satisfies the given predicate.
  //!
  //! - The search terminates at the first element where the predicate evaluates to true.
  //! - The index of the found element is written to ``d_out``.
  //! - If no element satisfies the predicate, ``num_items`` is written to ``d_out``.
  //! - The range ``[d_out, d_out + 1)`` shall not overlap ``[d_in, d_in + num_items)`` in any way.
  //! - @devicestorage
  //!
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! ==========================================================================
  //!
  //! The code snippet below illustrates the finding of the first element that satisfies the predicate.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_if_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin  find-if-predicate
  //!     :end-before: example-end  find-if-predicate
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_if_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin  device-find-if
  //!     :end-before: example-end  device-find-if
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing the result index @iterator
  //!
  //! @tparam ScanOpT
  //!   **[inferred]** Unary predicate functor type having member `bool operator()(const T &a)`
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
  //!   Random-access iterator to the output location for the index of the found element
  //!
  //! @param[in] scan_op
  //!   Unary predicate functor for determining whether an element satisfies the search condition
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., the length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t FindIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceFind::FindIf");

    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return detail::find::dispatch_t<InputIteratorT, OutputIteratorT, OffsetT, ScanOpT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<OffsetT>(num_items), scan_op, stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``, performs a binary search in the range
  //! ``[d_range, d_range + range_num_items)``, using ``comp`` as the comparator to find the iterator to the element
  //! of said range which **is not** ordered **before** ``value``.
  //!
  //! - The range ``[first, last)`` must be sorted consistently with ``comp``.
  //!
  //! .. versionadded:: 3.3.0
  //! @endrst
  //!
  //! @tparam RangeIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``ValuesIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam RangeNumItemsT
  //!   is an integral type representing the number of elements in the range to be searched.
  //!
  //! @tparam ValuesIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``RangeIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam ValuesNumItemsT
  //!   is a model of integral type representing the number of elements in the range of values to be searched for.
  //!
  //! @tparam OutputIteratorT
  //!   is a model of [Random Access Iterator], whose value type is assignable from ``RangeIteratorT``'s difference
  //!   type.
  //!
  //! @tparam CompareOpT
  //!   is a model of [Strict Weak Ordering], which forms a [Relation] with the value types of ``RangeIteratorT``
  //!   and ``ValuesIteratorT``.
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_range
  //!   Iterator to the beginning of the ordered range to be searched.
  //!
  //! @param[in] range_num_items
  //!   Number of elements in the ordered range to be searched.
  //!
  //! @param[in] d_values
  //!   Iterator to the beginning of the range of values to be searched for.
  //!
  //! @param[in] values_num_items
  //!   Number of elements in the range of values to be searched for.
  //!
  //! @param[out] d_output
  //!   Iterator to the beginning of the output range.
  //!
  //! @param[in] comp
  //!   Comparison function object which returns true if its first argument is ordered before the second in the
  //!   [Strict Weak Ordering] of the range to be searched.
  //!
  //! @param[in] stream
  //!   **[optional]** CUDA stream to launch kernels within.
  //!   Default is stream<sub>0</sub>.
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t LowerBound(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::LowerBound");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;

    return DeviceFor::ForEachN(
      d_temp_storage,
      temp_storage_bytes,
      ::cuda::make_zip_iterator(d_values, d_output),
      static_cast<ValuesOffsetT>(values_num_items),
      detail::find::make_comp_wrapper<detail::find::lower_bound>(
        d_range, static_cast<RangeOffsetT>(range_num_items), comp),
      stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``, performs a binary search in the range
  //! ``[d_range, d_range + range_num_items)``,
  //! using ``comp`` as the comparator to find the iterator to the element of said range which **is** ordered
  //! **after** ``value``.
  //!
  //! - The range ``[d_range, d_range + range_num_items)`` must be sorted consistently with ``comp``.
  //!
  //! .. versionadded:: 3.3.0
  //! @endrst
  //!
  //! @tparam RangeIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``ValuesIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam RangeNumItemsT
  //!   is an integral type representing the number of elements in the range to be searched.
  //!
  //! @tparam ValuesIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``RangeIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam ValuesNumItemsT
  //!   is a model of integral type representing the number of elements in the range of values to be searched for.
  //!
  //! @tparam OutputIteratorT
  //!   is a model of [Random Access Iterator], whose value type is assignable from ``RangeIteratorT``'s difference
  //!   type.
  //!
  //! @tparam CompareOpT
  //!   is a model of [Strict Weak Ordering], which forms a [Relation] with the value types of ``RangeIteratorT``
  //!   and ``ValuesIteratorT``.
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_range
  //!   Iterator to the beginning of the ordered range to be searched.
  //!
  //! @param[in] range_num_items
  //!   Number of elements in the ordered range to be searched.
  //!
  //! @param[in] d_values
  //!   Iterator to the beginning of the range of values to be searched for.
  //!
  //! @param[in] values_num_items
  //!   Number of elements in the range of values to be searched for.
  //!
  //! @param[out] d_output
  //!   Iterator to the beginning of the output range.
  //!
  //! @param[in] comp
  //!   Comparison function object which returns true if its first argument is ordered before the second in the
  //!   [Strict Weak Ordering] of the range to be searched.
  //!
  //! @param[in] stream
  //!   **[optional]** CUDA stream to launch kernels within.
  //!   Default is stream<sub>0</sub>.
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t UpperBound(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::UpperBound");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;

    return DeviceFor::ForEachN(
      d_temp_storage,
      temp_storage_bytes,
      ::cuda::make_zip_iterator(d_values, d_output),
      static_cast<ValuesOffsetT>(values_num_items),
      detail::find::make_comp_wrapper<detail::find::upper_bound>(
        d_range, static_cast<RangeOffsetT>(range_num_items), comp),
      stream);
  }
};

CUB_NAMESPACE_END
