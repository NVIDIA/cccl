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
#include <cub/detail/env_dispatch.cuh>
#include <cub/device/device_for.cuh>
#include <cub/device/device_transform.cuh>
#include <cub/device/dispatch/dispatch_find.cuh>
#include <cub/device/dispatch/dispatch_find_bound_sorted_values.cuh>
#include <cub/thread/thread_operators.cuh>

#include <cuda/__functional/always_true_false.h>
#include <cuda/__iterator/zip_iterator.h>
#include <cuda/__nvtx/nvtx.h>

CUB_NAMESPACE_BEGIN

//! @rst
//! @par Tuning
//! The FindIf algorithms that accept an environment can be tuned by passing a custom
//! :ref:`policy selector <cub-policy-selectors>` that returns a @ref FindPolicy, as shown in the
//! example below:
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_find_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin find-if-policy-selector
//!      :end-before: example-end find-if-policy-selector
//!
//!  .. literalinclude:: ../../../cub/test/catch2_test_device_find_env_api.cu
//!      :language: c++
//!      :dedent:
//!      :start-after: example-begin find-if-tuning
//!      :end-before: example-end find-if-tuning
//!
//! @endrst
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
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin find-if-predicate
  //!     :end-before: example-end find-if-predicate
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin device-find-if
  //!     :end-before: example-end device-find-if
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t FindIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT scan_op,
    NumItemsT num_items,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceFind::FindIf");

    using OffsetT = detail::choose_offset_t<NumItemsT>;

    using default_policy_selector = detail::find::policy_selector_from_types<detail::it_value_t<InputIteratorT>>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      d_temp_storage,
      temp_storage_bytes,
      env,
      [&](auto policy_selector, void* storage, size_t& bytes, cudaStream_t stream) {
        return detail::find::dispatch(
          storage, bytes, d_in, d_out, static_cast<OffsetT>(num_items), scan_op, stream, policy_selector);
      });
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``, performs a binary search in the range
  //! ``[d_range, d_range + range_num_items)``, using ``comp`` as the comparator to find the iterator to the
  //! **first** element of said range which **is not** ordered **before** ``value``.
  //!
  //! - The range ``[d_range, d_range + range_num_items)`` must be sorted consistently with ``comp``.
  //!
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the lower bound search.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin device-lower-bound
  //!     :end-before: example-end device-lower-bound
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t LowerBound(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceFind::LowerBound");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;

    return detail::dispatch_with_env(
      d_temp_storage,
      temp_storage_bytes,
      env,
      [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
        if (storage == nullptr)
        {
          bytes = 1;
          return cudaSuccess;
        }

        return DeviceTransform::__transform_internal(
          ::cuda::std::make_tuple(d_values),
          d_output,
          static_cast<ValuesOffsetT>(values_num_items),
          ::cuda::always_true{},
          detail::find::make_binary_search_transform_op<detail::find::lower_bound>(
            d_range, static_cast<RangeOffsetT>(range_num_items), comp),
          ::cuda::stream_ref{stream});
      });
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``, performs a binary search in the range
  //! ``[d_range, d_range + range_num_items)``,
  //! using ``comp`` as the comparator to find the iterator to the **first** element of said range which **is**
  //! ordered **after** ``value``.
  //!
  //! - The range ``[d_range, d_range + range_num_items)`` must be sorted consistently with ``comp``.
  //!
  //! .. versionadded:: 3.3.0
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the upper bound search.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin device-upper-bound
  //!     :end-before: example-end device-upper-bound
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
  //!
  //! @param[in] d_temp_storage
  //!   @devicestorage
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  CUB_RUNTIME_FUNCTION static cudaError_t UpperBound(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceFind::UpperBound");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;

    return detail::dispatch_with_env(
      d_temp_storage,
      temp_storage_bytes,
      env,
      [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
        if (storage == nullptr)
        {
          bytes = 1;
          return cudaSuccess;
        }

        return DeviceTransform::__transform_internal(
          ::cuda::std::make_tuple(d_values),
          d_output,
          static_cast<ValuesOffsetT>(values_num_items),
          ::cuda::always_true{},
          detail::find::make_binary_search_transform_op<detail::find::upper_bound>(
            d_range, static_cast<RangeOffsetT>(range_num_items), comp),
          ::cuda::stream_ref{stream});
      });
  }
  //! @rst
  //! Finds the first element in the input sequence that satisfies the given predicate.
  //!
  //! - The search terminates at the first element where the predicate evaluates to true.
  //! - The index of the found element is written to ``d_out``.
  //! - If no element satisfies the predicate, ``num_items`` is written to ``d_out``.
  //! - The range ``[d_out, d_out + 1)`` shall not overlap ``[d_in, d_in + num_items)`` in any way.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!

  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the finding of the first element that satisfies the predicate.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin find-if-predicate
  //!     :end-before: example-end find-if-predicate
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin find-if-env
  //!     :end-before: example-end find-if-env
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ScanOpT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t
  FindIf(InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, NumItemsT num_items, const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::FindIf");

    using OffsetT = detail::choose_offset_t<NumItemsT>;

    using default_policy_selector = detail::find::policy_selector_from_types<detail::it_value_t<InputIteratorT>>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* storage, size_t& bytes, cudaStream_t stream) {
        return detail::find::dispatch(
          storage, bytes, d_in, d_out, static_cast<OffsetT>(num_items), scan_op, stream, policy_selector);
      });
  }

  //! @rst
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``, performs a binary search in the range
  //! ``[d_range, d_range + range_num_items)``, using ``comp`` as the comparator to find the iterator to the
  //! **first** element of said range which **is not** ordered **before** ``value``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - The range ``[d_range, d_range + range_num_items)`` must be sorted consistently with ``comp``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the lower bound search.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin lower-bound-env
  //!     :end-before: example-end lower-bound-env
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t LowerBound(
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::LowerBound");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      if (storage == nullptr)
      {
        bytes = 1;
        return cudaSuccess;
      }

      return DeviceTransform::__transform_internal(
        ::cuda::std::make_tuple(d_values),
        d_output,
        static_cast<ValuesOffsetT>(values_num_items),
        ::cuda::always_true{},
        detail::find::make_binary_search_transform_op<detail::find::lower_bound>(
          d_range, static_cast<RangeOffsetT>(range_num_items), comp),
        ::cuda::stream_ref{stream});
    });
  }

  //! @rst
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``, performs a binary search in the range
  //! ``[d_range, d_range + range_num_items)``,
  //! using ``comp`` as the comparator to find the iterator to the **first** element of said range which **is**
  //! ordered **after** ``value``.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - The range ``[d_range, d_range + range_num_items)`` must be sorted consistently with ``comp``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the upper bound search.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin upper-bound-env
  //!     :end-before: example-end upper-bound-env
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
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
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t UpperBound(
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    const EnvT& env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::UpperBound");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;

    return detail::dispatch_with_env(env, [&]([[maybe_unused]] auto tuning, void* storage, size_t& bytes, auto stream) {
      if (storage == nullptr)
      {
        bytes = 1;
        return cudaSuccess;
      }

      return DeviceTransform::__transform_internal(
        ::cuda::std::make_tuple(d_values),
        d_output,
        static_cast<ValuesOffsetT>(values_num_items),
        ::cuda::always_true{},
        detail::find::make_binary_search_transform_op<detail::find::upper_bound>(
          d_range, static_cast<RangeOffsetT>(range_num_items), comp),
        ::cuda::stream_ref{stream});
    });
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Accelerated variant of :cpp:func:`LowerBound` that exploits the additional
  //! precondition that ``[d_values, d_values + values_num_items)`` is also
  //! sorted consistently with ``comp``.
  //!
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``,
  //! performs a search in ``[d_range, d_range + range_num_items)`` to find the
  //! iterator to the first element that is **not ordered before** ``value``.
  //!
  //! Because both sequences are sorted, the algorithm uses the Merge-Path
  //! algorithm (Oded et al., IPDPS 2012) to partition the combined traversal
  //! across thread blocks, achieving O(N+M) total device work rather than the
  //! O(M log N) of independent binary searches.
  //!
  //! - Both ``[d_range, d_range + range_num_items)`` **and**
  //!   ``[d_values, d_values + values_num_items)`` must be sorted consistently
  //!   with ``comp``.
  //! - @devicestorage
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! @endrst
  //!
  //! @tparam RangeIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a
  //!   [Relation] with the value type of ``ValuesIteratorT`` via ``CompareOpT``.
  //!
  //! @tparam RangeNumItemsT
  //!   is an integral type representing the number of elements in the range.
  //!
  //! @tparam ValuesIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a
  //!   [Relation] with the value type of ``RangeIteratorT`` via ``CompareOpT``.
  //!
  //! @tparam ValuesNumItemsT
  //!   is an integral type representing the number of values to search for.
  //!
  //! @tparam OutputIteratorT
  //!   is a model of [Random Access Iterator] whose value type is assignable
  //!   from ``RangeIteratorT``'s difference type.
  //!
  //! @tparam CompareOpT
  //!   is a model of [Strict Weak Ordering] over the value types of both
  //!   iterator types.
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
  //!   Iterator to the beginning of the ordered haystack range.
  //!
  //! @param[in] range_num_items
  //!   Number of elements in the haystack range.
  //!
  //! @param[in] d_values
  //!   Iterator to the beginning of the sorted range of needles.
  //!
  //! @param[in] values_num_items
  //!   Number of needle elements.
  //!
  //! @param[out] d_output
  //!   Iterator to the beginning of the output range.
  //!
  //! @param[in] comp
  //!   Comparison function object (Strict Weak Ordering).
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
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
  CUB_RUNTIME_FUNCTION static cudaError_t LowerBoundSortedValues(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceFind::LowerBoundSortedValues");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;
    using OffsetT       = ::cuda::std::common_type_t<RangeOffsetT, ValuesOffsetT>;

    return detail::find_bound_sorted_values::dispatch<detail::find_bound_sorted_values::lower_bound_mode>(
      d_temp_storage,
      temp_storage_bytes,
      d_range,
      static_cast<OffsetT>(range_num_items),
      d_values,
      static_cast<OffsetT>(values_num_items),
      d_output,
      comp,
      stream);
  }

  //! @rst
  //! Accelerated variant of :cpp:func:`LowerBound` that exploits the additional
  //! precondition that ``[d_values, d_values + values_num_items)`` is also
  //! sorted consistently with ``comp``.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Both ``[d_range, d_range + range_num_items)`` **and**
  //!   ``[d_values, d_values + values_num_items)`` must be sorted consistently
  //!   with ``comp``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the lower bound search.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_bound_sorted_values_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin lower-bound-sorted-values-env
  //!     :end-before: example-end lower-bound-sorted-values-env
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
  //!
  //! @param[in] d_range
  //!   Iterator to the beginning of the ordered haystack range.
  //!
  //! @param[in] range_num_items
  //!   Number of elements in the haystack range.
  //!
  //! @param[in] d_values
  //!   Iterator to the beginning of the sorted range of needles.
  //!
  //! @param[in] values_num_items
  //!   Number of needle elements.
  //!
  //! @param[out] d_output
  //!   Iterator to the beginning of the output range.
  //!
  //! @param[in] comp
  //!   Comparison function object (Strict Weak Ordering).
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t LowerBoundSortedValues(
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::LowerBoundSortedValues");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;
    using OffsetT       = ::cuda::std::common_type_t<RangeOffsetT, ValuesOffsetT>;

    using default_policy_selector =
      detail::find_bound_sorted_values::policy_selector_from_types<detail::it_value_t<RangeIteratorT>,
                                                                   detail::it_value_t<ValuesIteratorT>>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* storage, size_t& bytes, auto stream) {
        return detail::find_bound_sorted_values::dispatch<detail::find_bound_sorted_values::lower_bound_mode>(
          storage,
          bytes,
          d_range,
          static_cast<OffsetT>(range_num_items),
          d_values,
          static_cast<OffsetT>(values_num_items),
          d_output,
          comp,
          stream,
          policy_selector);
      });
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Accelerated variant of :cpp:func:`UpperBound` that exploits the additional
  //! precondition that ``[d_values, d_values + values_num_items)`` is also
  //! sorted consistently with ``comp``.
  //!
  //! For each ``value`` in ``[d_values, d_values + values_num_items)``,
  //! performs a search in ``[d_range, d_range + range_num_items)`` to find the
  //! iterator to the first element that is **ordered after** ``value``.
  //!
  //! Because both sequences are sorted, the algorithm uses the Merge-Path
  //! algorithm (Oded et al., IPDPS 2012) to partition the combined traversal
  //! across thread blocks, achieving O(N+M) total device work rather than the
  //! O(M log N) of independent binary searches.
  //!
  //! - Both ``[d_range, d_range + range_num_items)`` **and**
  //!   ``[d_values, d_values + values_num_items)`` must be sorted consistently
  //!   with ``comp``.
  //! - @devicestorage
  //!
  //! .. versionadded:: 3.5.0
  //!
  //! @endrst
  //!
  //! @tparam RangeIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a
  //!   [Relation] with the value type of ``ValuesIteratorT`` via ``CompareOpT``.
  //!
  //! @tparam RangeNumItemsT
  //!   is an integral type representing the number of elements in the range.
  //!
  //! @tparam ValuesIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a
  //!   [Relation] with the value type of ``RangeIteratorT`` via ``CompareOpT``.
  //!
  //! @tparam ValuesNumItemsT
  //!   is an integral type representing the number of values to search for.
  //!
  //! @tparam OutputIteratorT
  //!   is a model of [Random Access Iterator] whose value type is assignable
  //!   from ``RangeIteratorT``'s difference type.
  //!
  //! @tparam CompareOpT
  //!   is a model of [Strict Weak Ordering] over the value types of both
  //!   iterator types.
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
  //!   Iterator to the beginning of the ordered haystack range.
  //!
  //! @param[in] range_num_items
  //!   Number of elements in the haystack range.
  //!
  //! @param[in] d_values
  //!   Iterator to the beginning of the sorted range of needles.
  //!
  //! @param[in] values_num_items
  //!   Number of needle elements.
  //!
  //! @param[out] d_output
  //!   Iterator to the beginning of the output range.
  //!
  //! @param[in] comp
  //!   Comparison function object (Strict Weak Ordering).
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
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
  CUB_RUNTIME_FUNCTION static cudaError_t UpperBoundSortedValues(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    cudaStream_t stream = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceFind::UpperBoundSortedValues");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;
    using OffsetT       = ::cuda::std::common_type_t<RangeOffsetT, ValuesOffsetT>;

    return detail::find_bound_sorted_values::dispatch<detail::find_bound_sorted_values::upper_bound_mode>(
      d_temp_storage,
      temp_storage_bytes,
      d_range,
      static_cast<OffsetT>(range_num_items),
      d_values,
      static_cast<OffsetT>(values_num_items),
      d_output,
      comp,
      stream);
  }

  //! @rst
  //! Accelerated variant of :cpp:func:`UpperBound` that exploits the additional
  //! precondition that ``[d_values, d_values + values_num_items)`` is also
  //! sorted consistently with ``comp``.
  //!
  //! .. versionadded:: 3.5.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //! - Memory resource: Query via ``cuda::mr::get_memory_resource``
  //!
  //! - Both ``[d_range, d_range + range_num_items)`` **and**
  //!   ``[d_values, d_values + values_num_items)`` must be sorted consistently
  //!   with ``comp``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the upper bound search.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_find_bound_sorted_values_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin upper-bound-sorted-values-env
  //!     :end-before: example-end upper-bound-sorted-values-env
  //!
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
  //! @tparam EnvT
  //!   **[inferred]** Environment type (e.g., ``cuda::std::execution::env<...>``)
  //!
  //! @param[in] d_range
  //!   Iterator to the beginning of the ordered haystack range.
  //!
  //! @param[in] range_num_items
  //!   Number of elements in the haystack range.
  //!
  //! @param[in] d_values
  //!   Iterator to the beginning of the sorted range of needles.
  //!
  //! @param[in] values_num_items
  //!   Number of needle elements.
  //!
  //! @param[out] d_output
  //!   Iterator to the beginning of the output range.
  //!
  //! @param[in] comp
  //!   Comparison function object (Strict Weak Ordering).
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  //!
  //! [Random Access Iterator]: https://en.cppreference.com/w/cpp/iterator/random_access_iterator
  //! [Strict Weak Ordering]: https://en.cppreference.com/w/cpp/concepts/strict_weak_order
  //! [Relation]: https://en.cppreference.com/w/cpp/concepts/relation
  template <typename RangeIteratorT,
            typename RangeNumItemsT,
            typename ValuesIteratorT,
            typename ValuesNumItemsT,
            typename OutputIteratorT,
            typename CompareOpT,
            typename EnvT = ::cuda::std::execution::env<>>
  [[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t UpperBoundSortedValues(
    RangeIteratorT d_range,
    RangeNumItemsT range_num_items,
    ValuesIteratorT d_values,
    ValuesNumItemsT values_num_items,
    OutputIteratorT d_output,
    CompareOpT comp,
    EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::UpperBoundSortedValues");

    using RangeOffsetT  = detail::choose_offset_t<RangeNumItemsT>;
    using ValuesOffsetT = detail::choose_offset_t<ValuesNumItemsT>;
    using OffsetT       = ::cuda::std::common_type_t<RangeOffsetT, ValuesOffsetT>;

    using default_policy_selector =
      detail::find_bound_sorted_values::policy_selector_from_types<detail::it_value_t<RangeIteratorT>,
                                                                   detail::it_value_t<ValuesIteratorT>>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* storage, size_t& bytes, auto stream) {
        return detail::find_bound_sorted_values::dispatch<detail::find_bound_sorted_values::upper_bound_mode>(
          storage,
          bytes,
          d_range,
          static_cast<OffsetT>(range_num_items),
          d_values,
          static_cast<OffsetT>(values_num_items),
          d_output,
          comp,
          stream,
          policy_selector);
      });
  }
};

CUB_NAMESPACE_END
