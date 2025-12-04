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

#include <cub/device/device_for.cuh>

#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/upper_bound.h>

CUB_NAMESPACE_BEGIN

namespace detail::find
{
template <typename RangeIteratorT, typename CompareOpT, typename Mode>
struct comp_wrapper_t
{
  RangeIteratorT first;
  RangeIteratorT last;
  CompareOpT op;

  template <typename Value, typename Output>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(cuda::std::tuple<Value, Output> args)
  {
    cuda::std::get<1>(args) = Mode::Invoke(first, last, cuda::std::get<0>(args), op);
  }
};

template <typename Mode, typename RangeIteratorT, typename CompareOpT>
CUB_RUNTIME_FUNCTION auto make_comp_wrapper(RangeIteratorT first, RangeIteratorT last, CompareOpT comp)
{
  return comp_wrapper_t<RangeIteratorT, CompareOpT, Mode>(first, last, comp);
}

struct lower_bound
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static RangeIteratorT
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    return cuda::std::lower_bound(first, last, value, comp);
  }
};

struct upper_bound
{
  template <typename RangeIteratorT, typename T, typename CompareOpT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static RangeIteratorT
  Invoke(RangeIteratorT first, RangeIteratorT last, const T& value, CompareOpT comp)
  {
    return cuda::std::upper_bound(first, last, value, comp);
  }
};
} // namespace detail::find

struct DeviceFind
{
  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! For each ``value`` in ``[values_first, values_last)``, performs a binary search in the range ``[first, last)``,
  //! using ``comp`` as the comparator to find the iterator to the element of said range which **is not** ordered
  //! **before** ``value``.
  //!
  //! - The range ``[first, last)`` must be sorted consistently with ``comp``.
  //!
  //! @endrst
  //!
  //! @tparam RangeIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``ValuesIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam ValuesIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``RangeIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam OutputIteratorT
  //!   is a model of [Random Access Iterator], whose value type is assignable from ``RangeIteratorT``.
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
  //! @param[in] first
  //!   Iterator to the beginning of the ordered range to be searched.
  //!
  //! @param[in] last
  //!   Iterator denoting the one-past-the-end element of the ordered range to be searched.
  //!
  //! @param[in] values_first
  //!   Iterator to the beginning of the range of values to be searched for.
  //!
  //! @param[in] values_last
  //!   Iterator denoting the one-past-the-end element of the range of values to be searched for.
  //!
  //! @param[out] output
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
  template <typename RangeIteratorT, typename ValuesIteratorT, typename OutputIteratorT, typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t LowerBound(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT first,
    RangeIteratorT last,
    ValuesIteratorT values_first,
    ValuesIteratorT values_last,
    OutputIteratorT output,
    CompareOpT comp,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::LowerBound");
    return DeviceFor::ForEach(
      d_temp_storage,
      temp_storage_bytes,
      cuda::make_zip_iterator(values_first, output),
      cuda::make_zip_iterator(values_last, output + cuda::std::distance(values_first, values_last)),
      detail::find::make_comp_wrapper<detail::find::lower_bound>(first, last, comp),
      stream);
  }

  //! @rst
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! For each ``value`` in ``[values_first, values_last)``, performs a binary search in the range ``[first, last)``,
  //! using ``comp`` as the comparator to find the iterator to the element of said range which **is** ordered
  //! **after** ``value``.
  //!
  //! - The range ``[first, last)`` must be sorted consistently with ``comp``.
  //!
  //! @endrst
  //!
  //! @tparam RangeIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``ValuesIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam ValuesIteratorT
  //!   is a model of [Random Access Iterator], whose value type forms a [Relation] with the value type of
  //!   ``RangeIteratorT`` using ``CompareOpT`` as the predicate.
  //!
  //! @tparam OutputIteratorT
  //!   is a model of [Random Access Iterator], whose value type is assignable from ``RangeIteratorT``.
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
  //! @param[in] first
  //!   Iterator to the beginning of the ordered range to be searched.
  //!
  //! @param[in] last
  //!   Iterator denoting the one-past-the-end element of the ordered range to be searched.
  //!
  //! @param[in] values_first
  //!   Iterator to the beginning of the range of values to be searched for.
  //!
  //! @param[in] values_last
  //!   Iterator denoting the one-past-the-end element of the range of values to be searched for.
  //!
  //! @param[out] output
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
  template <typename RangeIteratorT, typename ValuesIteratorT, typename OutputIteratorT, typename CompareOpT>
  CUB_RUNTIME_FUNCTION static cudaError_t UpperBound(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RangeIteratorT first,
    RangeIteratorT last,
    ValuesIteratorT values_first,
    ValuesIteratorT values_last,
    OutputIteratorT output,
    CompareOpT comp,
    cudaStream_t stream = 0)
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceFind::UpperBound");
    return DeviceFor::ForEach(
      d_temp_storage,
      temp_storage_bytes,
      cuda::make_zip_iterator(values_first, output),
      cuda::make_zip_iterator(values_last, output + cuda::std::distance(values_first, values_last)),
      detail::find::make_comp_wrapper<detail::find::upper_bound>(first, last, comp),
      stream);
  }
};

CUB_NAMESPACE_END
