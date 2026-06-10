// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/seq.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/reduce.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/system/tbb/detail/execution_policy.h>

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/cassert>

#include <tbb/parallel_for.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
namespace reduce_intervals_detail
{
template <typename L, typename R>
inline L divide_ri(const L x, const R y)
{
  return (x + (y - 1)) / y;
}

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Size, typename BinaryFunction>
struct body
{
  RandomAccessIterator1 first;
  RandomAccessIterator2 result;
  Size n, interval_size;
  BinaryFunction binary_op;

  body(RandomAccessIterator1 first, RandomAccessIterator2 result, Size n, Size interval_size, BinaryFunction binary_op)
      : first(first)
      , result(result)
      , n(n)
      , interval_size(interval_size)
      , binary_op(binary_op)
  {}

  void operator()(const ::tbb::blocked_range<Size>& r) const
  {
    assert(r.size() == 1);

    Size interval_idx = r.begin();

    Size offset_to_first = interval_size * interval_idx;
    Size offset_to_last  = (::cuda::std::min) (n, offset_to_first + interval_size);

    RandomAccessIterator1 my_first = first + offset_to_first;
    RandomAccessIterator1 my_last  = first + offset_to_last;

    // carefully pass the init value for the interval with raw_reference_cast
    using sum_type = ::cuda::std::decay_t<decltype(binary_op(*my_first, *my_first))>;
    result[interval_idx] =
      thrust::reduce(thrust::seq, my_first + 1, my_last, sum_type(thrust::raw_reference_cast(*my_first)), binary_op);
  }
};

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Size, typename BinaryFunction>
body<RandomAccessIterator1, RandomAccessIterator2, Size, BinaryFunction> make_body(
  RandomAccessIterator1 first, RandomAccessIterator2 result, Size n, Size interval_size, BinaryFunction binary_op)
{
  return body<RandomAccessIterator1, RandomAccessIterator2, Size, BinaryFunction>(
    first, result, n, interval_size, binary_op);
}
} // namespace reduce_intervals_detail

template <typename DerivedPolicy,
          typename RandomAccessIterator1,
          typename Size,
          typename RandomAccessIterator2,
          typename BinaryFunction>
void reduce_intervals(
  thrust::tbb::execution_policy<DerivedPolicy>&,
  RandomAccessIterator1 first,
  RandomAccessIterator1 last,
  Size interval_size,
  RandomAccessIterator2 result,
  BinaryFunction binary_op)
{
  thrust::detail::it_difference_t<RandomAccessIterator1> n = last - first;

  Size num_intervals = reduce_intervals_detail::divide_ri(n, interval_size);

  ::tbb::parallel_for(::tbb::blocked_range<Size>(0, num_intervals, 1),
                      reduce_intervals_detail::make_body(first, result, Size(n), interval_size, binary_op),
                      ::tbb::simple_partitioner());
}

template <typename DerivedPolicy, typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
void reduce_intervals(
  thrust::tbb::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator1 first,
  RandomAccessIterator1 last,
  Size interval_size,
  RandomAccessIterator2 result)
{
  using value_type = thrust::detail::it_value_t<RandomAccessIterator1>;

  return thrust::system::tbb::detail::reduce_intervals(
    exec, first, last, interval_size, result, ::cuda::std::plus<value_type>());
}
} // namespace system::tbb::detail
THRUST_NAMESPACE_END
