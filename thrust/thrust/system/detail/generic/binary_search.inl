/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/binary_search.h>
#include <thrust/detail/function.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/for_each.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/detail/generic/scalar/binary_search.h>
#include <thrust/system/detail/generic/select_system.h>

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// XXX WAR circular #inclusion with this forward declaration
template <typename, typename>
class temporary_array;
} // namespace detail

namespace system::detail::generic
{
namespace detail
{
// short names to avoid nvcc bug
struct lbf
{
  template <typename RandomAccessIterator, typename T, typename StrictWeakOrdering>
  _CCCL_HOST_DEVICE thrust::detail::it_difference_t<RandomAccessIterator>
  operator()(RandomAccessIterator begin, RandomAccessIterator end, const T& value, StrictWeakOrdering comp)
  {
    return thrust::system::detail::generic::scalar::lower_bound(begin, end, value, comp) - begin;
  }
};

struct ubf
{
  template <typename RandomAccessIterator, typename T, typename StrictWeakOrdering>
  _CCCL_HOST_DEVICE thrust::detail::it_difference_t<RandomAccessIterator>
  operator()(RandomAccessIterator begin, RandomAccessIterator end, const T& value, StrictWeakOrdering comp)
  {
    return thrust::system::detail::generic::scalar::upper_bound(begin, end, value, comp) - begin;
  }
};

struct bsf
{
  template <typename RandomAccessIterator, typename T, typename StrictWeakOrdering>
  _CCCL_HOST_DEVICE bool
  operator()(RandomAccessIterator begin, RandomAccessIterator end, const T& value, StrictWeakOrdering comp)
  {
    RandomAccessIterator iter = thrust::system::detail::generic::scalar::lower_bound(begin, end, value, comp);

    thrust::detail::wrapped_function<StrictWeakOrdering, bool> wrapped_comp{comp};

    return iter != end && !wrapped_comp(value, *iter);
  }
};

template <typename ForwardIterator, typename StrictWeakOrdering, typename BinarySearchFunction>
struct binary_search_functor
{
  ForwardIterator begin;
  ForwardIterator end;
  StrictWeakOrdering comp;
  BinarySearchFunction func;

  _CCCL_HOST_DEVICE
  binary_search_functor(ForwardIterator begin, ForwardIterator end, StrictWeakOrdering comp, BinarySearchFunction func)
      : begin(begin)
      , end(end)
      , comp(comp)
      , func(func)
  {}

  template <typename Tuple>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    thrust::get<1>(t) = func(begin, end, thrust::get<0>(t), comp);
  }
}; // binary_search_functor

// Vector Implementation
template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering,
          typename BinarySearchFunction>
_CCCL_HOST_DEVICE OutputIterator binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp,
  BinarySearchFunction func)
{
  thrust::for_each(
    exec,
    thrust::make_zip_iterator(values_begin, output),
    thrust::make_zip_iterator(values_end, output + ::cuda::std::distance(values_begin, values_end)),
    detail::binary_search_functor<ForwardIterator, StrictWeakOrdering, BinarySearchFunction>(begin, end, comp, func));

  return output + ::cuda::std::distance(values_begin, values_end);
}

// Scalar Implementation
template <typename OutputType,
          typename DerivedPolicy,
          typename ForwardIterator,
          typename T,
          typename StrictWeakOrdering,
          typename BinarySearchFunction>
_CCCL_HOST_DEVICE OutputType binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp,
  BinarySearchFunction func)
{
  // use the vectorized path to implement the scalar version

  // allocate device buffers for value and output
  thrust::detail::temporary_array<T, DerivedPolicy> d_value(exec, 1);
  thrust::detail::temporary_array<OutputType, DerivedPolicy> d_output(exec, 1);

  { // copy value to device
    using value_in_system_t = typename thrust::iterator_system<const T*>::type;
    value_in_system_t value_in_system;
    using thrust::system::detail::generic::select_system;
    thrust::copy_n(select_system(thrust::detail::derived_cast(thrust::detail::strip_const(value_in_system)),
                                 thrust::detail::derived_cast(thrust::detail::strip_const(exec))),
                   &value,
                   1,
                   d_value.begin());
  }

  // perform the query
  thrust::system::detail::generic::detail::binary_search(
    exec, begin, end, d_value.begin(), d_value.end(), d_output.begin(), comp, func);

  OutputType output;
  { // copy result to host and return
    using result_out_system_t = typename thrust::iterator_system<OutputType*>::type;
    result_out_system_t result_out_system;
    using thrust::system::detail::generic::select_system;
    thrust::copy_n(select_system(thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
                                 thrust::detail::derived_cast(thrust::detail::strip_const(result_out_system))),
                   d_output.begin(),
                   1,
                   &output);
  }

  return output;
}
} // end namespace detail

//////////////////////
// Scalar Functions //
//////////////////////

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE ForwardIterator
lower_bound(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value)
{
  namespace p = thrust::placeholders;
  return thrust::lower_bound(exec, begin, end, value, ::cuda::std::less<>{});
}

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ForwardIterator lower_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  using difference_type = thrust::detail::it_difference_t<ForwardIterator>;

  return begin + detail::binary_search<difference_type>(exec, begin, end, value, comp, detail::lbf());
}

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE ForwardIterator
upper_bound(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value)
{
  namespace p = thrust::placeholders;
  return thrust::upper_bound(exec, begin, end, value, ::cuda::std::less<>{});
}

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ForwardIterator upper_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  using difference_type = thrust::detail::it_difference_t<ForwardIterator>;

  return begin + detail::binary_search<difference_type>(exec, begin, end, value, comp, detail::ubf());
}

template <typename DerivedPolicy, typename ForwardIterator, typename T>
_CCCL_HOST_DEVICE bool
binary_search(thrust::execution_policy<DerivedPolicy>& exec, ForwardIterator begin, ForwardIterator end, const T& value)
{
  return thrust::binary_search(exec, begin, end, value, ::cuda::std::less<>{});
}

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE bool binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  const T& value,
  StrictWeakOrdering comp)
{
  return detail::binary_search<bool>(exec, begin, end, value, comp, detail::bsf());
}

//////////////////////
// Vector Functions //
//////////////////////

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator lower_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output)
{
  namespace p = thrust::placeholders;
  return thrust::lower_bound(exec, begin, end, values_begin, values_end, output, ::cuda::std::less<>{});
}

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator lower_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp)
{
  return detail::binary_search(exec, begin, end, values_begin, values_end, output, comp, detail::lbf());
}

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator upper_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output)
{
  namespace p = thrust::placeholders;
  return thrust::upper_bound(exec, begin, end, values_begin, values_end, output, ::cuda::std::less<>{});
}

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator upper_bound(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp)
{
  return detail::binary_search(exec, begin, end, values_begin, values_end, output, comp, detail::ubf());
}

template <typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output)
{
  namespace p = thrust::placeholders;
  return thrust::binary_search(exec, begin, end, values_begin, values_end, output, ::cuda::std::less<>{});
}

template <typename DerivedPolicy,
          typename ForwardIterator,
          typename InputIterator,
          typename OutputIterator,
          typename StrictWeakOrdering>
_CCCL_HOST_DEVICE OutputIterator binary_search(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator begin,
  ForwardIterator end,
  InputIterator values_begin,
  InputIterator values_end,
  OutputIterator output,
  StrictWeakOrdering comp)
{
  return detail::binary_search(exec, begin, end, values_begin, values_end, output, comp, detail::bsf());
}

template <typename DerivedPolicy, typename ForwardIterator, typename LessThanComparable>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> equal_range(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const LessThanComparable& value)
{
  return thrust::equal_range(exec, first, last, value, ::cuda::std::less<>{});
}

template <typename DerivedPolicy, typename ForwardIterator, typename T, typename StrictWeakOrdering>
_CCCL_HOST_DEVICE ::cuda::std::pair<ForwardIterator, ForwardIterator> equal_range(
  thrust::execution_policy<DerivedPolicy>& exec,
  ForwardIterator first,
  ForwardIterator last,
  const T& value,
  StrictWeakOrdering comp)
{
  ForwardIterator lb = thrust::lower_bound(exec, first, last, value, comp);
  ForwardIterator ub = thrust::upper_bound(exec, first, last, value, comp);
  return ::cuda::std::make_pair(lb, ub);
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
