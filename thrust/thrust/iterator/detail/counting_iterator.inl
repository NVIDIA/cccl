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

#include <thrust/detail/type_traits.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>

#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

// forward declaration of counting_iterator
template <typename Incrementable, typename System, typename Traversal, typename Difference>
class counting_iterator;

namespace detail
{

template <typename T>
struct num_digits
    : eval_if<::cuda::std::numeric_limits<T>::is_specialized,
              integral_constant<int, ::cuda::std::numeric_limits<T>::digits>,
              integral_constant<int,
                                sizeof(T) * ::cuda::std::numeric_limits<unsigned char>::digits
                                  - (::cuda::std::numeric_limits<T>::is_signed ? 1 : 0)>>::type
{}; // end num_digits

template <typename Integer>
struct integer_difference
//: eval_if<
//    sizeof(Integer) >= sizeof(intmax_t),
//    eval_if<
//      is_signed<Integer>::value,
//      identity_<Integer>,
//      identity_<intmax_t>
//    >,
//    eval_if<
//      sizeof(Integer) < sizeof(std::ptrdiff_t),
//      identity_<std::ptrdiff_t>,
//      identity_<intmax_t>
//    >
//  >
{
private:

public:
  using type =
    typename eval_if<::cuda::std::numeric_limits<Integer>::is_signed
                       && (!::cuda::std::numeric_limits<Integer>::is_bounded
                           || (int(::cuda::std::numeric_limits<Integer>::digits) + 1 >= num_digits<intmax_t>::value)),
                     identity_<Integer>,
                     eval_if<int(::cuda::std::numeric_limits<Integer>::digits) + 1 < num_digits<int>::value,
                             identity_<int>,
                             eval_if<int(::cuda::std::numeric_limits<Integer>::digits) + 1 < num_digits<long>::value,
                                     identity_<long>,
                                     identity_<intmax_t>>>>::type;
}; // end integer_difference

template <typename Number>
struct numeric_difference
    : eval_if<::cuda::std::is_integral<Number>::value, integer_difference<Number>, identity_<Number>>
{}; // end numeric_difference

template <typename Number>
_CCCL_HOST_DEVICE typename numeric_difference<Number>::type numeric_distance(Number x, Number y)
{
  using difference_type = typename numeric_difference<Number>::type;
  return difference_type(y) - difference_type(x);
} // end numeric_distance

template <typename Incrementable, typename System, typename Traversal, typename Difference>
struct counting_iterator_base
{
  using system = typename thrust::detail::eval_if<::cuda::std::is_same<System, use_default>::value,
                                                  thrust::detail::identity_<thrust::any_system_tag>,
                                                  thrust::detail::identity_<System>>::type;

  using traversal = typename thrust::detail::ia_dflt_help<
    Traversal,
    thrust::detail::eval_if<thrust::detail::is_numeric<Incrementable>::value,
                            thrust::detail::identity_<random_access_traversal_tag>,
                            thrust::iterator_traversal<Incrementable>>>::type;

  // unlike Boost, we explicitly use std::ptrdiff_t as the difference type
  // for floating point counting_iterators
  using difference = typename thrust::detail::ia_dflt_help<
    Difference,
    thrust::detail::eval_if<thrust::detail::is_numeric<Incrementable>::value,
                            thrust::detail::eval_if<::cuda::std::is_integral<Incrementable>::value,
                                                    thrust::detail::numeric_difference<Incrementable>,
                                                    thrust::detail::identity_<::cuda::std::ptrdiff_t>>,
                            thrust::iterator_difference<Incrementable>>>::type;

  // our implementation departs from Boost's in that counting_iterator::dereference
  // returns a copy of its counter, rather than a reference to it. returning a reference
  // to the internal state of an iterator causes subtle bugs (consider the temporary
  // iterator created in the expression *(iter + i)) and has no compelling use case
  using type =
    thrust::iterator_adaptor<counting_iterator<Incrementable, System, Traversal, Difference>,
                             Incrementable,
                             Incrementable,
                             system,
                             traversal,
                             Incrementable,
                             difference>;
}; // end counting_iterator_base

template <typename Difference, typename Incrementable1, typename Incrementable2>
struct iterator_distance
{
  _CCCL_HOST_DEVICE static Difference distance(Incrementable1 x, Incrementable2 y)
  {
    return y - x;
  }
};

template <typename Difference, typename Incrementable1, typename Incrementable2>
struct number_distance
{
  _CCCL_HOST_DEVICE static Difference distance(Incrementable1 x, Incrementable2 y)
  {
    return static_cast<Difference>(numeric_distance(x, y));
  }
};

template <typename Difference, typename Incrementable1, typename Incrementable2, typename Enable = void>
struct counting_iterator_equal
{
  _CCCL_HOST_DEVICE static bool equal(Incrementable1 x, Incrementable2 y)
  {
    return x == y;
  }
};

// specialization for floating point equality
template <typename Difference, typename Incrementable1, typename Incrementable2>
struct counting_iterator_equal<Difference,
                               Incrementable1,
                               Incrementable2,
                               ::cuda::std::enable_if_t<::cuda::std::is_floating_point<Incrementable1>::value
                                                        || ::cuda::std::is_floating_point<Incrementable2>::value>>
{
  _CCCL_HOST_DEVICE static bool equal(Incrementable1 x, Incrementable2 y)
  {
    using d = number_distance<Difference, Incrementable1, Incrementable2>;
    return d::distance(x, y) == 0;
  }
};

} // namespace detail
THRUST_NAMESPACE_END
