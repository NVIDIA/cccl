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

#include <cuda/std/limits>
#include <cuda/std/type_traits>

// #include <stdint.h> // for intmax_t (not provided on MSVS 2005)

THRUST_NAMESPACE_BEGIN

namespace detail
{

// XXX good enough for the platforms we care about
using intmax_t = long long;

template <typename Number>
struct is_signed : integral_constant<bool, ::cuda::std::numeric_limits<Number>::is_signed>
{}; // end is_signed

template <typename T>
struct num_digits
    : eval_if<::cuda::std::numeric_limits<T>::is_specialized,
              integral_constant<int, ::cuda::std::numeric_limits<T>::digits>,
              integral_constant<int,
                                sizeof(T) * ::cuda::std::numeric_limits<unsigned char>::digits
                                  - (is_signed<T>::value ? 1 : 0)>>::type
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
  // XXX workaround a pedantic warning in old versions of g++
  //     which complains about &&ing with a constant value
  template <bool x, bool y>
  struct and_
  {
    static const bool value = false;
  };

  template <bool y>
  struct and_<true, y>
  {
    static const bool value = y;
  };

public:
  using type = typename eval_if<
    and_<::cuda::std::numeric_limits<Integer>::is_signed,
         (!::cuda::std::numeric_limits<Integer>::is_bounded
          || (int(::cuda::std::numeric_limits<Integer>::digits) + 1 >= num_digits<intmax_t>::value))>::value,
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

} // namespace detail

THRUST_NAMESPACE_END
