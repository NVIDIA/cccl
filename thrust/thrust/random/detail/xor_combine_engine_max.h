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

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/cstddef>
#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN
namespace random::detail
{
template <typename UIntType>
_CCCL_HOST_DEVICE constexpr auto lshift(UIntType lhs, UIntType rhs) -> UIntType
{
  const bool shift_will_overflow = rhs >= ::cuda::std::numeric_limits<UIntType>::digits;
  if (shift_will_overflow)
  {
    return 0;
  }
  return lhs << rhs;
}

template <typename UIntType>
_CCCL_HOST_DEVICE constexpr auto two_to_the_power(UIntType p) -> UIntType
{
  return lshift(UIntType{1}, p);
}

template <typename UIntType>
_CCCL_HOST_DEVICE constexpr auto log2(UIntType n) -> UIntType
{
  UIntType cur = 0;
  while (n > 1)
  {
    n /= 2;
    cur++;
  }
  return cur;
}

template <typename result_type, result_type a, result_type b, int d>
class xor_combine_engine_max_aux_constants
{
public:
  static constexpr result_type two_to_the_d = two_to_the_power(d);
  static constexpr result_type c            = lshift(a, result_type(d));
  static constexpr result_type t            = ::cuda::std::max(c, b);
  static constexpr result_type u            = ::cuda::std::min(c, b);
  static constexpr result_type p            = log2(u);
  static constexpr result_type two_to_the_p = two_to_the_power(p);
  static constexpr result_type k            = t / two_to_the_p;
};

template <typename result_type, result_type, result_type, int>
struct xor_combine_engine_max_aux;

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case4
{
  using constants = xor_combine_engine_max_aux_constants<result_type, a, b, d>;

  static constexpr result_type k_plus_1_times_two_to_the_p = lshift(constants::k + 1, constants::p);

  static constexpr result_type M =
    xor_combine_engine_max_aux<result_type,
                               (constants::u % constants::two_to_the_p) / constants::two_to_the_p,
                               constants::t % constants::two_to_the_p,
                               d>::value;

  static constexpr result_type value = k_plus_1_times_two_to_the_p + M;
};

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case3
{
  using constants = xor_combine_engine_max_aux_constants<result_type, a, b, d>;

  static constexpr result_type k_plus_1_times_two_to_the_p = lshift(constants::k + 1, constants::p);

  static constexpr result_type M =
    xor_combine_engine_max_aux<result_type,
                               (constants::t % constants::two_to_the_p) / constants::two_to_the_p,
                               constants::u % constants::two_to_the_p,
                               d>::value;

  static constexpr result_type value = k_plus_1_times_two_to_the_p + M;
};

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case2
{
  using constants = xor_combine_engine_max_aux_constants<result_type, a, b, d>;

  static constexpr result_type k_plus_1_times_two_to_the_p = lshift(constants::k + 1, constants::p);
  static constexpr result_type value                       = k_plus_1_times_two_to_the_p - 1;
};

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_case1
{
  static constexpr result_type c     = lshift(a, result_type(d));
  static constexpr result_type value = c + b;
};

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_2
{
  using constants = xor_combine_engine_max_aux_constants<result_type, a, b, d>;

  _CCCL_HOST_DEVICE static constexpr result_type compute_value()
  {
    // if k is odd...
    if constexpr (constants::k % 2 == 1)
    {
      return xor_combine_engine_max_aux_case2<result_type, a, b, d>::value;
    }
    // otherwise if a * 2^3 >= b, then case 3
    else if constexpr (a * constants::two_to_the_d >= b)
    {
      return xor_combine_engine_max_aux_case3<result_type, a, b, d>::value;
    }
    else
    {
      // otherwise, case 4
      return xor_combine_engine_max_aux_case4<result_type, a, b, d>::value;
    }
  }

  static constexpr result_type value = compute_value();
};

template <typename result_type, result_type a, result_type b, int d, bool use_case1 = (a == 0) || (b < two_to_the_power(d))>
struct xor_combine_engine_max_aux_1 : xor_combine_engine_max_aux_case1<result_type, a, b, d>
{};

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux_1<result_type, a, b, d, false> : xor_combine_engine_max_aux_2<result_type, a, b, d>
{};

template <typename result_type, result_type a, result_type b, int d>
struct xor_combine_engine_max_aux : xor_combine_engine_max_aux_1<result_type, a, b, d>
{};

template <typename Engine1, size_t s1, typename Engine2, size_t s2, typename result_type>
struct xor_combine_engine_max
{
  static constexpr size_t w = ::cuda::std::numeric_limits<result_type>::digits;
  static constexpr result_type m1 =
    ::cuda::std::min(result_type(Engine1::max - Engine1::min), result_type(two_to_the_power(w - s1) - 1));
  static constexpr result_type m2 =
    ::cuda::std::min(result_type(Engine2::max - Engine2::min), result_type(two_to_the_power(w - s2) - 1));
  static constexpr result_type s = s1 - s2;
  static constexpr result_type M = xor_combine_engine_max_aux<result_type, m1, m2, s>::value;
  // the value is M(m1,m2,s) lshift_w s2
  static constexpr result_type value = lshift(M, result_type(s2));
};
} // namespace random::detail

THRUST_NAMESPACE_END
