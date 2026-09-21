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

template <typename ResultType, ResultType A, ResultType B, int D>
class xor_combine_engine_max_aux_constants
{
public:
  static constexpr ResultType two_to_the_d = two_to_the_power(D);
  static constexpr ResultType c            = lshift(A, ResultType(D));
  static constexpr ResultType t            = (::cuda::std::max) (c, B);
  static constexpr ResultType u            = (::cuda::std::min) (c, B);
  static constexpr ResultType p            = log2(u);
  static constexpr ResultType two_to_the_p = two_to_the_power(p);
  static constexpr ResultType k            = t / two_to_the_p;
};

template <typename ResultType, ResultType, ResultType, int>
struct xor_combine_engine_max_aux;

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux_case4
{
  using constants = xor_combine_engine_max_aux_constants<ResultType, A, B, D>;

  static constexpr ResultType k_plus_1_times_two_to_the_p = lshift(constants::k + 1, constants::p);

  static constexpr ResultType M =
    xor_combine_engine_max_aux<ResultType,
                               (constants::u % constants::two_to_the_p) / constants::two_to_the_p,
                               constants::t % constants::two_to_the_p,
                               D>::value;

  static constexpr ResultType value = k_plus_1_times_two_to_the_p + M;
};

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux_case3
{
  using constants = xor_combine_engine_max_aux_constants<ResultType, A, B, D>;

  static constexpr ResultType k_plus_1_times_two_to_the_p = lshift(constants::k + 1, constants::p);

  static constexpr ResultType M =
    xor_combine_engine_max_aux<ResultType,
                               (constants::t % constants::two_to_the_p) / constants::two_to_the_p,
                               constants::u % constants::two_to_the_p,
                               D>::value;

  static constexpr ResultType value = k_plus_1_times_two_to_the_p + M;
};

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux_case2
{
  using constants = xor_combine_engine_max_aux_constants<ResultType, A, B, D>;

  static constexpr ResultType k_plus_1_times_two_to_the_p = lshift(constants::k + 1, constants::p);
  static constexpr ResultType value                       = k_plus_1_times_two_to_the_p - 1;
};

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux_case1
{
  static constexpr ResultType c     = lshift(A, ResultType(D));
  static constexpr ResultType value = c + B;
};

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux_2
{
  using constants = xor_combine_engine_max_aux_constants<ResultType, A, B, D>;

  _CCCL_HOST_DEVICE static constexpr ResultType compute_value()
  {
    // if k is odd...
    if constexpr (constants::k % 2 == 1)
    {
      return xor_combine_engine_max_aux_case2<ResultType, A, B, D>::value;
    }
    // otherwise if A * 2^3 >= B, then case 3
    else if constexpr (A * constants::two_to_the_d >= B)
    {
      return xor_combine_engine_max_aux_case3<ResultType, A, B, D>::value;
    }
    else
    {
      // otherwise, case 4
      return xor_combine_engine_max_aux_case4<ResultType, A, B, D>::value;
    }
  }

  static constexpr ResultType value = compute_value();
};

template <typename ResultType, ResultType A, ResultType B, int D, bool UseCase1 = (A == 0) || (B < two_to_the_power(D))>
struct xor_combine_engine_max_aux_1 : xor_combine_engine_max_aux_case1<ResultType, A, B, D>
{};

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux_1<ResultType, A, B, D, false> : xor_combine_engine_max_aux_2<ResultType, A, B, D>
{};

template <typename ResultType, ResultType A, ResultType B, int D>
struct xor_combine_engine_max_aux : xor_combine_engine_max_aux_1<ResultType, A, B, D>
{};

template <typename Engine1, size_t S1, typename Engine2, size_t S2, typename ResultType>
struct xor_combine_engine_max
{
  static constexpr size_t w = ::cuda::std::numeric_limits<ResultType>::digits;
  static constexpr ResultType m1 =
    (::cuda::std::min) (ResultType(Engine1::max - Engine1::min), ResultType(two_to_the_power(w - S1) - 1));
  static constexpr ResultType m2 =
    (::cuda::std::min) (ResultType(Engine2::max - Engine2::min), ResultType(two_to_the_power(w - S2) - 1));
  static constexpr ResultType s = S1 - S2;
  static constexpr ResultType M = xor_combine_engine_max_aux<ResultType, m1, m2, s>::value;
  // the value is M(m1,m2,s) lshift_w S2
  static constexpr ResultType value = lshift(M, ResultType(S2));
};
} // namespace random::detail

THRUST_NAMESPACE_END
