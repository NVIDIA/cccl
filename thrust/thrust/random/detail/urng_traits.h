// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA Corporation. All rights reserved.
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

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_convertible.h>

THRUST_NAMESPACE_BEGIN

namespace random::detail
{
template <typename UniformRandomNumberGenerator>
using urng_result_t = typename UniformRandomNumberGenerator::result_type;

template <typename UniformRandomNumberGenerator, typename = void>
inline constexpr bool has_static_min_member_v = false;

template <typename UniformRandomNumberGenerator>
inline constexpr bool has_static_min_member_v<
  UniformRandomNumberGenerator,
  ::cuda::std::enable_if_t<::cuda::std::is_convertible_v<decltype(UniformRandomNumberGenerator::min),
                                                         urng_result_t<UniformRandomNumberGenerator>>>> = true;

template <typename UniformRandomNumberGenerator, typename = void>
inline constexpr bool has_static_max_member_v = false;

template <typename UniformRandomNumberGenerator>
inline constexpr bool has_static_max_member_v<
  UniformRandomNumberGenerator,
  ::cuda::std::enable_if_t<::cuda::std::is_convertible_v<decltype(UniformRandomNumberGenerator::max),
                                                         urng_result_t<UniformRandomNumberGenerator>>>> = true;

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr typename UniformRandomNumberGenerator::result_type uniform_random_number_generator_min()
{
  if constexpr (has_static_min_member_v<UniformRandomNumberGenerator>)
  {
    return UniformRandomNumberGenerator::min;
  }
  else
  {
    return (UniformRandomNumberGenerator::min) ();
  }
}

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr typename UniformRandomNumberGenerator::result_type uniform_random_number_generator_max()
{
  if constexpr (has_static_max_member_v<UniformRandomNumberGenerator>)
  {
    return UniformRandomNumberGenerator::max;
  }
  else
  {
    return (UniformRandomNumberGenerator::max) ();
  }
}

template <typename UniformRandomNumberGenerator>
struct urng_traits
{
  using result_type = typename UniformRandomNumberGenerator::result_type;

  _CCCL_HOST_DEVICE static constexpr result_type min THRUST_PREVENT_MACRO_SUBSTITUTION()
  {
    return uniform_random_number_generator_min<UniformRandomNumberGenerator>();
  }

  _CCCL_HOST_DEVICE static constexpr result_type max THRUST_PREVENT_MACRO_SUBSTITUTION()
  {
    return uniform_random_number_generator_max<UniformRandomNumberGenerator>();
  }
};
} // namespace random::detail

THRUST_NAMESPACE_END
