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

THRUST_NAMESPACE_BEGIN

namespace random::detail
{
template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr auto uniform_random_number_generator_min_impl(int)
  -> decltype((UniformRandomNumberGenerator::min) ())
{
  return (UniformRandomNumberGenerator::min) ();
}

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr typename UniformRandomNumberGenerator::result_type
uniform_random_number_generator_min_impl(...)
{
  return UniformRandomNumberGenerator::min;
}

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr auto uniform_random_number_generator_max_impl(int)
  -> decltype((UniformRandomNumberGenerator::max) ())
{
  return (UniformRandomNumberGenerator::max) ();
}

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr typename UniformRandomNumberGenerator::result_type
uniform_random_number_generator_max_impl(...)
{
  return UniformRandomNumberGenerator::max;
}

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr typename UniformRandomNumberGenerator::result_type uniform_random_number_generator_min()
{
  return uniform_random_number_generator_min_impl<UniformRandomNumberGenerator>(0);
}

template <typename UniformRandomNumberGenerator>
_CCCL_HOST_DEVICE constexpr typename UniformRandomNumberGenerator::result_type uniform_random_number_generator_max()
{
  return uniform_random_number_generator_max_impl<UniformRandomNumberGenerator>(0);
}

template <typename UniformRandomNumberGenerator>
struct urng_traits
{
  using result_type = typename UniformRandomNumberGenerator::result_type;

  _CCCL_HOST_DEVICE static constexpr result_type min()
  {
    return uniform_random_number_generator_min<UniformRandomNumberGenerator>();
  }

  _CCCL_HOST_DEVICE static constexpr result_type max()
  {
    return uniform_random_number_generator_max<UniformRandomNumberGenerator>();
  }
};
} // namespace random::detail

THRUST_NAMESPACE_END
