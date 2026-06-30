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
#include <thrust/erase.h>

THRUST_NAMESPACE_BEGIN

template <class Vector, class U, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int>>
_CCCL_HOST typename Vector::size_type erase(Vector& c, const U& value)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::erase");
  using value_type = typename Vector::value_type;

  auto first = thrust::remove(c.begin(), c.end(), static_cast<value_type>(value));

  auto removed = static_cast<typename Vector::size_type>(::cuda::std::distance(first, c.end()));

  c.erase(first, c.end());

  return removed;
}

template <typename DerivedPolicy, class Vector, class U, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int>>
_CCCL_HOST typename Vector::size_type
erase(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Vector& c, const U& value)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::erase");
  using value_type = typename Vector::value_type;

  auto first = thrust::remove(exec, c.begin(), c.end(), static_cast<value_type>(value));

  auto removed = static_cast<typename Vector::size_type>(::cuda::std::distance(first, c.end()));
  c.erase(first, c.end());

  return removed;
}

template <class Vector, class Predicate, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int>>
_CCCL_HOST typename Vector::size_type erase_if(Vector& c, Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::erase_if");

  auto first = thrust::remove_if(c.begin(), c.end(), pred);

  auto removed = static_cast<typename Vector::size_type>(::cuda::std::distance(first, c.end()));

  c.erase(first, c.end());

  return removed;
}

template <typename DerivedPolicy, class Vector, class Predicate, ::cuda::std::enable_if_t<is_thrust_vector_v<Vector>, int>>
_CCCL_HOST typename Vector::size_type
erase_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, Vector& c, Predicate pred)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::erase_if");

  auto first = thrust::remove_if(exec, c.begin(), c.end(), pred);

  auto removed = static_cast<typename Vector::size_type>(::cuda::std::distance(first, c.end()));

  c.erase(first, c.end());

  return removed;
}

THRUST_NAMESPACE_END
