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
#include <thrust/system/detail/generic/adjacent_difference.h>
#include <thrust/system/tbb/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system::tbb::detail
{
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
OutputIterator adjacent_difference(
  execution_policy<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op)
{
  // tbb prefers generic::adjacent_difference to cpp::adjacent_difference
  return thrust::system::detail::generic::adjacent_difference(exec, first, last, result, binary_op);
} // end adjacent_difference()
} // namespace system::tbb::detail
THRUST_NAMESPACE_END
