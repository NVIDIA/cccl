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
#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN

namespace system::detail::generic::scalar
{
template <typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE RandomAccessIterator
lower_bound_n(RandomAccessIterator first, Size n, const T& val, BinaryPredicate comp);

template <typename RandomAccessIterator, typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE RandomAccessIterator
lower_bound(RandomAccessIterator first, RandomAccessIterator last, const T& val, BinaryPredicate comp);

template <typename RandomAccessIterator, typename Size, typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE RandomAccessIterator
upper_bound_n(RandomAccessIterator first, Size n, const T& val, BinaryPredicate comp);

template <typename RandomAccessIterator, typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE RandomAccessIterator
upper_bound(RandomAccessIterator first, RandomAccessIterator last, const T& val, BinaryPredicate comp);

template <typename RandomAccessIterator, typename T, typename BinaryPredicate>
_CCCL_HOST_DEVICE ::cuda::std::pair<RandomAccessIterator, RandomAccessIterator>
equal_range(RandomAccessIterator first, RandomAccessIterator last, const T& val, BinaryPredicate comp);

template <typename RandomAccessIterator, typename T, typename Compare>
_CCCL_HOST_DEVICE bool
binary_search(RandomAccessIterator first, RandomAccessIterator last, const T& value, Compare comp);
} // namespace system::detail::generic::scalar

THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/scalar/binary_search.inl>
