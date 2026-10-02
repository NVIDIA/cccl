// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__iterator/iterator_traits.h>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// the following iterator helpers are not named iter_value_t etc, like the C++20 facilities, because they are defined in
// terms of C++17 iterator_traits and not the new C++20 indirectly_readable trait etc. This allows them to detect nested
// value_type, difference_type and reference aliases, which the new C+20 traits do not consider (they only consider
// specializations of iterator_traits). Also, a value_type of void remains supported (needed by some output iterators).

template <typename It>
using it_value_t = typename ::cuda::std::iterator_traits<It>::value_type;

template <typename It>
using it_reference_t = typename ::cuda::std::iterator_traits<It>::reference;

template <typename It>
using it_difference_t = typename ::cuda::std::iterator_traits<It>::difference_type;

template <typename It>
using it_pointer_t = typename ::cuda::std::iterator_traits<It>::pointer;
} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
