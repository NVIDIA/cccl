// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits.h>

#include <limits>

THRUST_NAMESPACE_BEGIN

template <typename T>
struct CCCL_DEPRECATED_BECAUSE("Use cuda::std::numeric_limits") numeric_limits : std::numeric_limits<T>
{};

THRUST_NAMESPACE_END
