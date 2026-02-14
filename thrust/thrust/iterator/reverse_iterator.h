// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! @file thrust/iterator/reverse_iterator.h
//! @brief An iterator adaptor which adapts another iterator to traverse backwards
//! @deprecated thrust::reverse iterator has been deprecated in favor of cuda::std::reverse_iterator
/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
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

#include <cuda/std/__iterator/reverse_iterator.h>

THRUST_NAMESPACE_BEGIN

//! Deprecated [Since 3.1]
using ::cuda::std::make_reverse_iterator;
using ::cuda::std::reverse_iterator;

THRUST_NAMESPACE_END
