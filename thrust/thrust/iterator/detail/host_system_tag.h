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

#include __THRUST_HOST_SYSTEM_ALGORITH_DETAIL_HEADER_INCLUDE(execution_policy.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/detail/execution_policy.h>
#  include <thrust/system/omp/detail/execution_policy.h>
#  include <thrust/system/tbb/detail/execution_policy.h>
#endif

THRUST_NAMESPACE_BEGIN

using host_system_tag = thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::tag;

THRUST_NAMESPACE_END
