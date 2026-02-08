// SPDX-FileCopyrightText: Copyright (c) 2018-2020, NVIDIA Corporation. All rights reserved.
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

#include __THRUST_HOST_SYSTEM_ALGORITH_HEADER_INCLUDE(memory_resource.h)

// Some build systems need a hint to know which files we could include
#if 0
#  include <thrust/system/cpp/memory_resource.h>
#  include <thrust/system/omp/memory_resource.h>
#  include <thrust/system/tbb/memory_resource.h>
#endif

THRUST_NAMESPACE_BEGIN

using host_memory_resource = thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::memory_resource;

THRUST_NAMESPACE_END
