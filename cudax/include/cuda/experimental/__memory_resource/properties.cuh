//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH
#define _CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

using ::cuda::mr::device_accessible;
using ::cuda::mr::host_accessible;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_PROPERTIES_CUH
