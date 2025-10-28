//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
#define _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/any_resource.h>
#include <cuda/__memory_resource/get_property.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__utility/basic_any.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
using ::cuda::mr::any_resource;
using ::cuda::mr::any_synchronous_resource;
using ::cuda::mr::make_any_resource;
using ::cuda::mr::make_any_synchronous_resource;
using ::cuda::mr::resource_ref;
using ::cuda::mr::synchronous_resource_ref;

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__MEMORY_RESOURCE_ANY_RESOURCE_H
