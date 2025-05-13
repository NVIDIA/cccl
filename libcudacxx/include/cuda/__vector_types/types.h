//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___VECTOR_TYPES_TYPES_H
#define _CUDA___VECTOR_TYPES_TYPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  if !_CCCL_CUDA_COMPILATION()
#    include <vector_types.h>
#  endif // _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

using ::char1;
using ::char2;
using ::char3;
using ::char4;

using ::uchar1;
using ::uchar2;
using ::uchar3;
using ::uchar4;

using ::short1;
using ::short2;
using ::short3;
using ::short4;

using ::ushort1;
using ::ushort2;
using ::ushort3;
using ::ushort4;

using ::int1;
using ::int2;
using ::int3;
using ::int4;

using ::uint1;
using ::uint2;
using ::uint3;
using ::uint4;

using ::long1;
using ::long2;
using ::long3;
using ::long4;

using ::ulong1;
using ::ulong2;
using ::ulong3;
using ::ulong4;

using ::longlong1;
using ::longlong2;
using ::longlong3;
using ::longlong4;

using ::ulonglong1;
using ::ulonglong2;
using ::ulonglong3;
using ::ulonglong4;

using ::float1;
using ::float2;
using ::float3;
using ::float4;

using ::double1;
using ::double2;
using ::double3;
using ::double4;

using ::dim3;

_LIBCUDACXX_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___VECTOR_TYPES_TYPES_H
