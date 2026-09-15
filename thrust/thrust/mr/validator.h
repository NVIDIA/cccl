// SPDX-FileCopyrightText: Copyright (c) 2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#ifndef CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <thrust/mr/validator.h> is not supported when compiling with NVRTC, which supports device code only. You can define CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK to disable this warning."
#  endif // _CCCL_COMPILER(NVRTC)
#endif // CCCL_DISABLE_NVRTC_COMPATIBILITY_CHECK

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/config/memory_resource.h>
#include <thrust/mr/memory_resource.h>

#include <cuda/std/__type_traits/is_base_of.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{
template <typename MR>
struct validator
{
  static_assert(::cuda::std::is_base_of_v<memory_resource<typename MR::pointer>, MR>,
                "a type used as a memory resource must derive from memory_resource");
};

template <typename T, typename U>
struct validator2
    : private validator<T>
    , private validator<U>
{};

template <typename T>
struct validator2<T, T> : private validator<T>
{};
} // namespace mr
THRUST_NAMESPACE_END
