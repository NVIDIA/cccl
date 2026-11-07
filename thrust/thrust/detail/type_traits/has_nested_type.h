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

#include <cuda/std/__type_traits/integral_constant.h>

#define __THRUST_DEFINE_HAS_NESTED_TYPE(trait_name, nested_type_name)      \
  template <typename T>                                                    \
  struct trait_name                                                        \
  {                                                                        \
    using yes_type = char;                                                 \
    using no_type  = int;                                                  \
    template <typename S>                                                  \
    _CCCL_HOST_DEVICE static yes_type test(typename S::nested_type_name*); \
    template <typename S>                                                  \
    _CCCL_HOST_DEVICE static no_type test(...);                            \
    static bool constexpr value = sizeof(test<T>(0)) == sizeof(yes_type);  \
    using type                  = ::cuda::std::bool_constant<value>;       \
  };
