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

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * \brief A simple "null" marker type
 */
struct NullType
{
  using value_type = NullType;

  NullType() = default;

  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE explicit NullType(const T&)
  {}

  template <typename T>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE NullType& operator=(const T&)
  {
    return *this;
  }

  friend _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const NullType&, const NullType&)
  {
    return true;
  }

  friend _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const NullType&, const NullType&)
  {
    return false;
  }
};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
