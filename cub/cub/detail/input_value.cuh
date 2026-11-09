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

#include <cub/detail/future_value.cuh>
#include <cub/detail/uninitialized_copy.cuh>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
/**
 * \brief Allows algorithms to instantiate a single kernel to support both immediate value and future value.
 */
template <typename T, typename IterT = T*>
struct InputValue
{
  using value_type    = T;
  using iterator_type = IterT;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator T() const
  {
    if (m_is_future)
    {
      return m_future_value;
    }
    return m_immediate_value;
  }
  explicit _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InputValue(T immediate_value)
      : m_is_future(false)
      , m_immediate_value(immediate_value)
  {}
  explicit _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InputValue(FutureValue<T, IterT> future_value)
      : m_is_future(true)
      , m_future_value(future_value)
  {}
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE InputValue(const InputValue& other)
      : m_is_future(other.m_is_future)
  {
    if (m_is_future)
    {
      m_future_value = other.m_future_value;
    }
    else
    {
      detail::uninitialized_copy_single(&m_immediate_value, other.m_immediate_value);
    }
  }

private:
  bool m_is_future;
  union
  {
    FutureValue<T, IterT> m_future_value;
    T m_immediate_value;
  };
};
} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
