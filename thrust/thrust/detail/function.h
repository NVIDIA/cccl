// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
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
#include <thrust/detail/raw_reference_cast.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename Function, typename Result>
struct wrapped_function
{
  // mutable because Function::operator() might be const
  mutable Function m_f;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename... Ts>
  inline _CCCL_HOST_DEVICE Result operator()(Ts&&... args) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(::cuda::std::forward<Ts>(args))...));
  }
}; // end wrapped_function
} // namespace detail

THRUST_NAMESPACE_END
