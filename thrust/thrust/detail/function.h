/*
 *  Copyright 2024 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
  _CCCL_FORCEINLINE _CCCL_HOST_DEVICE Result operator()(Ts&&... args) const
  {
    return static_cast<Result>(m_f(thrust::raw_reference_cast(::cuda::std::forward<Ts>(args))...));
  }
}; // end wrapped_function
} // namespace detail

THRUST_NAMESPACE_END
