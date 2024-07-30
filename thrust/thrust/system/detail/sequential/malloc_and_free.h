/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

_CCCL_IMPLICIT_SYSTEM_HEADER
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cstdlib> // for malloc & free

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{

template <typename DerivedPolicy>
inline _CCCL_HOST_DEVICE void* malloc(execution_policy<DerivedPolicy>&, std::size_t n)
{
  return std::malloc(n);
} // end mallc()

template <typename DerivedPolicy, typename Pointer>
inline _CCCL_HOST_DEVICE void free(sequential::execution_policy<DerivedPolicy>&, Pointer ptr)
{
  std::free(thrust::raw_pointer_cast(ptr));
} // end mallc()

} // namespace sequential
} // namespace detail
} // namespace system
THRUST_NAMESPACE_END
