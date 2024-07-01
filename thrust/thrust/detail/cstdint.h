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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_MSVC)
#  include <stdint.h>
#endif

THRUST_NAMESPACE_BEGIN

namespace detail
{

#if defined(_CCCL_COMPILER_MSVC)

#  if (_MSC_VER < 1300)
using int8_t   = signed char;
using int16_t  = signed short;
using int32_t  = signed int;
using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
#  else
using int8_t   = signed __int8;
using int16_t  = signed __int16;
using int32_t  = signed __int32;
using uint8_t  = unsigned __int8;
using uint16_t = unsigned __int16;
using uint32_t = unsigned __int32;
#  endif
using int64_t  = signed __int64;
using uint64_t = unsigned __int64;

#else

using int8_t   = ::int8_t;
using int16_t  = ::int16_t;
using int32_t  = ::int32_t;
using int64_t  = ::int64_t;
using uint8_t  = ::uint8_t;
using uint16_t = ::uint16_t;
using uint32_t = ::uint32_t;
using uint64_t = ::uint64_t;

#endif

// an oracle to tell us how to define intptr_t
template <int word_size = sizeof(void*)>
struct divine_intptr_t;
template <int word_size = sizeof(void*)>
struct divine_uintptr_t;

// 32b platforms
template <>
struct divine_intptr_t<4>
{
  using type = thrust::detail::int32_t;
};
template <>
struct divine_uintptr_t<4>
{
  using type = thrust::detail::uint32_t;
};

// 64b platforms
template <>
struct divine_intptr_t<8>
{
  using type = thrust::detail::int64_t;
};
template <>
struct divine_uintptr_t<8>
{
  using type = thrust::detail::uint64_t;
};

using intptr_t  = divine_intptr_t<>::type;
using uintptr_t = divine_uintptr_t<>::type;

} // namespace detail

THRUST_NAMESPACE_END
