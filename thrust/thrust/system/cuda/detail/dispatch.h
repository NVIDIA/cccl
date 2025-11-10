/*
 *  Copyright 2018 NVIDIA Corporation
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

#include <thrust/detail/preprocessor.h>

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__exception/throw_error.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <string>

THRUST_NAMESPACE_BEGIN
namespace detail
{
_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES(::cuda::std::is_arithmetic_v<T>)
[[nodiscard]] _CCCL_API constexpr bool is_negative([[maybe_unused]] T x) noexcept
{
  if constexpr (::cuda::std::is_unsigned_v<T>)
  {
    return false;
  }
  else
  {
    return x < 0;
  }
}
} // namespace detail
THRUST_NAMESPACE_END

#if defined(THRUST_FORCE_32_BIT_OFFSET_TYPE) && defined(THRUST_FORCE_64_BIT_OFFSET_TYPE)
#  error "Only THRUST_FORCE_32_BIT_OFFSET_TYPE or THRUST_FORCE_64_BIT_OFFSET_TYPE may be defined!"
#endif // THRUST_FORCE_32_BIT_OFFSET_TYPE && THRUST_FORCE_64_BIT_OFFSET_TYPE

#define _THRUST_INDEX_TYPE_DISPATCH(index_type, status, call, count, arguments) \
  {                                                                             \
    auto THRUST_PP_CAT2(count, _fixed) = static_cast<index_type>(count);        \
    status                             = call arguments;                        \
  }

#define _THRUST_INDEX_TYPE_DISPATCH2(index_type, status, call, count1, count2, arguments) \
  {                                                                                       \
    auto THRUST_PP_CAT2(count1, _fixed) = static_cast<index_type>(count1);                \
    auto THRUST_PP_CAT2(count2, _fixed) = static_cast<index_type>(count2);                \
    status                              = call arguments;                                 \
  }

#define _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                           \
  if (thrust::detail::is_negative(count))                                            \
  {                                                                                  \
    ::cuda::std::__throw_runtime_error("Invalid input range, passed negative size"); \
  }

#define _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2) \
  _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count1)                \
  _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count2)

#if defined(THRUST_FORCE_64_BIT_OFFSET_TYPE)
//! @brief Always dispatches to 64 bit offset version of an algorithm
#  define THRUST_INDEX_TYPE_DISPATCH(status, call, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)               \
    _THRUST_INDEX_TYPE_DISPATCH(std::int64_t, status, call, count, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH but with two counts
#  define THRUST_DOUBLE_INDEX_TYPE_DISPATCH(status, call, count1, count2, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2)                     \
    _THRUST_INDEX_TYPE_DISPATCH2(std::int64_t, status, call, count1, count2, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH but with two different call implementations
#  define THRUST_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                            \
    _THRUST_INDEX_TYPE_DISPATCH(std::int64_t, status, call_64, count, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH2 but uses two counts.
#  define THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count1, count2, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2)                                  \
    _THRUST_INDEX_TYPE_DISPATCH2(std::int64_t, status, call_64, count1, count2, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH2 but always dispatching to uint64_t. `count` must not be negative.
#  define THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                                     \
    _THRUST_INDEX_TYPE_DISPATCH(std::uint64_t, status, call_64, count, arguments)

#elif defined(THRUST_FORCE_32_BIT_OFFSET_TYPE)

//! @brief Ensures that the size of the input does not overflow the offset type
#  define _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW(index_type, count)                       \
    if (static_cast<std::uint64_t>(count)                                                     \
        > static_cast<std::uint64_t>(::cuda::std::numeric_limits<index_type>::max()))         \
    {                                                                                         \
      ::cuda::std::__throw_runtime_error(                                                     \
        "Input size exceeds the maximum allowable value for " #index_type                     \
        ". It was used because the macro THRUST_FORCE_32_BIT_OFFSET_TYPE was defined. "       \
        "To handle larger input sizes, either remove this macro to dynamically dispatch "     \
        "between 32-bit and 64-bit index types, or define THRUST_FORCE_64_BIT_OFFSET_TYPE."); \
    }

//! @brief Ensures that the sizes of the inputs do not overflow the offset type, but two counts
#  define _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW2(index_type, count1, count2)             \
    if (static_cast<std::uint64_t>(count1) + static_cast<std::uint64_t>(count2)               \
        > static_cast<std::uint64_t>(::cuda::std::numeric_limits<index_type>::max()))         \
    {                                                                                         \
      ::cuda::std::__throw_runtime_error(                                                     \
        "Input size exceeds the maximum allowable value for " #index_type                     \
        ". It was used because the macro THRUST_FORCE_32_BIT_OFFSET_TYPE was defined. "       \
        "To handle larger input sizes, either remove this macro to dynamically dispatch "     \
        "between 32-bit and 64-bit index types, or define THRUST_FORCE_64_BIT_OFFSET_TYPE."); \
    }

//! @brief Always dispatches to 32 bit offset version of an algorithm but throws if count would overflow
#  define THRUST_INDEX_TYPE_DISPATCH(status, call, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)               \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW(std::int32_t, count)  \
    _THRUST_INDEX_TYPE_DISPATCH(std::int32_t, status, call, count, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH but with two counts
#  define THRUST_DOUBLE_INDEX_TYPE_DISPATCH(status, call, count1, count2, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2)                     \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW2(std::int32_t, count1, count2)        \
    _THRUST_INDEX_TYPE_DISPATCH2(std::int32_t, status, call, count1, count2, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH but with two different call implementations
#  define THRUST_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                            \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW(std::int32_t, count)               \
    _THRUST_INDEX_TYPE_DISPATCH(std::int32_t, status, call_32, count, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH2 but uses two counts.
#  define THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count1, count2, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2)                                  \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW2(std::int32_t, count1, count2)                     \
    _THRUST_INDEX_TYPE_DISPATCH2(std::int32_t, status, call_32, count1, count2, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH but always dispatching to uint64_t. `count` must not be negative.
#  define THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                                     \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_OVERFLOW(std::uint32_t, count)                       \
    _THRUST_INDEX_TYPE_DISPATCH(std::uint32_t, status, call_32, count, arguments)

#else // ^^^ THRUST_FORCE_32_BIT_OFFSET_TYPE ^^^ / vvv !THRUST_FORCE_32_BIT_OFFSET_TYPE vvv

#  define _THRUST_INDEX_TYPE_DISPATCH_SELECT(index_type, count) \
    (static_cast<std::uint64_t>(count) <= static_cast<std::uint64_t>(::cuda::std::numeric_limits<index_type>::max()))

#  define _THRUST_INDEX_TYPE_DISPATCH_SELECT2(index_type, count1, count2)    \
    (static_cast<std::uint64_t>(count1) + static_cast<std::uint64_t>(count2) \
     <= static_cast<std::uint64_t>(::cuda::std::numeric_limits<index_type>::max()))

//! Dispatch between 32-bit and 64-bit index_type based versions of the same algorithm implementation. This version
//! assumes that callables for both branches consist of the same tokens, and is intended to be used with Thrust-style
//! dispatch interfaces, that always deduce the size type from the arguments.
#  define THRUST_INDEX_TYPE_DISPATCH(status, call, count, arguments)            \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                          \
    if _THRUST_INDEX_TYPE_DISPATCH_SELECT (std::int32_t, count)                 \
      _THRUST_INDEX_TYPE_DISPATCH(std::int32_t, status, call, count, arguments) \
    else                                                                        \
      _THRUST_INDEX_TYPE_DISPATCH(std::int64_t, status, call, count, arguments)

//! Dispatch between 32-bit and 64-bit index_type based versions of the same algorithm implementation. This version
//! assumes that callables for both branches consist of the same tokens, and is intended to be used with Thrust-style
//! dispatch interfaces, that always deduce the size type from the arguments.
//!
//! This version of the macro supports providing two count variables, which is necessary for set algorithms.
#  define THRUST_DOUBLE_INDEX_TYPE_DISPATCH(status, call, count1, count2, arguments)      \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2)                          \
    if _THRUST_INDEX_TYPE_DISPATCH_SELECT2 (std::int32_t, count1, count2)                 \
      _THRUST_INDEX_TYPE_DISPATCH2(std::int32_t, status, call, count1, count2, arguments) \
    else                                                                                  \
      _THRUST_INDEX_TYPE_DISPATCH2(std::int64_t, status, call, count1, count2, arguments)

//! Dispatch between 32-bit and 64-bit index_type based versions of the same algorithm implementation. This version
//! allows using different token sequences for callables in both branches, and is intended to be used with CUB-style
//! dispatch interfaces, where the "simple" interface always forces the size to be `int` (making it harder for us to
//! use), but the complex interface that we end up using doesn't actually provide a way to fully deduce the type from
//! just the call, making the size type appear in the token sequence of the callable.
//!
//!  See reduce_n_impl to see an example of how this is meant to be used.
#  define THRUST_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count, arguments)  \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                             \
    if _THRUST_INDEX_TYPE_DISPATCH_SELECT (std::int32_t, count)                    \
      _THRUST_INDEX_TYPE_DISPATCH(std::int32_t, status, call_32, count, arguments) \
    else                                                                           \
      _THRUST_INDEX_TYPE_DISPATCH(std::int64_t, status, call_64, count, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH2 but uses two counts.
#  define THRUST_DOUBLE_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count1, count2, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW2(count1, count2)                                  \
    if _THRUST_INDEX_TYPE_DISPATCH_SELECT2 (std::int32_t, count1, count2)                         \
      _THRUST_INDEX_TYPE_DISPATCH2(std::int32_t, status, call_32, count1, count2, arguments)      \
    else                                                                                          \
      _THRUST_INDEX_TYPE_DISPATCH2(std::int64_t, status, call_64, count1, count2, arguments)

//! Like \ref THRUST_INDEX_TYPE_DISPATCH2 but dispatching to uint32_t and uint64_t, respectively, depending on the
//! `count` argument. `count` must not be negative.
#  define THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(status, call_32, call_64, count, arguments) \
    _THRUST_INDEX_TYPE_DISPATCH_GUARD_UNDERFLOW(count)                                     \
    if _THRUST_INDEX_TYPE_DISPATCH_SELECT (std::uint32_t, count)                           \
      _THRUST_INDEX_TYPE_DISPATCH(std::uint32_t, status, call_32, count, arguments)        \
    else                                                                                   \
      _THRUST_INDEX_TYPE_DISPATCH(std::uint64_t, status, call_64, count, arguments)

#endif // !THRUST_FORCE_32_BIT_OFFSET_TYPE
