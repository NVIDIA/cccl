//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___VECTOR_TYPES_STRUCTURED_BINDINGS_H
#define _CUDA___VECTOR_TYPES_STRUCTURED_BINDINGS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__vector_types/types.h>
#  include <cuda/std/__fwd/get.h>
#  include <cuda/std/__tuple_dir/structured_bindings.h>
#  include <cuda/std/__tuple_dir/tuple_element.h>
#  include <cuda/std/__tuple_dir/tuple_size.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/forward_like.h>

#  include <cuda/std/__cccl/prologue.h>

#  define _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N(__name, __type, __size)                             \
    _LIBCUDACXX_BEGIN_NAMESPACE_STD                                                                 \
                                                                                                    \
    template <>                                                                                     \
    struct tuple_size<__name##__size> : integral_constant<size_t, __size>                           \
    {};                                                                                             \
    template <size_t _Ip>                                                                           \
    struct tuple_element<_Ip, __name##__size>                                                       \
    {                                                                                               \
      static_assert(_Ip < __size, "tuple_element index out of range");                              \
      using type = __type;                                                                          \
    };                                                                                              \
                                                                                                    \
    _LIBCUDACXX_END_NAMESPACE_STD                                                                   \
                                                                                                    \
    template <>                                                                                     \
    struct std::tuple_size<__name##__size> : _CUDA_VSTD::tuple_size<__name##__size>                 \
    {};                                                                                             \
    template <size_t _Ip>                                                                           \
    struct std::tuple_element<_Ip, __name##__size> : _CUDA_VSTD::tuple_element<_Ip, __name##__size> \
    {};

#  define _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(__name, __type) \
    _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N(__name, __type, 1)  \
    _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N(__name, __type, 2)  \
    _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N(__name, __type, 3)  \
    _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N(__name, __type, 4)

_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(char, signed char)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(uchar, unsigned char)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(short, short)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(ushort, unsigned short)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(int, int)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(uint, unsigned int)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(long, long)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(ulong, unsigned long)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(longlong, long long)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(ulonglong, unsigned long long)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(float, float)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL(double, double)
_LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N(dim, unsigned int, 3)

#  undef _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL_N
#  undef _LIBCUDACXX_VEC_IMPL_TUPLE_PROTOCOL

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Ip, class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __cccl_cuda_vector_type_get_impl(_Tp&& __val) noexcept
  -> decltype(_CUDA_VSTD::forward_like<decltype(__val)>(__val.x))
{
  if constexpr (_Ip == 0)
  {
    return _CUDA_VSTD::forward_like<decltype(__val)>(__val.x);
  }
  else if constexpr (_Ip == 1)
  {
    return _CUDA_VSTD::forward_like<decltype(__val)>(__val.y);
  }
  else if constexpr (_Ip == 2)
  {
    return _CUDA_VSTD::forward_like<decltype(__val)>(__val.z);
  }
  else if constexpr (_Ip == 3)
  {
    return _CUDA_VSTD::forward_like<decltype(__val)>(__val.w);
  }
  else
  {
    _CCCL_UNREACHABLE();
  }
}

#  define _LIBCUDACXX_VEC_IMPL_GET(__name, __base_type)                                                      \
    template <size_t _Ip>                                                                                    \
    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr __base_type& get(__name& __val) noexcept               \
    {                                                                                                        \
      static_assert(_Ip < _CUDA_VSTD::tuple_size_v<__name>, "tuple_element index out of range");             \
      return _CUDA_VSTD::__cccl_cuda_vector_type_get_impl<_Ip>(_CUDA_VSTD::forward<decltype(__val)>(__val)); \
    }                                                                                                        \
    template <size_t _Ip>                                                                                    \
    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const __base_type& get(const __name& __val) noexcept   \
    {                                                                                                        \
      static_assert(_Ip < _CUDA_VSTD::tuple_size_v<__name>, "tuple_element index out of range");             \
      return _CUDA_VSTD::__cccl_cuda_vector_type_get_impl<_Ip>(_CUDA_VSTD::forward<decltype(__val)>(__val)); \
    }                                                                                                        \
    template <size_t _Ip>                                                                                    \
    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr __base_type&& get(__name&& __val) noexcept             \
    {                                                                                                        \
      static_assert(_Ip < _CUDA_VSTD::tuple_size_v<__name>, "tuple_element index out of range");             \
      return _CUDA_VSTD::__cccl_cuda_vector_type_get_impl<_Ip>(_CUDA_VSTD::forward<decltype(__val)>(__val)); \
    }                                                                                                        \
    template <size_t _Ip>                                                                                    \
    [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr const __base_type&& get(const __name&& __val) noexcept \
    {                                                                                                        \
      static_assert(_Ip < _CUDA_VSTD::tuple_size_v<__name>, "tuple_element index out of range");             \
      return _CUDA_VSTD::__cccl_cuda_vector_type_get_impl<_Ip>(_CUDA_VSTD::forward<decltype(__val)>(__val)); \
    }

#  define _LIBCUDACXX_VEC_IMPL_GET_1_TO_4(__name, __base_type) \
    _LIBCUDACXX_VEC_IMPL_GET(__name##1, __base_type)           \
    _LIBCUDACXX_VEC_IMPL_GET(__name##2, __base_type)           \
    _LIBCUDACXX_VEC_IMPL_GET(__name##3, __base_type)           \
    _LIBCUDACXX_VEC_IMPL_GET(__name##4, __base_type)

_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(char, signed char)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(uchar, unsigned char)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(short, short)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(ushort, unsigned short)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(int, int)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(uint, unsigned int)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(long, long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(ulong, unsigned long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(longlong, long long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(ulonglong, unsigned long long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(float, float)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(double, double)
_LIBCUDACXX_VEC_IMPL_GET(dim3, unsigned int)

_LIBCUDACXX_END_NAMESPACE_STD

_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(char, signed char)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(uchar, unsigned char)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(short, short)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(ushort, unsigned short)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(int, int)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(uint, unsigned int)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(long, long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(ulong, unsigned long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(longlong, long long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(ulonglong, unsigned long long)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(float, float)
_LIBCUDACXX_VEC_IMPL_GET_1_TO_4(double, double)
_LIBCUDACXX_VEC_IMPL_GET(dim3, unsigned int)

#  undef _LIBCUDACXX_VEC_IMPL_GET
#  undef _LIBCUDACXX_VEC_IMPL_GET_1_TO_4

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___VECTOR_TYPES_STRUCTURED_BINDINGS_H
