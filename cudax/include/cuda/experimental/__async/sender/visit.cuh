//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_VISIT
#define __CUDAX_ASYNC_DETAIL_VISIT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/copy_cvref.h>

#include <cuda/experimental/__async/sender/type_traits.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{

// Specialize this for each sender type that can be used to initialize a structured binding.
template <class _Sndr>
inline constexpr size_t structured_binding_size = static_cast<size_t>(-1);

template <size_t _Arity>
struct __sender_type_cannot_be_used_to_initialize_a_structured_binding;

template <size_t _Arity>
extern __sender_type_cannot_be_used_to_initialize_a_structured_binding<_Arity> __unpack;

#define _CCCL_BIND_CHILD(_Ord) , _CCCL_PP_CAT(__child, _Ord)
#define _CCCL_FWD_CHILD(_Ord)  , _CCCL_FWD_LIKE(_Sndr, _CCCL_PP_CAT(__child, _Ord))
#define _CCCL_FWD_LIKE(_X, _Y) static_cast<_CUDA_VSTD::__copy_cvref_t<_X&&, decltype(_Y)>>(_Y)

#define _CCCL_UNPACK_SENDER(_Arity)                                                                             \
  template <>                                                                                                   \
  [[maybe_unused]]                                                                                              \
  _CCCL_GLOBAL_CONSTANT auto __unpack<2 + _Arity> =                                                             \
    [](auto& __visitor, auto&& __sndr, auto& __context) -> decltype(auto) {                                     \
    using _Sndr                                                      = decltype(__sndr);                        \
    auto&& [__tag, __data _CCCL_PP_REPEAT(_Arity, _CCCL_BIND_CHILD)] = static_cast<_Sndr&&>(__sndr);            \
    return __visitor(__context, __tag, _CCCL_FWD_LIKE(_Sndr, __data) _CCCL_PP_REPEAT(_Arity, _CCCL_FWD_CHILD)); \
  }

_CCCL_UNPACK_SENDER(0);
_CCCL_UNPACK_SENDER(1);
_CCCL_UNPACK_SENDER(2);
_CCCL_UNPACK_SENDER(3);
_CCCL_UNPACK_SENDER(4);
_CCCL_UNPACK_SENDER(5);
_CCCL_UNPACK_SENDER(6);
_CCCL_UNPACK_SENDER(7);

#undef _CCCL_FWD_LIKE
#undef _CCCL_FWD_CHILD
#undef _CCCL_BIND_CHILD

[[maybe_unused]]
_CCCL_GLOBAL_CONSTANT auto visit = [](auto& __visitor, auto&& sndr, auto& __context) -> decltype(auto) {
  using _Sndr = __decay_t<decltype(sndr)>;
  return __unpack<structured_binding_size<_Sndr>>(__visitor, static_cast<decltype(sndr)&&>(sndr), __context);
};

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif // __CUDAX_ASYNC_DETAIL_VISIT
