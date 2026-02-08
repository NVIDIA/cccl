// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FUNCTIONAL_INVOKE_H
#define _CUDA_STD___FUNCTIONAL_INVOKE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/reference_wrapper.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_core_convertible.h>
#include <cuda/std/__type_traits/is_member_function_pointer.h>
#include <cuda/std/__type_traits/is_member_object_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/nat.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __any
{
  _CCCL_API inline __any(...);
};

template <class _DecayedFp>
struct __member_pointer_class_type
{};

template <class _Ret, class _ClassType>
struct __member_pointer_class_type<_Ret _ClassType::*>
{
  using type = _ClassType;
};

template <class _DecayedFp>
using __member_pointer_class_type_t = typename __member_pointer_class_type<_DecayedFp>::type;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = decay_t<_A0>,
          class _ClassT  = __member_pointer_class_type_t<_DecayFp>>
using __enable_if_bullet1 = enable_if_t<is_member_function_pointer_v<_DecayFp> && is_base_of_v<_ClassT, _DecayA0>>;

template <class _Fp, class _A0, class _DecayFp = decay_t<_Fp>, class _DecayA0 = decay_t<_A0>>
using __enable_if_bullet2 =
  enable_if_t<is_member_function_pointer_v<_DecayFp> && __is_cuda_std_reference_wrapper_v<_DecayA0>>;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = decay_t<_A0>,
          class _ClassT  = __member_pointer_class_type_t<_DecayFp>>
using __enable_if_bullet3 = enable_if_t<is_member_function_pointer_v<_DecayFp> && !is_base_of_v<_ClassT, _DecayA0>
                                        && !__is_cuda_std_reference_wrapper_v<_DecayA0>>;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = decay_t<_A0>,
          class _ClassT  = __member_pointer_class_type_t<_DecayFp>>
using __enable_if_bullet4 = enable_if_t<is_member_object_pointer_v<_DecayFp> && is_base_of_v<_ClassT, _DecayA0>>;

template <class _Fp, class _A0, class _DecayFp = decay_t<_Fp>, class _DecayA0 = decay_t<_A0>>
using __enable_if_bullet5 =
  enable_if_t<is_member_object_pointer_v<_DecayFp> && __is_cuda_std_reference_wrapper_v<_DecayA0>>;

template <class _Fp,
          class _A0,
          class _DecayFp = decay_t<_Fp>,
          class _DecayA0 = decay_t<_A0>,
          class _ClassT  = __member_pointer_class_type_t<_DecayFp>>
using __enable_if_bullet6 = enable_if_t<is_member_object_pointer_v<_DecayFp> && !is_base_of_v<_ClassT, _DecayA0>
                                        && !__is_cuda_std_reference_wrapper_v<_DecayA0>>;

// __invoke forward declarations

// fall back - none of the bullets

template <class... _Args>
_CCCL_API inline __nat __invoke(__any, _Args&&... __args);

// bullets 1, 2 and 3

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class... _Args, class = __enable_if_bullet1<_Fp, _A0>>
_CCCL_API constexpr decltype((::cuda::std::declval<_A0>()
                              .*::cuda::std::declval<_Fp>())(::cuda::std::declval<_Args>()...))
__invoke(_Fp&& __f,
         _A0&& __a0,
         _Args&&... __args) noexcept(noexcept((static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...)))
{
  return (static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class... _Args, class = __enable_if_bullet2<_Fp, _A0>>
_CCCL_API constexpr decltype((::cuda::std::declval<_A0>().get()
                              .*::cuda::std::declval<_Fp>())(::cuda::std::declval<_Args>()...))
__invoke(_Fp&& __f, _A0&& __a0, _Args&&... __args) noexcept(noexcept((__a0.get().*__f)(static_cast<_Args&&>(__args)...)))
{
  return (__a0.get().*__f)(static_cast<_Args&&>(__args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class... _Args, class = __enable_if_bullet3<_Fp, _A0>>
_CCCL_API constexpr decltype(((*::cuda::std::declval<_A0>())
                              .*::cuda::std::declval<_Fp>())(::cuda::std::declval<_Args>()...))
__invoke(_Fp&& __f,
         _A0&& __a0,
         _Args&&... __args) noexcept(noexcept(((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...)))
{
  return ((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...);
}

// bullets 4, 5 and 6

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class = __enable_if_bullet4<_Fp, _A0>>
_CCCL_API constexpr decltype(::cuda::std::declval<_A0>().*::cuda::std::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0) noexcept(noexcept(static_cast<_A0&&>(__a0).*__f))
{
  return static_cast<_A0&&>(__a0).*__f;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class = __enable_if_bullet5<_Fp, _A0>>
_CCCL_API constexpr decltype(::cuda::std::declval<_A0>().get().*::cuda::std::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0) noexcept(noexcept(__a0.get().*__f))
{
  return __a0.get().*__f;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class _A0, class = __enable_if_bullet6<_Fp, _A0>>
_CCCL_API constexpr decltype((*::cuda::std::declval<_A0>()).*::cuda::std::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0) noexcept(noexcept((*static_cast<_A0&&>(__a0)).*__f))
{
  return (*static_cast<_A0&&>(__a0)).*__f;
}

// bullet 7

_CCCL_EXEC_CHECK_DISABLE
template <class _Fp, class... _Args>
_CCCL_API constexpr decltype(::cuda::std::declval<_Fp>()(::cuda::std::declval<_Args>()...))
__invoke(_Fp&& __f, _Args&&... __args) noexcept(noexcept(static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...)))
{
  return static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...);
}

// __is_invocable
template <class _Fp, class... _Args>
using __invoke_result_t =
  decltype(::cuda::std::__invoke(::cuda::std::declval<_Fp>(), ::cuda::std::declval<_Args>()...));

template <class _Fp, class... _Args>
_CCCL_CONCEPT __is_invocable =
  _CCCL_REQUIRES_EXPR((_Fp, variadic _Args))(requires(!is_same_v<__nat, __invoke_result_t<_Fp, _Args...>>));

template <class _Ret, class _Fp, class... _Args>
_CCCL_CONCEPT __is_invocable_r = _CCCL_REQUIRES_EXPR((_Ret, _Fp, variadic _Args))(
  requires(__is_invocable<_Fp, _Args...>),
  requires((is_void_v<_Ret> || __is_core_convertible<__invoke_result_t<_Fp, _Args...>, _Ret>::value)));

template <class _Fp, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT invoke_result //
    : public enable_if<__is_invocable<_Fp, _Args...>, __invoke_result_t<_Fp, _Args...>>
{
#if _CCCL_CUDA_COMPILER(NVCC) && defined(__CUDACC_EXTENDED_LAMBDA__) && !_CCCL_DEVICE_COMPILATION()
#  if _CCCL_CUDACC_BELOW(12, 3)
  static_assert(!__nv_is_extended_device_lambda_closure_type(remove_cvref_t<_Fp>),
                "Attempt to use an extended __device__ lambda in a context "
                "that requires querying its return type in host code. Use a "
                "named function object, an extended __host__ __device__ lambda, or "
                "cuda::proclaim_return_type instead.");
#  else // ^^^ _CCCL_CUDACC_BELOW(12, 3) ^^^ / vvv _CCCL_CUDACC_AT_LEAST(12, 3) vvv
  static_assert(
    !__nv_is_extended_device_lambda_closure_type(remove_cvref_t<_Fp>)
      || __nv_is_extended_host_device_lambda_closure_type(remove_cvref_t<_Fp>)
      || __nv_is_extended_device_lambda_with_preserved_return_type(remove_cvref_t<_Fp>),
    "Attempt to use an extended __device__ lambda in a context "
    "that requires querying its return type in host code. Use a "
    "named function object, an extended __host__ __device__ lambda, "
    "cuda::proclaim_return_type, or an extended __device__ lambda "
    "with a trailing return type instead ([] __device__ (...) -> RETURN_TYPE {...}).");
#  endif // _CCCL_CUDACC_AT_LEAST(12, 3)
#endif
};

// is_invocable

template <class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_invocable : bool_constant<__is_invocable<_Fn, _Args...>>
{};

template <class _Ret, class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_invocable_r : bool_constant<__is_invocable_r<_Ret, _Fn, _Args...>>
{};

template <class _Fn, class... _Args>
inline constexpr bool is_invocable_v = __is_invocable<_Fn, _Args...>;

template <class _Ret, class _Fn, class... _Args>
inline constexpr bool is_invocable_r_v = __is_invocable_r<_Ret, _Fn, _Args...>;

// is_nothrow_invocable

template <class _Tp>
_CCCL_API constexpr void __cccl_test_noexcept_conversion(_Tp) noexcept;

template <bool _IsInvocable, bool _IsCVVoid, class _Ret, class _Fp, class... _Args>
inline constexpr bool __nothrow_invocable_r_imp = false;

template <class _Ret, class _Fp, class... _Args>
inline constexpr bool __nothrow_invocable_r_imp<true, false, _Ret, _Fp, _Args...> =
  noexcept(::cuda::std::__cccl_test_noexcept_conversion<_Ret>(
    ::cuda::std::__invoke(declval<_Fp>(), ::cuda::std::declval<_Args>()...)));

template <class _Ret, class _Fp, class... _Args>
inline constexpr bool __nothrow_invocable_r_imp<true, true, _Ret, _Fp, _Args...> =
  noexcept(::cuda::std::__invoke(::cuda::std::declval<_Fp>(), ::cuda::std::declval<_Args>()...));

template <class _Fp, class... _Args>
inline constexpr bool is_nothrow_invocable_v =
  __nothrow_invocable_r_imp<__is_invocable<_Fp, _Args...>, true, void, _Fp, _Args...>;

template <class _Ret, class _Fp, class... _Args>
inline constexpr bool is_nothrow_invocable_r_v =
  __nothrow_invocable_r_imp<__is_invocable_r<_Ret, _Fp, _Args...>, is_void_v<_Ret>, _Ret, _Fp, _Args...>;

template <class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT is_nothrow_invocable : bool_constant<is_nothrow_invocable_v<_Fn, _Args...>>
{};

template <class _Ret, class _Fn, class... _Args>
struct _CCCL_TYPE_VISIBILITY_DEFAULT
is_nothrow_invocable_r : bool_constant<is_nothrow_invocable_r_v<_Ret, _Fn, _Args...>>
{};

// Not going directly through __invoke_result_t because we want the additional device lambda checks in invoke_result
template <class _Fn, class... _Args>
using invoke_result_t = typename invoke_result<_Fn, _Args...>::type;

template <class _Fn, class... _Args>
_CCCL_API constexpr invoke_result_t<_Fn, _Args...>
invoke(_Fn&& __f, _Args&&... __args) noexcept(is_nothrow_invocable_v<_Fn, _Args...>)
{
  return ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)...);
}

_CCCL_TEMPLATE(class _Ret, class _Fn, class... _Args)
_CCCL_REQUIRES(is_invocable_r_v<_Ret, _Fn, _Args...>)
_CCCL_API constexpr _Ret invoke_r(_Fn&& __f, _Args&&... __args) noexcept(is_nothrow_invocable_r_v<_Ret, _Fn, _Args...>)
{
  if constexpr (is_void_v<_Ret>)
  {
    ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)...);
  }
  else
  {
    return ::cuda::std::__invoke(::cuda::std::forward<_Fn>(__f), ::cuda::std::forward<_Args>(__args)...);
  }
}

/// The type of intermediate accumulator (according to P2322R6)
template <typename Invocable, typename InputT, typename InitT = InputT>
using __accumulator_t = decay_t<invoke_result_t<Invocable, InitT, InputT>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FUNCTIONAL_INVOKE_H
