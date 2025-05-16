//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_VARIANT
#define __CUDAX_ASYNC_DETAIL_VARIANT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <new> // IWYU pragma: keep

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/********************************************************************************/
/* NB: The variant type implemented here default-constructs into the valueless  */
/* state. This is different from std::variant which default-constructs into the */
/* first alternative. This is done to simplify the implementation and to avoid  */
/* the need for a default constructor for each alternative type.                */
/********************************************************************************/

template <class _Idx, class... _Ts>
class __variant_impl;

template <>
class __variant_impl<_CUDA_VSTD::index_sequence<>>
{
public:
  template <class _Fn, class... _Us>
  _CCCL_API void __visit(_Fn&&, _Us&&...) const noexcept
  {}
};

template <size_t... _Idx, class... _Ts>
class __variant_impl<_CUDA_VSTD::index_sequence<_Idx...>, _Ts...>
{
  static constexpr size_t __max_size = __maximum({sizeof(_Ts)...});
  static_assert(__max_size != 0);
  size_t __index_{__npos};
  alignas(_Ts...) unsigned char __storage_[__max_size];

  template <size_t _Ny>
  using __at _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_index_c<_Ny, _Ts...>;

  _CCCL_API void __destroy() noexcept
  {
    if (__index_ != __npos)
    {
      // make this local in case destroying the sub-object destroys *this
      const auto index = execution::__exchange(__index_, __npos);
      ((_Idx == index ? _CUDA_VSTD::destroy_at(static_cast<__at<_Idx>*>(__ptr())) : void(0)), ...);
    }
  }

public:
  __variant_impl(__variant_impl&&) = delete;

  _CCCL_API __variant_impl() noexcept {}

  _CCCL_API ~__variant_impl()
  {
    __destroy();
  }

  _CCCL_TRIVIAL_API void* __ptr() noexcept
  {
    return __storage_;
  }

  _CCCL_TRIVIAL_API size_t __index() const noexcept
  {
    return __index_;
  }

  template <class _Ty, class... _As>
  _CCCL_API auto __emplace(_As&&... __as) //
    noexcept(__nothrow_constructible<_Ty, _As...>) -> _Ty&
  {
    constexpr size_t __new_index = execution::__index_of<_Ty, _Ts...>();
    static_assert(__new_index != __npos, "Type not in variant");

    __destroy();
    _Ty* __value = ::new (__ptr()) _Ty{static_cast<_As&&>(__as)...};
    __index_     = __new_index;
    return *_CUDA_VSTD::launder(__value);
  }

  template <size_t _Ny, class... _As>
  _CCCL_API auto __emplace_at(_As&&... __as) //
    noexcept(__nothrow_constructible<__at<_Ny>, _As...>) -> __at<_Ny>&
  {
    static_assert(_Ny < sizeof...(_Ts), "variant index is too large");

    __destroy();
    __at<_Ny>* __value = ::new (__ptr()) __at<_Ny>{static_cast<_As&&>(__as)...};
    __index_           = _Ny;
    return *_CUDA_VSTD::launder(__value);
  }

  template <class _Fn, class... _As>
  _CCCL_API auto __emplace_from(_Fn&& __fn, _As&&... __as) //
    noexcept(__nothrow_callable<_Fn, _As...>) -> __call_result_t<_Fn, _As...>&
  {
    using __result_t _CCCL_NODEBUG_ALIAS = __call_result_t<_Fn, _As...>;
    constexpr size_t __new_index         = execution::__index_of<__result_t, _Ts...>();
    static_assert(__new_index != __npos, "_Type not in variant");

    __destroy();
    __result_t* __value = ::new (__ptr()) __result_t(static_cast<_Fn&&>(__fn)(static_cast<_As&&>(__as)...));
    __index_            = __new_index;
    return *_CUDA_VSTD::launder(__value);
  }

  template <class _Fn, class _Self, class... _As>
  _CCCL_API static void __visit(_Fn&& __fn, _Self&& __self, _As&&... __as) //
    noexcept((__nothrow_callable<_Fn, _As..., __copy_cvref_t<_Self, _Ts>> && ...))
  {
    // make this local in case destroying the sub-object destroys *this
    const auto index = __self.__index_;
    _CCCL_ASSERT(index != __npos, "");
    ((_Idx == index
        ? static_cast<_Fn&&>(__fn)(static_cast<_As&&>(__as)..., static_cast<_Self&&>(__self).template __get<_Idx>())
        : void()),
     ...);
  }

  template <size_t _Ny>
  _CCCL_API auto __get() && noexcept -> __at<_Ny>&&
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return static_cast<__at<_Ny>&&>(*static_cast<__at<_Ny>*>(__ptr()));
  }

  template <size_t _Ny>
  _CCCL_API auto __get() & noexcept -> __at<_Ny>&
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return *static_cast<__at<_Ny>*>(__ptr());
  }

  template <size_t _Ny>
  _CCCL_API auto __get() const& noexcept -> const __at<_Ny>&
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return *static_cast<const __at<_Ny>*>(__ptr());
  }
};

#if _CCCL_COMPILER(MSVC)
template <class... _Ts>
struct __mk_variant_
{
  using __indices_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>;
  using type _CCCL_NODEBUG_ALIAS        = __variant_impl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __variant _CCCL_NODEBUG_ALIAS = typename __mk_variant_<_Ts...>::type;
#else
template <class... _Ts>
using __variant _CCCL_NODEBUG_ALIAS = __variant_impl<_CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>, _Ts...>;
#endif

template <class... _Ts>
using __decayed_variant _CCCL_NODEBUG_ALIAS = __variant<__decay_t<_Ts>...>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif
