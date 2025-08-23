//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_VARIANT
#define __CUDAX_EXECUTION_VARIANT

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/assert.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <exception> // IWYU pragma: keep
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
class __variant_impl<::cuda::std::index_sequence<>>
{
public:
  template <class _Fn, class... _Us>
  _CCCL_API static void __visit(_Fn&&, _Us&&...) noexcept
  {
    _CCCL_ASSERT(false, "cannot visit a stateless variant");
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr size_t __index() noexcept
  {
    return __npos;
  }
};

template <size_t... _Idx, class... _Ts>
class __variant_impl<::cuda::std::index_sequence<_Idx...>, _Ts...>
{
  static constexpr size_t __max_size = __maximum({sizeof(_Ts)...});
  static_assert(__max_size != 0);
  size_t __index_{__npos};
  alignas(_Ts...) unsigned char __storage_[__max_size];

  template <size_t _Ny>
  using __at _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_index_c<_Ny, _Ts...>;

  _CCCL_API void __destroy() noexcept
  {
    if (__index_ != __npos)
    {
      // make this local in case destroying the sub-object destroys *this
      const auto index = execution::__exchange(__index_, __npos);
      ((_Idx == index ? ::cuda::std::destroy_at(static_cast<__at<_Idx>*>(__ptr())) : void(0)), ...);
    }
  }

public:
  _CCCL_API __variant_impl() noexcept {}

  _CCCL_TEMPLATE(class...)
  _CCCL_REQUIRES((::cuda::std::move_constructible<_Ts> && ...))
  __variant_impl(__variant_impl&& __other) noexcept
  {
    if (__other.__index_ != __npos)
    {
      ((_Idx == __other.__index_
          ? static_cast<void>(__emplace<__at<_Idx>>(static_cast<__variant_impl&&>(__other).template __get<_Idx>()))
          : void(0)),
       ...);
      __index_ = __other.__index_;
      __other.__destroy();
    }
  }

  _CCCL_API ~__variant_impl()
  {
    __destroy();
  }

  [[nodiscard]] _CCCL_NODEBUG_API void* __ptr() noexcept
  {
    return __storage_;
  }

  [[nodiscard]] _CCCL_NODEBUG_API size_t __index() const noexcept
  {
    return __index_;
  }

  template <class _Ty>
  _CCCL_API auto __emplace(_Ty&& __value) noexcept(__nothrow_decay_copyable<_Ty>) -> decay_t<_Ty>&
  {
    return __emplace<decay_t<_Ty>, _Ty>(static_cast<_Ty&&>(__value));
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Ty, class... _As>
  _CCCL_API auto __emplace(_As&&... __as) noexcept(__nothrow_constructible<_Ty, _As...>) -> _Ty&
  {
    constexpr size_t __new_index = execution::__index_of<_Ty, _Ts...>();
    static_assert(__new_index != __npos, "Type not in variant");

    __destroy();
    _Ty* __value = ::new (__ptr()) _Ty{static_cast<_As&&>(__as)...};
    __index_     = __new_index;
    return *::cuda::std::launder(__value);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t _Ny, class... _As>
  _CCCL_API auto __emplace_at(_As&&... __as) noexcept(__nothrow_constructible<__at<_Ny>, _As...>) -> __at<_Ny>&
  {
    static_assert(_Ny < sizeof...(_Ts), "variant index is too large");

    __destroy();
    __at<_Ny>* __value = ::new (__ptr()) __at<_Ny>{static_cast<_As&&>(__as)...};
    __index_           = _Ny;
    return *::cuda::std::launder(__value);
  }

  _CCCL_EXEC_CHECK_DISABLE
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
    return *::cuda::std::launder(__value);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Fn, class _Self, class... _As>
  _CCCL_API static void __visit(_Fn&& __fn, _Self&& __self, _As&&... __as) //
    noexcept((__nothrow_callable<_Fn, _As..., ::cuda::std::__copy_cvref_t<_Self, _Ts>> && ...))
  {
    // make this local in case destroying the sub-object destroys *this
    const auto index = __self.__index_;
    _CCCL_ASSERT(index != __npos, "cannot visit a stateless variant");
    ((_Idx == index
        ? static_cast<_Fn&&>(__fn)(static_cast<_As&&>(__as)..., static_cast<_Self&&>(__self).template __get<_Idx>())
        : void()),
     ...);
  }

  template <size_t _Ny>
  [[nodiscard]] _CCCL_API auto __get() && noexcept -> __at<_Ny>&&
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return static_cast<__at<_Ny>&&>(*static_cast<__at<_Ny>*>(__ptr()));
  }

  template <size_t _Ny>
  [[nodiscard]] _CCCL_API auto __get() & noexcept -> __at<_Ny>&
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return *static_cast<__at<_Ny>*>(__ptr());
  }

  template <size_t _Ny>
  [[nodiscard]] _CCCL_API auto __get() const& noexcept -> const __at<_Ny>&
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return *static_cast<const __at<_Ny>*>(__ptr());
  }
};

template <class... _Ts>
struct __variant : __variant_impl<::cuda::std::index_sequence_for<_Ts...>, _Ts...>
{};

template <class... _Ts>
using __decayed_variant _CCCL_NODEBUG_ALIAS = __variant<decay_t<_Ts>...>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_VARIANT
