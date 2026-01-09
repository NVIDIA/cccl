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
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/construct_at.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/monostate.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>

#include <exception> // IWYU pragma: keep

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
/********************************************************************************/
/* NB: The variant type implemented here default-constructs into the valueless  */
/* state. This is different from std::variant which default-constructs into the */
/* first alternative. This is done to simplify the implementation and to avoid  */
/* the need for a default constructor for each alternative type.                */
/********************************************************************************/

using __monostate = ::cuda::std::monostate;

template <size_t _Idx, bool _Check = true, class _CvVariant>
[[nodiscard]]
_CCCL_TRIVIAL_API constexpr auto&& __variant_get(_CvVariant&& __var) noexcept
{
  using __variant_t = ::cuda::std::remove_reference_t<_CvVariant>;
  using __element_t = typename __variant_t::template __at<_Idx>;
  using __result_t  = ::cuda::std::__copy_cvref_t<_CvVariant, __element_t>;
  if constexpr (_Check)
  {
    _CCCL_ASSERT(__var.__index() == _Idx, "variant index mismatch");
  }
  return static_cast<__result_t&&>(*static_cast<__element_t*>(__var.__ptr()));
}

struct __visit_t
{
  _CCCL_EXEC_CHECK_DISABLE
  template <size_t... _Idx, class _Fn, class _CvVariant, class... _As>
  _CCCL_API static void
  __visit(::cuda::std::index_sequence<_Idx...>*,
          const size_t __index,
          _Fn&& __fn,
          _CvVariant&& __var,
          _As&&... __as) //
    noexcept(noexcept((
      (_Idx == __index ? declval<_Fn>()(declval<_As>()..., execution::__variant_get<_Idx, false>(declval<_CvVariant>()))
                       : void()),
      ...)))
  {
    _CCCL_ASSERT(__index != __npos, "cannot visit a stateless variant");
    // Use a fold expression to avoid the need for a loop.
    ((_Idx == __index
        ? static_cast<_Fn&&>(
            __fn)(static_cast<_As&&>(__as)..., execution::__variant_get<_Idx, false>(static_cast<_CvVariant&&>(__var)))
        : void()),
     ...);
  }

  template <class _Fn, class _CvVariant, class... _As>
  _CCCL_TRIVIAL_API void operator()(_Fn&& __fn, _CvVariant&& __var, _As&&... __as) const noexcept(
    noexcept(__visit_t::__visit(__var.__indices(), size_t(), declval<_Fn>(), declval<_CvVariant>(), declval<_As>()...)))
  {
    __visit_t::__visit(
      __var.__indices(),
      __var.__index(),
      static_cast<_Fn&&>(__fn),
      static_cast<_CvVariant&&>(__var),
      static_cast<_As&&>(__as)...);
  }
};

_CCCL_GLOBAL_CONSTANT __visit_t __visit{};

namespace __detail
{
struct __destroy_fn
{
  template <class _Ty>
  _CCCL_API void operator()(_Ty& __ty) const noexcept
  {
    ::cuda::std::__destroy_at(::cuda::std::addressof(__ty));
  }
};

struct __move_to_fn
{
  template <class _Ty>
  _CCCL_API void operator()(_Ty&& __from) const noexcept
  {
    ::cuda::std::__construct_at(static_cast<decay_t<_Ty>*>(__to), static_cast<_Ty&&>(__from));
  }

  void* __to;
};
} // namespace __detail

template <class... _Ts>
class __variant
{
public:
  template <size_t _Ny>
  using __at _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_index_c<_Ny, _Ts...>;

  _CCCL_API __variant() noexcept {}

  _CCCL_TEMPLATE(class...)
  _CCCL_REQUIRES((::cuda::std::move_constructible<_Ts> && ...))
  __variant(__variant&& __other) noexcept
  {
    if (__other.__index_ != __npos)
    {
      __visit_t::__visit(
        __indices(), __other.__index(), __detail::__move_to_fn{__ptr()}, static_cast<__variant&&>(__other));
      __index_ = __other.__index_;
      __other.__reset();
    }
  }

  _CCCL_API ~__variant()
  {
    __reset();
  }

  [[nodiscard]] _CCCL_TRIVIAL_API void* __ptr() noexcept
  {
    return __storage_;
  }

  [[nodiscard]] _CCCL_TRIVIAL_API size_t __index() const noexcept
  {
    return __index_;
  }

  template <int = 0, class _Ty>
  _CCCL_API auto __emplace(_Ty&& __value) noexcept(__nothrow_decay_copyable<_Ty>) -> decay_t<_Ty>&
  {
    return __emplace<decay_t<_Ty>>(static_cast<_Ty&&>(__value));
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Ty, class... _As>
  _CCCL_API auto __emplace(_As&&... __as) noexcept(__nothrow_constructible<_Ty, _As...>) -> _Ty&
  {
    constexpr size_t __new_index = __index_of<_Ty, _Ts...>();
    static_assert(__new_index != __npos, "Type not in variant");

    __reset();
    _Ty* __value = ::new (__ptr()) _Ty{static_cast<_As&&>(__as)...};
    __index_     = __new_index;
    return *::cuda::std::launder(__value);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t _Ny, class... _As>
  _CCCL_API auto __emplace_at(_As&&... __as) noexcept(__nothrow_constructible<__at<_Ny>, _As...>) -> __at<_Ny>&
  {
    static_assert(_Ny < sizeof...(_Ts), "variant index is too large");

    __reset();
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
    constexpr size_t __new_index         = __index_of<__result_t, _Ts...>();
    static_assert(__new_index != __npos, "_Type not in variant");

    __reset();
    __result_t* __value = ::new (__ptr()) __result_t(static_cast<_Fn&&>(__fn)(static_cast<_As&&>(__as)...));
    __index_            = __new_index;
    return *::cuda::std::launder(__value);
  }

  _CCCL_API void __reset() noexcept
  {
    if (__index_ != __npos)
    {
      // We must set the index to __npos *before* destroying the value on the off chance that
      // destroying the active value might cause the destruction of *this. But then, we must
      // tell the __visit function what the old index was so it can destroy the correct type.
      const auto __index = ::cuda::std::exchange(__index_, __npos);
      __visit_t::__visit(__indices(), __index, __detail::__destroy_fn{}, *this);
    }
  }

private:
  friend struct __visit_t;

  template <size_t, bool _Check, class _CvVariant>
  friend _CCCL_API constexpr auto&& __variant_get(_CvVariant&& __var) noexcept;

  _CCCL_TRIVIAL_API static constexpr auto __indices() noexcept -> ::cuda::std::make_index_sequence<sizeof...(_Ts)>*
  {
    return nullptr;
  }

  size_t __index_{__npos};
  alignas(_Ts...) unsigned char __storage_[__maximum({sizeof(_Ts)...})];
};

template <>
class __variant<>
{
public:
  _CCCL_HIDE_FROM_ABI __variant() noexcept = default;

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr size_t __index() noexcept
  {
    return __npos;
  }

private:
  friend struct __visit_t;

  _CCCL_TRIVIAL_API static constexpr auto __indices() noexcept -> ::cuda::std::index_sequence<>*
  {
    return nullptr;
  }
};

template <class... _Ts>
using __nullable_variant _CCCL_NODEBUG_ALIAS = __variant<__monostate, _Ts...>;

template <class... _Ts>
using __decayed_variant _CCCL_NODEBUG_ALIAS = __variant<decay_t<_Ts>...>;

template <class... _Ts>
using __nullable_decayed_variant _CCCL_NODEBUG_ALIAS = __variant<__monostate, decay_t<_Ts>...>;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_VARIANT
