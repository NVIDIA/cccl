//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/experimental/__async/meta.cuh>
#include <cuda/experimental/__async/type_traits.cuh>
#include <cuda/experimental/__async/utility.cuh>

#include <new> // IWYU pragma: keep

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
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
class __variant_impl<__mindices<>>
{
public:
  template <class _Fn, class... _Us>
  _CCCL_HOST_DEVICE void __visit(_Fn&&, _Us&&...) const noexcept
  {}
};

template <size_t... _Idx, class... _Ts>
class __variant_impl<__mindices<_Idx...>, _Ts...>
{
  static constexpr size_t __max_size = __maximum({sizeof(_Ts)...});
  static_assert(__max_size != 0);
  size_t __index_{__npos};
  alignas(_Ts...) unsigned char __storage_[__max_size];

  template <size_t _Ny>
  using __at = __m_at_c<_Ny, _Ts...>;

  _CCCL_HOST_DEVICE void __destroy() noexcept
  {
    if (__index_ != __npos)
    {
      // make this local in case destroying the sub-object destroys *this
      const auto index = __async::__exchange(__index_, __npos);
      ((_Idx == index ? _CUDA_VSTD::destroy_at(static_cast<__at<_Idx>*>(__ptr())) : void(0)), ...);
    }
  }

public:
  _CUDAX_IMMOVABLE(__variant_impl);

  _CCCL_HOST_DEVICE __variant_impl() noexcept {}

  _CCCL_HOST_DEVICE ~__variant_impl()
  {
    __destroy();
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void* __ptr() noexcept
  {
    return __storage_;
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE size_t __index() const noexcept
  {
    return __index_;
  }

  template <class _Ty, class... _As>
  _CCCL_HOST_DEVICE _Ty& __emplace(_As&&... __as) //
    noexcept(__nothrow_constructible<_Ty, _As...>)
  {
    constexpr size_t __new_index = __async::__index_of<_Ty, _Ts...>();
    static_assert(__new_index != __npos, "Type not in variant");

    __destroy();
    _Ty* __value = ::new (__ptr()) _Ty{static_cast<_As&&>(__as)...};
    __index_     = __new_index;
    return *_CUDA_VSTD::launder(__value);
  }

  template <size_t _Ny, class... _As>
  _CCCL_HOST_DEVICE __at<_Ny>& __emplace_at(_As&&... __as) //
    noexcept(__nothrow_constructible<__at<_Ny>, _As...>)
  {
    static_assert(_Ny < sizeof...(_Ts), "variant index is too large");

    __destroy();
    __at<_Ny>* __value = ::new (__ptr()) __at<_Ny>{static_cast<_As&&>(__as)...};
    __index_           = _Ny;
    return *_CUDA_VSTD::launder(__value);
  }

  template <class _Fn, class... _As>
  _CCCL_HOST_DEVICE auto __emplace_from(_Fn&& __fn, _As&&... __as) //
    noexcept(__nothrow_callable<_Fn, _As...>) -> __call_result_t<_Fn, _As...>&
  {
    using __result_t             = __call_result_t<_Fn, _As...>;
    constexpr size_t __new_index = __async::__index_of<__result_t, _Ts...>();
    static_assert(__new_index != __npos, "_Type not in variant");

    __destroy();
    __result_t* __value = ::new (__ptr()) __result_t(static_cast<_Fn&&>(__fn)(static_cast<_As&&>(__as)...));
    __index_            = __new_index;
    return *_CUDA_VSTD::launder(__value);
  }

  template <class _Fn, class _Self, class... _As>
  _CCCL_HOST_DEVICE static void __visit(_Fn&& __fn, _Self&& __self, _As&&... __as) //
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
  _CCCL_HOST_DEVICE __at<_Ny>&& __get() && noexcept
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return static_cast<__at<_Ny>&&>(*static_cast<__at<_Ny>*>(__ptr()));
  }

  template <size_t _Ny>
  _CCCL_HOST_DEVICE __at<_Ny>& __get() & noexcept
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return *static_cast<__at<_Ny>*>(__ptr());
  }

  template <size_t _Ny>
  _CCCL_HOST_DEVICE const __at<_Ny>& __get() const& noexcept
  {
    _CCCL_ASSERT(_Ny == __index_, "");
    return *static_cast<const __at<_Ny>*>(__ptr());
  }
};

#if defined(_CCCL_COMPILER_MSVC)
template <class... _Ts>
struct __mk_variant_
{
  using __indices_t = __mmake_indices<sizeof...(_Ts)>;
  using type        = __variant_impl<__indices_t, _Ts...>;
};

template <class... _Ts>
using __variant = __t<__mk_variant_<_Ts...>>;
#else
template <class... _Ts>
using __variant = __variant_impl<__mmake_indices<sizeof...(_Ts)>, _Ts...>;
#endif

template <class... _Ts>
using __decayed_variant = __variant<__decay_t<_Ts>...>;
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
