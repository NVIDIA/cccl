//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_VARIANT_H
#define __CUDAX_ASYNC_DETAIL_VARIANT_H

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

template <class Idx, class... Ts>
class _variant_impl;

template <>
class _variant_impl<_mindices<>>
{
public:
  template <class Fn, class... Us>
  _CCCL_HOST_DEVICE void visit(Fn&&, Us&&...) const noexcept
  {}
};

template <size_t... Idx, class... Ts>
class _variant_impl<_mindices<Idx...>, Ts...>
{
  static constexpr size_t _max_size = _max({sizeof(Ts)...});
  static_assert(_max_size != 0);
  size_t _index{_npos};
  alignas(Ts...) unsigned char _storage[_max_size];

  template <size_t Ny>
  using _at = _m_at_c<Ny, Ts...>;

  _CCCL_HOST_DEVICE void _destroy() noexcept
  {
    if (_index != _npos)
    {
      // make this local in case destroying the sub-object destroys *this
      const auto index = __async::_exchange(_index, _npos);
      ((Idx == index ? _CUDA_VSTD::destroy_at(static_cast<_at<Idx>*>(_ptr())) : void(0)), ...);
    }
  }

public:
  _CUDAX_IMMOVABLE(_variant_impl);

  _CCCL_HOST_DEVICE _variant_impl() noexcept {}

  _CCCL_HOST_DEVICE ~_variant_impl()
  {
    _destroy();
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE void* _ptr() noexcept
  {
    return _storage;
  }

  _CCCL_HOST_DEVICE _CUDAX_ALWAYS_INLINE size_t index() const noexcept
  {
    return _index;
  }

  template <class Ty, class... As>
  _CCCL_HOST_DEVICE Ty& emplace(As&&... as) //
    noexcept(_nothrow_constructible<Ty, As...>)
  {
    constexpr size_t _new_index = __async::_index_of<Ty, Ts...>();
    static_assert(_new_index != _npos, "Type not in variant");

    _destroy();
    Ty* _value = ::new (_ptr()) Ty{static_cast<As&&>(as)...};
    _index     = _new_index;
    return *_CUDA_VSTD::launder(_value);
  }

  template <size_t Ny, class... As>
  _CCCL_HOST_DEVICE _at<Ny>& emplace_at(As&&... as) //
    noexcept(_nothrow_constructible<_at<Ny>, As...>)
  {
    static_assert(Ny < sizeof...(Ts), "variant index is too large");

    _destroy();
    _at<Ny>* _value = ::new (_ptr()) _at<Ny>{static_cast<As&&>(as)...};
    _index          = Ny;
    return *_CUDA_VSTD::launder(_value);
  }

  template <class Fn, class... As>
  _CCCL_HOST_DEVICE auto emplace_from(Fn&& fn, As&&... as) //
    noexcept(_nothrow_callable<Fn, As...>) -> _call_result_t<Fn, As...>&
  {
    using _result_t             = _call_result_t<Fn, As...>;
    constexpr size_t _new_index = __async::_index_of<_result_t, Ts...>();
    static_assert(_new_index != _npos, "Type not in variant");

    _destroy();
    _result_t* _value = ::new (_ptr()) _result_t(static_cast<Fn&&>(fn)(static_cast<As&&>(as)...));
    _index            = _new_index;
    return *_CUDA_VSTD::launder(_value);
  }

  template <class Fn, class Self, class... As>
  _CCCL_HOST_DEVICE static void visit(Fn&& fn, Self&& self, As&&... as) //
    noexcept((_nothrow_callable<Fn, As..., _copy_cvref_t<Self, Ts>> && ...))
  {
    // make this local in case destroying the sub-object destroys *this
    const auto index = self._index;
    _LIBCUDACXX_ASSERT(index != _npos, "");
    ((Idx == index ? static_cast<Fn&&>(fn)(static_cast<As&&>(as)..., static_cast<Self&&>(self).template get<Idx>())
                   : void()),
     ...);
  }

  template <size_t Ny>
  _CCCL_HOST_DEVICE _at<Ny>&& get() && noexcept
  {
    _LIBCUDACXX_ASSERT(Ny == _index, "");
    return static_cast<_at<Ny>&&>(*static_cast<_at<Ny>*>(_ptr()));
  }

  template <size_t Ny>
  _CCCL_HOST_DEVICE _at<Ny>& get() & noexcept
  {
    _LIBCUDACXX_ASSERT(Ny == _index, "");
    return *static_cast<_at<Ny>*>(_ptr());
  }

  template <size_t Ny>
  _CCCL_HOST_DEVICE const _at<Ny>& get() const& noexcept
  {
    _LIBCUDACXX_ASSERT(Ny == _index, "");
    return *static_cast<const _at<Ny>*>(_ptr());
  }
};

template <class... Ts>
using _variant = _variant_impl<_mmake_indices<sizeof...(Ts)>, Ts...>;

template <class... Ts>
using _decayed_variant = _variant<_decay_t<Ts>...>;
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
