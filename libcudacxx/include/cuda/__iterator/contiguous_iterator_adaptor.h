//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ITERATOR_CONTIGUOUS_ITERATOR_ADAPTOR_H
#define _CUDA___ITERATOR_CONTIGUOUS_ITERATOR_ADAPTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/iterator.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief True when `_Iter` is an iterator adaptor whose `base()` returns the iterator it adapts, so not e.g. the
//! counter of a counting iterator.
template <class _Iter>
_CCCL_CONCEPT __iter_has_member_base = _CCCL_REQUIRES_EXPR((_Iter), const _Iter& __iter)(
  requires(::cuda::std::__can_to_address<::cuda::std::remove_cvref_t<decltype(__iter.base())>>
           || ::cuda::std::input_or_output_iterator<::cuda::std::remove_cvref_t<decltype(__iter.base())>>));

//! @brief True when `_Iter` is an iterator adaptor that can rebuild itself over the storage at a `_Tp*`.
template <class _Iter, class _Tp>
_CCCL_CONCEPT __iter_has_member_rebind =
  _CCCL_REQUIRES_EXPR((_Iter, _Tp), const _Iter& __iter, _Tp* __ptr)((__iter.__rebind(__ptr)));

//! @brief Peel iterator adaptors until a contiguous iterator or an iterator that adapts no other iterator remains.
//! @param[in] __iter The iterator to unwrap.
//! @return The address of the storage of a contiguous iterator. Otherwise the unwrapped base iterator of an iterator
//! adaptor, or `__iter` when it is not an iterator adaptor.
template <class _Iter>
[[nodiscard]] _CCCL_API constexpr auto __unwrap_iterator_adaptor(const _Iter& __iter)
{
  if constexpr (::cuda::std::__can_to_address<_Iter>)
  { // Support standard contiguous iterators
    return ::cuda::std::to_address(__iter);
  }
  else if constexpr (__iter_has_member_base<_Iter>)
  {
    return ::cuda::__unwrap_iterator_adaptor(__iter.base());
  }
  else
  {
    return __iter;
  }
}

//! @brief The iterator that remains after peeling iterator adaptors from `_Iter`.
template <class _Iter>
using __unwrapped_iterator_t = decltype(::cuda::__unwrap_iterator_adaptor(::cuda::std::declval<_Iter>()));

//! @brief The value type of the iterator that remains after peeling iterator adaptors from `_Iter`.
template <class _Iter>
using __unwrapped_iter_value_t = ::cuda::std::iter_value_t<__unwrapped_iterator_t<_Iter>>;

//! @brief True when `_Iter` is a raw pointer or opts into contiguous addressing through `pointer_traits::to_address`.
//! A contiguous iterator tag is not enough: `cuda::std::to_address` also accepts iterators such as `reverse_iterator`
//! through `operator->`.
template <class _Iter>
inline constexpr bool __iter_has_contiguous_storage =
  ::cuda::std::is_pointer_v<_Iter> || ::cuda::std::__has_cuda_std_to_address<_Iter>;

//! @brief True when `_Iter` can be rebuilt over the storage at a `_Tp*`, because it is a contiguous iterator or an
//! iterator adaptor that can rebuild itself over it.
template <class _Iter, class _Tp>
_CCCL_CONCEPT __iter_can_rebind =
  (__iter_has_contiguous_storage<_Iter> && ::cuda::std::is_convertible_v<_Tp*, __unwrapped_iterator_t<_Iter>>)
  || __iter_has_member_rebind<_Iter, _Tp>;

//! @brief Rebuild `__iter` over the storage at `__ptr`.
//! @param[in] __iter The iterator to rebuild.
//! @param[in] __ptr The address of the storage that replaces the storage `__iter` reads from.
//! @return A pointer to `__ptr` with the qualifiers of the address of a contiguous iterator. Otherwise a copy of the
//! iterator adaptor over its rebuilt base iterator.
_CCCL_TEMPLATE(class _Iter, class _Tp)
_CCCL_REQUIRES(__iter_can_rebind<_Iter, _Tp>)
[[nodiscard]] _CCCL_API constexpr auto __rebind_iterator_adaptor([[maybe_unused]] const _Iter& __iter, _Tp* __ptr)
{
  if constexpr (__iter_has_member_rebind<_Iter, _Tp>)
  {
    return __iter.__rebind(__ptr);
  }
  else
  {
    return static_cast<__unwrapped_iterator_t<_Iter>>(__ptr);
  }
}

//! @brief True when `_Iter` is an iterator adaptor over contiguous storage, which unwraps to a pointer and can be
//! rebuilt over a copy of the storage behind it.
template <class _Iter>
_CCCL_CONCEPT __iter_can_unwrap = _CCCL_REQUIRES_EXPR((_Iter)) //
  (requires(::cuda::std::is_pointer_v<__unwrapped_iterator_t<_Iter>>),
   requires(__iter_has_member_rebind<_Iter, __unwrapped_iter_value_t<_Iter>>));

template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_impl =
  __iter_has_contiguous_storage<_Iter> && ::cuda::std::is_pointer_v<__unwrapped_iterator_t<_Iter>>;

template <class _Fn, class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_impl<::cuda::transform_iterator<_Fn, _Iter>> =
  __is_contiguous_iterator_or_adaptor_impl<::cuda::std::remove_cvref_t<_Iter>>;

//! @brief True when `_Iter` is contiguous storage, or an adaptor whose stored iterator is.
//! @tparam _Iter The iterator type to query. cv-qualifiers and references are ignored.
template <class _Iter>
inline constexpr bool __is_contiguous_iterator_or_adaptor_v =
  __is_contiguous_iterator_or_adaptor_impl<::cuda::std::remove_cvref_t<_Iter>>;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_CONTIGUOUS_ITERATOR_ADAPTOR_H
