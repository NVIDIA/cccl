//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_STORE_H
#define _CUDA_STD___SIMD_STORE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/is_valid_alignment.h>
#include <cuda/__memory/ptr_rebind.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_HOST_DEVICE_API constexpr void
__check_store_preconditions(_Up* const __ptr, flags<_Flags...>, const __simd_size_type __count = 1) noexcept
{
  static_assert(__is_vectorizable_v<_Up>, "cuda::std::simd::store: range_value_t<R> must be a vectorizable type");

  static_assert(__explicitly_convertible_to<_Tp, _Up>,
                "cuda::std::simd::store: value_type must satisfy explicitly-convertible-to<range_value_t<R>>");

  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Tp, _Up>,
                "cuda::std::simd::store: Conversion from value_type to range_value_t<R> is not value-preserving; use "
                "flag_convert");

  _CCCL_ASSERT(__count == 0 || __ptr != nullptr, "cuda::std::simd::store: pointer is nullptr");
  ::cuda::std::simd::__assert_load_store_alignment<basic_vec<_Tp, _Abi>, _Up, _Flags...>(__ptr);
}

// [simd.loadstore] helper: core partial store to pointer + count + mask
template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_HOST_DEVICE_API constexpr void __partial_store_to_ptr(
  const basic_vec<_Tp, _Abi>& __v,
  _Up* const __ptr,
  const __simd_size_type __count,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __flags = {}) noexcept
{
  ::cuda::std::simd::__check_store_preconditions<_Tp, _Abi>(__ptr, __flags, __count);
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
  {
    if (__mask[__i] && __i < __count)
    {
      __ptr[__i] = static_cast<_Up>(__v[__i]);
    }
  }
}

template <typename _Tp, typename _Abi, typename _Up, typename... _Flags>
_CCCL_HOST_DEVICE_API constexpr void
__full_store_to_ptr(const basic_vec<_Tp, _Abi>& __v, _Up* const __ptr, flags<_Flags...> __flags) noexcept
{
  ::cuda::std::simd::__check_store_preconditions<_Tp, _Abi>(__ptr, __flags);
  using __vec_t                     = basic_vec<_Tp, _Abi>;
  constexpr auto __simd_size        = __vec_t::__size;
  constexpr bool __has_aligned_flag = __has_aligned_flag_v<_Flags...>;

  if constexpr (__has_aligned_flag || __has_overaligned_flag_v<_Flags...>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __base_alignment = __has_aligned_flag ? alignment_v<__vec_t, _Up> : alignof(_Up);
      constexpr auto __ptr_alignment  = ::cuda::std::max(__base_alignment, __overaligned_value_v<_Flags...>);
      constexpr auto __data_size      = __simd_size * sizeof(_Up);

      _Up __tmp[__simd_size]{};
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
      {
        __tmp[__i] = static_cast<_Up>(__v[__i]);
      }
      // vectorized store to pointer
      if constexpr (__is_cuda_vectorizable_v<_Up> && __simd_size > 1 && __ptr_alignment >= __data_size
                    && ::cuda::__is_valid_alignment(__data_size))
      {
        struct alignas(__data_size) __aligned_t
        {
          char __data[__data_size];
        };
        // nvcc performance bug: memcpy to pointer could not be vectorized
        const auto __aligned_ptr = ::cuda::ptr_rebind<__aligned_t>(__ptr);
        __aligned_t __data{};
        ::cuda::std::memcpy(&__data, &__tmp, sizeof(__tmp));
        *::cuda::std::assume_aligned<__ptr_alignment>(__aligned_ptr) = __data;
      }
      // rely on compiler vectorization
      else
      {
        const auto __aligned_ptr = ::cuda::std::assume_aligned<__ptr_alignment>(__ptr);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
        {
          __aligned_ptr[__i] = __tmp[__i];
        }
      }
      return;
    }
  }
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::__partial_store_to_ptr(__v, __ptr, __simd_size, __true_mask, __flags);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] partial_store

// partial_store: range, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> //
                 _CCCL_AND ::cuda::std::ranges::sized_range<_Range> //
                   _CCCL_AND indirectly_writable<::cuda::std::ranges::iterator_t<_Range>,
                                                 ::cuda::std::ranges::range_value_t<_Range>> //
                     _CCCL_AND __explicitly_convertible_to<_Tp, ::cuda::std::ranges::range_value_t<_Range>>)
_CCCL_HOST_DEVICE_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __r,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  const auto __range_size = ::cuda::std::ranges::size(__r);
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__range_size),
               "cuda::std::simd::partial_store: range size out of range");
  const auto __size = static_cast<__simd_size_type>(__range_size);
  const auto __ptr  = ::cuda::std::ranges::data(__r);

  ::cuda::std::simd::__partial_store_to_ptr(__v, __ptr, __size, __mask, __f);
}

// partial_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> //
                 _CCCL_AND ::cuda::std::ranges::sized_range<_Range> //
                   _CCCL_AND indirectly_writable<::cuda::std::ranges::iterator_t<_Range>,
                                                 ::cuda::std::ranges::range_value_t<_Range>> //
                     _CCCL_AND __explicitly_convertible_to<_Tp, ::cuda::std::ranges::range_value_t<_Range>>)
_CCCL_HOST_DEVICE_API constexpr void
partial_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::partial_store(__v, ::cuda::std::forward<_Range>(__r), __true_mask, __f);
}

// partial_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> //
                 _CCCL_AND indirectly_writable<_Ip, iter_value_t<_Ip>> //
                   _CCCL_AND __explicitly_convertible_to<_Tp, iter_value_t<_Ip>>)
_CCCL_HOST_DEVICE_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__n), "cuda::std::simd::partial_store: n out of range");
  const auto __size = static_cast<__simd_size_type>(__n);

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::to_address(__first), __size, __mask, __f);
}

// partial_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> //
                 _CCCL_AND indirectly_writable<_Ip, iter_value_t<_Ip>> //
                   _CCCL_AND __explicitly_convertible_to<_Tp, iter_value_t<_Ip>>)
_CCCL_HOST_DEVICE_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::partial_store(__v, __first, __n, __true_mask, __f);
}

// partial_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> //
                 _CCCL_AND indirectly_writable<_Ip, iter_value_t<_Ip>> //
                   _CCCL_AND __explicitly_convertible_to<_Tp, iter_value_t<_Ip>> //
                     _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
_CCCL_HOST_DEVICE_API constexpr void partial_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  const auto __distance = ::cuda::std::distance(__first, __last);
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__distance),
               "cuda::std::simd::partial_store: distance(first, last) out of range");
  const auto __size = static_cast<__simd_size_type>(__distance);

  ::cuda::std::simd::__partial_store_to_ptr(__v, ::cuda::std::to_address(__first), __size, __mask, __f);
}

// partial_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> //
                 _CCCL_AND indirectly_writable<_Ip, iter_value_t<_Ip>> //
                   _CCCL_AND __explicitly_convertible_to<_Tp, iter_value_t<_Ip>> //
                     _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
_CCCL_HOST_DEVICE_API constexpr void
partial_store(const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const _Sp __last, flags<_Flags...> __f = {})
{
  constexpr auto __true_mask = typename basic_vec<_Tp, _Abi>::mask_type(true);

  ::cuda::std::simd::partial_store(__v, __first, __last, __true_mask, __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] unchecked_store

// unchecked_store: range, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
_CCCL_HOST_DEVICE_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  _Range&& __r,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= __simd_size,
                  "cuda::std::simd::unchecked_store: requires ::cuda::std::ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __simd_size),
               "cuda::std::simd::unchecked_store: requires ::cuda::std::ranges::size(r) >= V::size()");
  const auto __ptr = ::cuda::std::ranges::data(__r);

  ::cuda::std::simd::__partial_store_to_ptr(__v, __ptr, __simd_size, __mask, __f);
}

// unchecked_store: range, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
_CCCL_HOST_DEVICE_API constexpr void
unchecked_store(const basic_vec<_Tp, _Abi>& __v, _Range&& __r, flags<_Flags...> __f = {})
{
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= basic_vec<_Tp, _Abi>::__size,
                  "cuda::std::simd::unchecked_store: requires ::cuda::std::ranges::size(r) >= V::size()");
  }
  [[maybe_unused]] constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __simd_size),
               "cuda::std::simd::unchecked_store: requires ::cuda::std::ranges::size(r) >= V::size()");

  ::cuda::std::simd::__full_store_to_ptr(__v, ::cuda::std::ranges::data(__r), __f);
}

// unchecked_store: iterator + count, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
_CCCL_HOST_DEVICE_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const iter_difference_t<_Ip> __n,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::size();
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __simd_size),
               "cuda::std::simd::unchecked_store: requires n >= V::size()");
  const auto __ptr = ::cuda::std::to_address(__first);

  ::cuda::std::simd::__partial_store_to_ptr(__v, __ptr, __simd_size, __mask, __f);
}

// unchecked_store: iterator + count, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
_CCCL_HOST_DEVICE_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  [[maybe_unused]] constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __simd_size),
               "cuda::std::simd::unchecked_store: requires n >= V::size()");
  const auto __ptr = ::cuda::std::to_address(__first);

  ::cuda::std::simd::__full_store_to_ptr(__v, __ptr, __f);
}

// unchecked_store: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
_CCCL_HOST_DEVICE_API constexpr void unchecked_store(
  const basic_vec<_Tp, _Abi>& __v,
  const _Ip __first,
  const _Sp __last,
  const typename basic_vec<_Tp, _Abi>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __simd_size),
               "cuda::std::simd::unchecked_store: requires distance(first, last) >= V::size()");
  const auto __ptr = ::cuda::std::to_address(__first);

  ::cuda::std::simd::__partial_store_to_ptr(__v, __ptr, __simd_size, __mask, __f);
}

// unchecked_store: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
_CCCL_HOST_DEVICE_API constexpr void
unchecked_store(const basic_vec<_Tp, _Abi>& __v, const _Ip __first, const _Sp __last, flags<_Flags...> __f = {})
{
  [[maybe_unused]] constexpr auto __simd_size = basic_vec<_Tp, _Abi>::__size;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __simd_size),
               "cuda::std::simd::unchecked_store: requires distance(first, last) >= V::size()");
  const auto __ptr = ::cuda::std::to_address(__first);

  ::cuda::std::simd::__full_store_to_ptr(__v, __ptr, __f);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_STORE_H
