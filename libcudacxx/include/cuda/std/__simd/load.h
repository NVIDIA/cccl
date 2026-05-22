//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_LOAD_H
#define _CUDA_STD___SIMD_LOAD_H

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
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__memory/assume_aligned.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/data.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/concepts.h>
#include <cuda/std/__simd/flag.h>
#include <cuda/std/__simd/utility.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.loadstore] helper: resolves default V template parameter for load functions
// When _Vp = void (default), resolves to basic_vec<_Up>; otherwise uses the explicit _Vp
template <typename _Vp, typename _Up>
using __load_vec_t = conditional_t<is_void_v<_Vp>, basic_vec<_Up>, _Vp>;

template <typename _Result, typename _Up, typename... _Flags>
_CCCL_HOST_DEVICE_API constexpr void
__check_load_preconditions(const _Up* __ptr, flags<_Flags...>, const __simd_size_type __count = 1) noexcept
{
  using __value_t = typename _Result::value_type;
  static_assert(same_as<remove_cvref_t<_Result>, _Result>, "V must not be a reference or cv-qualified type");

  static_assert(__is_vectorizable_v<__value_t> && __is_enabled_abi_v<typename _Result::abi_type>,
                "cuda::std::simd::load: V must be an enabled specialization of basic_vec");
  static_assert(__is_vectorizable_v<_Up>, "range_value_t<R> must be a vectorizable type");

  static_assert(__explicitly_convertible_to<_Up, __value_t>,
                "cuda::std::simd::load: range_value_t<R> must satisfy explicitly-convertible-to<value_type>");

  static_assert(__has_convert_flag_v<_Flags...> || __is_value_preserving_v<_Up, __value_t>,
                "cuda::std::simd::load: Conversion from range_value_t<R> to value_type is not value-preserving; use "
                "flag_convert");

  _CCCL_ASSERT(__count == 0 || __ptr != nullptr, "cuda::std::simd::load: range data is nullptr");
  ::cuda::std::simd::__assert_load_store_alignment<_Result, _Up, _Flags...>(__ptr);
}

// [simd.loadstore] helper: core partial load from pointer + count + mask
template <typename _Result, typename _Up, typename... _Flags>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Result __partial_load_from_ptr(
  const _Up* __ptr,
  const __simd_size_type __count,
  const typename _Result::mask_type& __mask,
  flags<_Flags...> __flags = {}) noexcept
{
  using __value_t = typename _Result::value_type;
  ::cuda::std::simd::__check_load_preconditions<_Result>(__ptr, __flags, __count);
  constexpr auto __simd_size = _Result::__size;

  _Result __result;
  _CCCL_PRAGMA_UNROLL_FULL()
  for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
  {
    const auto __value = (__mask[__i] && __i < __count) ? static_cast<__value_t>(__ptr[__i]) : __value_t{};
    __result.__set(__i, __value);
  }
  return __result;
}

template <typename _Result, typename _Up, typename... _Flags>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr _Result
__full_load_from_ptr(const _Up* __ptr, const typename _Result::mask_type& __mask, flags<_Flags...> __flags) noexcept
{
  ::cuda::std::simd::__check_load_preconditions<_Result>(__ptr, __flags);
  constexpr bool __has_aligned_flag = __has_aligned_flag_v<_Flags...>;

  if constexpr (__has_aligned_flag || __has_overaligned_flag_v<_Flags...>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      // minimum condition for pointer alignment
      constexpr auto __base_alignment = __has_aligned_flag ? alignment_v<_Result, _Up> : alignof(_Up);
      constexpr auto __ptr_alignment  = ::cuda::std::max(__base_alignment, __overaligned_value_v<_Flags...>);
      constexpr auto __simd_size      = _Result::__size;
      constexpr auto __data_size      = __simd_size * sizeof(_Up);

      _Up __tmp[__simd_size]{};
      // vectorized load from pointer
      if constexpr (__is_cuda_vectorizable_v<_Up> && __simd_size > 1 && __ptr_alignment >= __data_size
                    && ::cuda::__is_valid_alignment(__data_size))
      {
        struct alignas(__data_size) __aligned_t
        {
          _Up __data[__simd_size];
        };
        // nvcc performance bug: memcpy from pointer could not be vectorized
        const auto __aligned_ptr = ::cuda::ptr_rebind<__aligned_t>(__ptr);
        const auto __data        = *::cuda::std::assume_aligned<__ptr_alignment>(__aligned_ptr);
        ::cuda::std::memcpy(&__tmp, &__data, sizeof(__tmp));
      }
      // rely on compiler vectorization
      else
      {
        const auto __aligned_ptr = ::cuda::std::assume_aligned<__ptr_alignment>(__ptr);
        _CCCL_PRAGMA_UNROLL_FULL()
        for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
        {
          __tmp[__i] = __aligned_ptr[__i];
        }
      }
      using __value_t = typename _Result::value_type;
      _Result __result;
      _CCCL_PRAGMA_UNROLL_FULL()
      for (__simd_size_type __i = 0; __i < __simd_size; ++__i)
      {
        const auto __value = (!__mask[__i]) ? __value_t{} : static_cast<__value_t>(__tmp[__i]);
        __result.__set(__i, __value);
      }
      return __result;
    }
  }
  return ::cuda::std::simd::__partial_load_from_ptr<_Result>(__ptr, _Result::__size, __mask, __flags);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] partial_load

// partial_load: range, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>
partial_load(_Range&& __r,
             const typename __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>::mask_type& __mask,
             flags<_Flags...> __f = {})
{
  using __result_t        = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  const auto __range_size = ::cuda::std::ranges::size(__r);
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__range_size),
               "cuda::std::simd::partial_load: range size out of range");
  const auto __size = static_cast<__simd_size_type>(__range_size);

  return ::cuda::std::simd::__partial_load_from_ptr<__result_t>(::cuda::std::ranges::data(__r), __size, __mask, __f);
}

// partial_load: range, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>
partial_load(_Range&& __r, flags<_Flags...> __f = {})
{
  using __result_t           = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  constexpr auto __true_mask = typename __result_t::mask_type(true);

  return ::cuda::std::simd::partial_load<_Vp>(::cuda::std::forward<_Range>(__r), __true_mask, __f);
}

// partial_load: iterator + count, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> partial_load(
  const _Ip __first,
  const iter_difference_t<_Ip> __n,
  const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__n), "cuda::std::simd::partial_load: n out of range");
  const auto __ptr  = ::cuda::std::to_address(__first);
  const auto __size = static_cast<__simd_size_type>(__n);

  return ::cuda::std::simd::__partial_load_from_ptr<__result_t>(__ptr, __size, __mask, __f);
}

// partial_load: iterator + count, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
partial_load(const _Ip __first, const iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  using __result_t           = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  constexpr auto __true_mask = typename __result_t::mask_type(true);

  return ::cuda::std::simd::partial_load<_Vp>(__first, __n, __true_mask, __f);
}

// partial_load: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> partial_load(
  const _Ip __first,
  const _Sp __last,
  const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using __result_t      = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  const auto __ptr      = ::cuda::std::to_address(__first);
  const auto __distance = ::cuda::std::distance(__first, __last);
  _CCCL_ASSERT(::cuda::std::in_range<__simd_size_type>(__distance),
               "cuda::std::simd::partial_load: distance(first, last) out of range");
  const auto __size = static_cast<__simd_size_type>(__distance);

  return ::cuda::std::simd::__partial_load_from_ptr<__result_t>(__ptr, __size, __mask, __f);
}

// partial_load: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
partial_load(const _Ip __first, const _Sp __last, flags<_Flags...> __f = {})
{
  using __result_t           = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  constexpr auto __true_mask = typename __result_t::mask_type(true);

  return ::cuda::std::simd::partial_load<_Vp>(__first, __last, __true_mask, __f);
}

//----------------------------------------------------------------------------------------------------------------------
// [simd.loadstore] unchecked_load

// unchecked_load: range, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>
unchecked_load(_Range&& __r,
               const typename __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>::mask_type& __mask,
               flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  if constexpr (__has_static_size<_Range>)
  {
    static_assert(__static_range_size_v<_Range> >= __result_t::__size,
                  "cuda::std::simd::unchecked_load: requires ::cuda::std::ranges::size(r) >= V::size()");
  }
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::ranges::size(__r), __result_t::__size),
               "cuda::std::simd::unchecked_load: requires ::cuda::std::ranges::size(r) >= V::size()");

  return ::cuda::std::simd::__full_load_from_ptr<__result_t>(::cuda::std::ranges::data(__r), __mask, __f);
}

// unchecked_load: range, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Range, typename... _Flags)
_CCCL_REQUIRES(::cuda::std::ranges::contiguous_range<_Range> _CCCL_AND ::cuda::std::ranges::sized_range<_Range>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>
unchecked_load(_Range&& __r, flags<_Flags...> __f = {})
{
  using __result_t           = __load_vec_t<_Vp, ::cuda::std::ranges::range_value_t<_Range>>;
  constexpr auto __true_mask = typename __result_t::mask_type(true);

  return ::cuda::std::simd::unchecked_load<_Vp>(::cuda::std::forward<_Range>(__r), __true_mask, __f);
}

// unchecked_load: iterator + count, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> unchecked_load(
  const _Ip __first,
  const iter_difference_t<_Ip> __n,
  const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__n, __result_t::__size),
               "cuda::std::simd::unchecked_load: requires n >= V::size()");
  const auto __ptr = ::cuda::std::to_address(__first);

  return ::cuda::std::simd::__full_load_from_ptr<__result_t>(__ptr, __mask, __f);
}

// unchecked_load: iterator + count, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
unchecked_load(const _Ip __first, const iter_difference_t<_Ip> __n, flags<_Flags...> __f = {})
{
  using __result_t           = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  constexpr auto __true_mask = typename __result_t::mask_type(true);

  return ::cuda::std::simd::unchecked_load<_Vp>(__first, __n, __true_mask, __f);
}

// unchecked_load: iterator + sentinel, masked
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>> unchecked_load(
  const _Ip __first,
  const _Sp __last,
  const typename __load_vec_t<_Vp, iter_value_t<_Ip>>::mask_type& __mask,
  flags<_Flags...> __f = {})
{
  using __result_t = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(::cuda::std::distance(__first, __last), __result_t::__size),
               "unchecked_load requires distance(first, last) >= V::size()");

  return ::cuda::std::simd::__full_load_from_ptr<__result_t>(::cuda::std::to_address(__first), __mask, __f);
}

// unchecked_load: iterator + sentinel, no mask
_CCCL_TEMPLATE(typename _Vp = void, typename _Ip, typename _Sp, typename... _Flags)
_CCCL_REQUIRES(contiguous_iterator<_Ip> _CCCL_AND sized_sentinel_for<_Sp, _Ip>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __load_vec_t<_Vp, iter_value_t<_Ip>>
unchecked_load(const _Ip __first, const _Sp __last, flags<_Flags...> __f = {})
{
  using __result_t           = __load_vec_t<_Vp, iter_value_t<_Ip>>;
  constexpr auto __true_mask = typename __result_t::mask_type(true);

  return ::cuda::std::simd::unchecked_load<_Vp>(__first, __last, __true_mask, __f);
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_LOAD_H
