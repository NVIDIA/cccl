//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___WARP_WARP_SHUFFLE_H
#define _CUDA___WARP_WARP_SHUFFLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  if __cccl_ptx_isa >= 600

#    include <cuda/__cmath/ceil_div.h>
#    include <cuda/__cmath/pow2.h>
#    include <cuda/__ptx/instructions/get_sreg.h>
#    include <cuda/__ptx/instructions/shfl_sync.h>
#    include <cuda/__type_traits/is_trivially_copyable.h>
#    include <cuda/std/__memory/addressof.h>
#    include <cuda/std/__type_traits/enable_if.h>
#    include <cuda/std/__type_traits/integral_constant.h>
#    include <cuda/std/__type_traits/is_array.h>
#    include <cuda/std/__type_traits/is_default_constructible.h>
#    include <cuda/std/__type_traits/is_pointer.h>
#    include <cuda/std/__type_traits/make_nbit_int.h>
#    include <cuda/std/__type_traits/num_bits.h>
#    include <cuda/std/cstdint>

#    include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

template <typename _Tp>
struct warp_shuffle_result
{
  _Tp data;
  bool pred;

  template <typename _Up = _Tp>
  [[nodiscard]] _CCCL_DEVICE_API operator ::cuda::std::enable_if_t<!::cuda::std::is_array_v<_Up>, _Up>() const
  {
    return data;
  }
};

template <typename _Up>
inline constexpr bool __is_8bit_16bit_shuffle_path_v =
  (sizeof(_Up) == sizeof(::cuda::std::uint8_t) || sizeof(_Up) == sizeof(::cuda::std::uint16_t))
  && !::cuda::std::is_array_v<_Up>;

template <typename _Up>
inline constexpr bool __is_64bit_shuffle_path_v =
  sizeof(_Up) == sizeof(::cuda::std::uint64_t) && !::cuda::std::is_array_v<_Up>;

template <typename _Up>
inline constexpr bool __is_64bit_array_shuffle_path_v =
  sizeof(_Up) == sizeof(::cuda::std::uint64_t) && ::cuda::std::is_array_v<_Up>;

#    if _CCCL_HAS_INT128() && __cccl_ptx_isa >= 830
template <typename _Up>
inline constexpr bool __is_128bit_shuffle_path_v = sizeof(_Up) == sizeof(__uint128_t) && !::cuda::std::is_array_v<_Up>;

template <int _Width>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<__uint128_t>
__warp_shuffle_up_128(__uint128_t __data, int __delta, ::cuda::std::uint32_t __lane_mask)
{
  ::cuda::std::uint64_t __lo;
  ::cuda::std::uint64_t __hi;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("mov.b128 {%0, %1}, %2;" : "=l"(__lo), "=l"(__hi) : "q"(__data));),
    (__lo = static_cast<::cuda::std::uint64_t>(__data); __hi = static_cast<::cuda::std::uint64_t>(__data >> 64);))
  __lo = ::__shfl_up_sync(__lane_mask, __lo, __delta, _Width);
  __hi = ::__shfl_up_sync(__lane_mask, __hi, __delta, _Width);

  const auto __lane = ::cuda::ptx::get_sreg_laneid();
  const auto __pred = (__lane & (_Width - 1)) >= static_cast<::cuda::std::uint32_t>(__delta);
  __uint128_t __shuffled;
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("mov.b128 %0, {%1, %2};" : "=q"(__shuffled) : "l"(__lo), "l"(__hi));),
               (__shuffled = (static_cast<__uint128_t>(__hi) << 64) | static_cast<__uint128_t>(__lo);))
  return warp_shuffle_result<__uint128_t>{__shuffled, __pred};
}

template <int _Width>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<__uint128_t>
__warp_shuffle_down_128(__uint128_t __data, int __delta, ::cuda::std::uint32_t __lane_mask)
{
  ::cuda::std::uint64_t __lo;
  ::cuda::std::uint64_t __hi;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("mov.b128 {%0, %1}, %2;" : "=l"(__lo), "=l"(__hi) : "q"(__data));),
    (__lo = static_cast<::cuda::std::uint64_t>(__data); __hi = static_cast<::cuda::std::uint64_t>(__data >> 64);))
  __lo = ::__shfl_down_sync(__lane_mask, __lo, __delta, _Width);
  __hi = ::__shfl_down_sync(__lane_mask, __hi, __delta, _Width);

  const auto __lane = ::cuda::ptx::get_sreg_laneid();
  const auto __pred = (__lane & (_Width - 1)) + static_cast<::cuda::std::uint32_t>(__delta) < _Width;
  __uint128_t __shuffled;
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("mov.b128 %0, {%1, %2};" : "=q"(__shuffled) : "l"(__lo), "l"(__hi));),
               (__shuffled = (static_cast<__uint128_t>(__hi) << 64) | static_cast<__uint128_t>(__lo);))
  return warp_shuffle_result<__uint128_t>{__shuffled, __pred};
}
#    endif // _CCCL_HAS_INT128() && __cccl_ptx_isa >= 830

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_idx(
  const _Tp& __data,
  const int __src_lane,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  static_assert(::cuda::std::is_default_constructible_v<_Tp>, "_Tp must be default constructible");
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = ::cuda::std::is_same_v<_Up, void*> || ::cuda::std::is_same_v<_Up, const void*>;
  static_assert(!::cuda::std::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(::cuda::is_power_of_two(_Width) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  static_assert(::cuda::is_trivially_copyable_v<_Up>, "_Up must be trivially copyable");

  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else if constexpr (__is_8bit_16bit_shuffle_path_v<_Up>)
  {
    using __unsigned_t                 = ::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Up>>;
    const ::cuda::std::uint32_t __word = ::cuda::std::bit_cast<__unsigned_t>(__data); // upcast to 32 bits
    const auto __shuffled              = ::__shfl_sync(__lane_mask, __word, __src_lane, _Width);
    const auto __narrowed              = static_cast<__unsigned_t>(__shuffled);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), true};
  }
  else if constexpr (__is_64bit_shuffle_path_v<_Up>)
  {
    const auto __dword    = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__data);
    const auto __shuffled = ::__shfl_sync(__lane_mask, __dword, __src_lane, _Width);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__shuffled), true};
  }
  else if constexpr (__is_64bit_array_shuffle_path_v<_Up>)
  {
    ::cuda::std::uint64_t __dword;
    ::cuda::std::memcpy(&__dword, &__data, sizeof(__dword));
    const auto __shuffled = ::__shfl_sync(__lane_mask, __dword, __src_lane, _Width);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__shuffled), true};
  }
  else
  {
    constexpr int __ratio          = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
    constexpr auto __clamp_segmask = (_Width - 1u) | ((__warp_size - _Width) << 8);
    bool __pred;
    ::cuda::std::uint32_t __array[__ratio]{}; // zero-initialize -> avoid undetermined values progatation (reg pressure)
    ::cuda::std::memcpy(
      static_cast<void*>(__array), static_cast<const void*>(::cuda::std::addressof(__data)), sizeof(_Up));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_idx(__array[i], __pred, __src_lane, __clamp_segmask, __lane_mask);
    }
    warp_shuffle_result<_Up> __result;
    __result.pred = true;
    ::cuda::std::memcpy(
      static_cast<void*>(::cuda::std::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_idx(const _Tp& __data, const int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_idx(__data, __src_lane, 0xFFFFFFFFu, __width);
}

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Tp> warp_shuffle_up(
  const _Tp& __data,
  const int __delta,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  static_assert(::cuda::std::is_default_constructible_v<_Tp>, "_Tp must be default constructible");
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = ::cuda::std::is_same_v<_Up, void*> || ::cuda::std::is_same_v<_Up, const void*>;
  static_assert(!::cuda::std::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(::cuda::is_power_of_two(_Width) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  static_assert(::cuda::is_trivially_copyable_v<_Up>, "_Up must be trivially copyable");
  _CCCL_ASSERT(__delta >= 0 && __delta < _Width, "delta must be in the range [0, _Width)");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __delta, &__pred1),
                                                           "all active lanes must have the same delta");))

  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else
  {
    constexpr int __ratio = ::cuda::ceil_div(sizeof(_Up), sizeof(uint32_t));
    auto __clamp_segmask  = (__warp_size - _Width) << 8;
    bool __pred;

    if constexpr (__is_8bit_16bit_shuffle_path_v<_Up>)
    {
      using __unsigned_t                 = ::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Up>>;
      const ::cuda::std::uint32_t __word = ::cuda::std::bit_cast<__unsigned_t>(__data); // upcast to 32 bits
      const auto __shuffled = ::cuda::ptx::shfl_sync_up(__word, __pred, __delta, __clamp_segmask, __lane_mask);
      const auto __narrowed = static_cast<__unsigned_t>(__shuffled);
      return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), __pred};
    }
    else if constexpr (__is_64bit_shuffle_path_v<_Up>)
    {
      auto __word = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__data);
      ::cuda::std::uint32_t __lo;
      ::cuda::std::uint32_t __hi;
      asm("mov.b64 {%0, %1}, %2;" : "=r"(__lo), "=r"(__hi) : "l"(__word));
      __lo = ::__shfl_up_sync(__lane_mask, __lo, __delta, _Width);
      __hi = ::cuda::ptx::shfl_sync_up(__hi, __pred, __delta, __clamp_segmask, __lane_mask);
      asm("mov.b64 %0, {%1, %2};" : "=l"(__word) : "r"(__lo), "r"(__hi));
      return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__word), __pred};
    }
#    if _CCCL_HAS_INT128() && __cccl_ptx_isa >= 830
    else if constexpr (__is_128bit_shuffle_path_v<_Up>)
    {
      const auto __word     = ::cuda::std::bit_cast<__uint128_t>(__data);
      const auto __shuffled = ::cuda::device::__warp_shuffle_up_128<_Width>(__word, __delta, __lane_mask);
      return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__shuffled.data), __shuffled.pred};
    }
#    endif // _CCCL_HAS_INT128() && __cccl_ptx_isa >= 830
    else
    {
      ::cuda::std::uint32_t __array[__ratio]{}; // zero-initialize -> avoid undetermined values progatation (reg
                                                // pressure)
      ::cuda::std::memcpy(
        static_cast<void*>(__array), static_cast<const void*>(::cuda::std::addressof(__data)), sizeof(_Up));

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < __ratio; ++i)
      {
        __array[i] = ::cuda::ptx::shfl_sync_up(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
      }
      warp_shuffle_result<_Up> __result;
      __result.pred = __pred;
      ::cuda::std::memcpy(
        static_cast<void*>(::cuda::std::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
      return __result;
    }
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_up(const _Tp& __data, int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_up(__data, __src_lane, 0xFFFFFFFFu, __width);
}

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_down(
  const _Tp& __data,
  const int __delta,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  static_assert(::cuda::std::is_default_constructible_v<_Tp>, "_Tp must be default constructible");
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = ::cuda::std::is_same_v<_Up, void*> || ::cuda::std::is_same_v<_Up, const void*>;
  static_assert(!::cuda::std::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(::cuda::is_power_of_two(_Width) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  static_assert(::cuda::is_trivially_copyable_v<_Up>, "_Up must be trivially copyable");
  _CCCL_ASSERT(__delta >= 0 && __delta < _Width, "__delta must be in the range [0, _Width)");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __delta, &__pred1),
                                                           "all active lanes must have the same delta");))

  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else
  {
    constexpr int __ratio          = ::cuda::ceil_div(sizeof(_Up), sizeof(::cuda::std::uint32_t));
    constexpr auto __clamp_segmask = (_Width - 1u) | ((__warp_size - _Width) << 8);
    bool __pred;

    if constexpr (__is_8bit_16bit_shuffle_path_v<_Up>)
    {
      using __unsigned_t                 = ::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Up>>;
      const ::cuda::std::uint32_t __word = ::cuda::std::bit_cast<__unsigned_t>(__data); // upcast to 32 bits
      const auto __shuffled = ::cuda::ptx::shfl_sync_down(__word, __pred, __delta, __clamp_segmask, __lane_mask);
      const auto __narrowed = static_cast<__unsigned_t>(__shuffled);
      return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), __pred};
    }
    else if constexpr (__is_64bit_shuffle_path_v<_Up>)
    {
      auto __word = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__data);
      ::cuda::std::uint32_t __lo;
      ::cuda::std::uint32_t __hi;
      asm("mov.b64 {%0, %1}, %2;" : "=r"(__lo), "=r"(__hi) : "l"(__word));
      __lo = ::__shfl_down_sync(__lane_mask, __lo, __delta, _Width);
      __hi = ::cuda::ptx::shfl_sync_down(__hi, __pred, __delta, __clamp_segmask, __lane_mask);
      asm("mov.b64 %0, {%1, %2};" : "=l"(__word) : "r"(__lo), "r"(__hi));
      return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__word), __pred};
    }
#    if _CCCL_HAS_INT128() && __cccl_ptx_isa >= 830
    else if constexpr (__is_128bit_shuffle_path_v<_Up>)
    {
      const auto __word     = ::cuda::std::bit_cast<__uint128_t>(__data);
      const auto __shuffled = ::cuda::device::__warp_shuffle_down_128<_Width>(__word, __delta, __lane_mask);
      return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__shuffled.data), __shuffled.pred};
    }
#    endif // _CCCL_HAS_INT128() && __cccl_ptx_isa >= 830
    else
    {
      ::cuda::std::uint32_t __array[__ratio]{}; // zero-initialize -> avoid undetermined values progatation (reg
                                                // pressure)
      ::cuda::std::memcpy(
        static_cast<void*>(__array), static_cast<const void*>(::cuda::std::addressof(__data)), sizeof(_Up));

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < __ratio; ++i)
      {
        __array[i] = ::cuda::ptx::shfl_sync_down(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
      }
      warp_shuffle_result<_Up> __result;
      __result.pred = __pred;
      ::cuda::std::memcpy(
        static_cast<void*>(::cuda::std::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
      return __result;
    }
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Tp>
warp_shuffle_down(const _Tp& __data, const int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_down(__data, __src_lane, 0xFFFFFFFFu, __width);
}

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_xor(
  const _Tp& __data,
  const int __xor_mask,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  static_assert(::cuda::std::is_default_constructible_v<_Tp>, "_Tp must be default constructible");
  constexpr auto __warp_size   = 32u;
  constexpr bool __is_void_ptr = ::cuda::std::is_same_v<_Up, void*> || ::cuda::std::is_same_v<_Up, const void*>;
  static_assert(!::cuda::std::is_pointer_v<_Up> || __is_void_ptr,
                "non-void pointers are not allowed to prevent bug-prone code");
  static_assert(::cuda::is_power_of_two(_Width) && _Width >= 1 && _Width <= __warp_size,
                "_Width must be a power of 2 and less or equal to the warp size");
  static_assert(::cuda::is_trivially_copyable_v<_Up>, "_Up must be trivially copyable");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __xor_mask, &__pred1),
                                                           "all active lanes must have the same xor_mask");))

  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__xor_mask == 0, "delta must be 0 when Width == 1");
    return warp_shuffle_result<_Up>{__data, true};
  }
  else if constexpr (__is_8bit_16bit_shuffle_path_v<_Up>)
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
    using __unsigned_t                 = ::cuda::std::__make_nbit_uint_t<::cuda::std::__num_bits_v<_Up>>;
    const ::cuda::std::uint32_t __word = ::cuda::std::bit_cast<__unsigned_t>(__data); // upcast to 32 bits
    const auto __shuffled              = ::__shfl_xor_sync(__lane_mask, __word, __xor_mask, _Width);
    const auto __narrowed              = static_cast<__unsigned_t>(__shuffled);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), true};
  }
  else if constexpr (__is_64bit_shuffle_path_v<_Up>)
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
    const auto __dword    = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__data);
    const auto __shuffled = ::__shfl_xor_sync(__lane_mask, __dword, __xor_mask, _Width);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__shuffled), true};
  }
  else if constexpr (__is_64bit_array_shuffle_path_v<_Up>)
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
    ::cuda::std::uint64_t __dword;
    ::cuda::std::memcpy(&__dword, &__data, sizeof(__dword));
    const auto __shuffled = ::__shfl_xor_sync(__lane_mask, __dword, __xor_mask, _Width);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__shuffled), true};
  }
  else
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "delta must be in the range [1, _Width)");
    constexpr int __ratio          = ::cuda::ceil_div(sizeof(_Up), sizeof(::cuda::std::uint32_t));
    constexpr auto __clamp_segmask = (_Width - 1u) | ((__warp_size - _Width) << 8);
    bool __pred;
    ::cuda::std::uint32_t __array[__ratio]{}; // zero-initialize -> avoid undetermined values progatation (reg pressure)
    ::cuda::std::memcpy(
      static_cast<void*>(__array), static_cast<const void*>(::cuda::std::addressof(__data)), sizeof(_Up));

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_bfly(__array[i], __pred, __xor_mask, __clamp_segmask, __lane_mask);
    }
    warp_shuffle_result<_Up> __result;
    __result.pred = true; // 0 < __xor_mask < _Width => lane ^ xor_mask is always in the segment range
    ::cuda::std::memcpy(
      static_cast<void*>(::cuda::std::addressof(__result.data)), static_cast<void*>(__array), sizeof(_Up));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_xor(const _Tp& __data, const int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_xor(__data, __src_lane, 0xFFFFFFFFu, __width);
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 600
#endif // _CCCL_CUDA_COMPILATION()
#endif // _CUDA___WARP_WARP_SHUFFLE_H
