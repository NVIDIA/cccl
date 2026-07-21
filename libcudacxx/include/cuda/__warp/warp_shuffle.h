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
#    include <cuda/__ptx/instructions/shfl_sync.h>
#    include <cuda/__type_traits/is_trivially_copyable.h>
#    include <cuda/std/__memory/addressof.h>
#    include <cuda/std/__type_traits/enable_if.h>
#    include <cuda/std/__type_traits/integral_constant.h>
#    include <cuda/std/__type_traits/is_array.h>
#    include <cuda/std/__type_traits/is_default_constructible.h>
#    include <cuda/std/__type_traits/make_nbit_int.h>
#    include <cuda/std/array>
#    include <cuda/std/climits>
#    include <cuda/std/cstdint>

#    include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DEVICE

inline constexpr auto __warp_threads = 32u;

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

//----------------------------------------------------------------------------------------------------------------------
// Internal helper

// PTX shuffles 32-bit words. These paths avoid generic array packing, which adds instructions and increases register
// pressure for 8-/16-bit values and fragments 64-bit values into independent 32-bit registers.

#    define _CCCL_WARP_SHUFFLE_INT128_OPTIMIZED() (_CCCL_HAS_INT128() && __cccl_ptx_isa >= 830)

// bit_cast does not support arrays. Larger types use their specialized or generic array paths.
// CUDA 12.0 cannot discard the generic return type when deducing __shuffle_cast's auto return type.
template <typename _Up>
inline constexpr bool __is_shuffle_bitcast_path_v =
  !_CCCL_CUDA_COMPILER(NVCC, ==, 12, 0) && !_CCCL_CUDA_COMPILER(NVRTC, ==, 12, 0)
  && !::cuda::std::is_array_v<_Up> && (sizeof(_Up) == 1 || sizeof(_Up) == 2 || sizeof(_Up) == 4);

// FP4 and FP6 have 8-bit storage, so use the storage width rather than the number of value bits.
template <typename _Tp>
using __shuffle_bitcast_storage_t = ::cuda::std::__make_nbit_uint_t<sizeof(_Tp) * CHAR_BIT>;

template <typename _Tp>
_CCCL_DEVICE_API auto __shuffle_cast(const _Tp& __data) noexcept
{
  if constexpr (__is_shuffle_bitcast_path_v<_Tp>)
  {
    using __unsigned_t = __shuffle_bitcast_storage_t<_Tp>;
    return static_cast<::cuda::std::uint32_t>(::cuda::std::bit_cast<__unsigned_t>(__data));
  }
  else if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint64_t) && !::cuda::std::is_array_v<_Tp>)
  {
    const auto __value = ::cuda::std::bit_cast<::cuda::std::uint64_t>(__data);
    ::cuda::std::array<::cuda::std::uint32_t, 2> __array;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(__array[0]), "=r"(__array[1]) : "l"(__value));
    return __array;
  }
#    if _CCCL_WARP_SHUFFLE_INT128_OPTIMIZED()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t) && !::cuda::std::is_array_v<_Tp>)
  {
    const auto __value = ::cuda::std::bit_cast<__uint128_t>(__data);
    ::cuda::std::array<::cuda::std::uint32_t, 4> __array;
    NV_IF_TARGET(
      NV_PROVIDES_SM_70,
      (asm("mov.b128 {%0, %1, %2, %3}, %4;" : "=r"(__array[0]),
           "=r"(__array[1]),
           "=r"(__array[2]),
           "=r"(__array[3]) : "q"(__value));),
      ({
        asm("mov.b64 {%0, %1}, %2;" : "=r"(__array[0]), "=r"(__array[1]) : "l"(__value));
        asm("mov.b64 {%0, %1}, %2;" : "=r"(__array[2]), "=r"(__array[3]) : "l"(__value >> 64));
      }))
    return __array;
  }
  else
#    endif // _CCCL_WARP_SHUFFLE_INT128_OPTIMIZED()
  {
    constexpr auto __ratio = ::cuda::ceil_div(sizeof(_Tp), sizeof(::cuda::std::uint32_t));
    using __array_t        = ::cuda::std::array<::cuda::std::uint32_t, __ratio>;
    __array_t __array{}; // zero-initialize -> avoid undetermined values progatation (reg pressure)
    ::cuda::std::memcpy(
      static_cast<void*>(__array.data()), static_cast<const void*>(::cuda::std::addressof(__data)), sizeof(_Tp));
    return __array;
  }
}

template <typename _Tp, ::cuda::std::size_t _Ratio>
_CCCL_DEVICE_API auto
__make_shuffle_result(const ::cuda::std::array<::cuda::std::uint32_t, _Ratio>& __array, const bool __pred) noexcept
{
  if constexpr (sizeof(_Tp) == sizeof(::cuda::std::uint64_t) && !::cuda::std::is_array_v<_Tp>)
  {
    ::cuda::std::uint64_t __shuffled;
    asm("mov.b64 %0, {%1, %2};" : "=l"(__shuffled) : "r"(__array[0]), "r"(__array[1]));
    return warp_shuffle_result<_Tp>{::cuda::std::bit_cast<_Tp>(__shuffled), __pred};
  }
#    if _CCCL_WARP_SHUFFLE_INT128_OPTIMIZED()
  else if constexpr (sizeof(_Tp) == sizeof(__uint128_t) && !::cuda::std::is_array_v<_Tp>)
  {
    __uint128_t __shuffled;
    NV_IF_TARGET(
      NV_PROVIDES_SM_70,
      (asm("mov.b128 %0, {%1, %2, %3, %4};" : "=q"(__shuffled) : "r"(__array[0]),
           "r"(__array[1]),
           "r"(__array[2]),
           "r"(__array[3]));),
      ({
        ::cuda::std::uint64_t __lo, __hi;
        asm("mov.b64 %0, {%1, %2};" : "=l"(__lo) : "r"(__array[0]), "r"(__array[1]));
        asm("mov.b64 %0, {%1, %2};" : "=l"(__hi) : "r"(__array[2]), "r"(__array[3]));
        __shuffled = (static_cast<__uint128_t>(__hi) << 64) | static_cast<__uint128_t>(__lo);
      }))
    return warp_shuffle_result<_Tp>{::cuda::std::bit_cast<_Tp>(__shuffled), __pred};
  }
  else
#    endif // _CCCL_WARP_SHUFFLE_INT128_OPTIMIZED()
  {
    warp_shuffle_result<_Tp> __result;
    __result.pred = __pred; // __src_lane is always in range [minLane, maxLane]
    ::cuda::std::memcpy(
      static_cast<void*>(::cuda::std::addressof(__result.data)), static_cast<const void*>(__array.data()), sizeof(_Tp));
    return __result;
  }
}

template <int _Width, typename _Tp, typename _Up>
_CCCL_DEVICE_API constexpr void __warp_shuffle_preconditions()
{
  static_assert(::cuda::std::is_default_constructible_v<_Tp>,
                "cuda::device::warp_shuffle: _Tp must be default constructible");
  static_assert(::cuda::is_power_of_two(_Width) && _Width >= 1 && _Width <= __warp_threads,
                "cuda::device::warp_shuffle: _Width must be a power of 2 and less or equal to the warp size");
  static_assert(::cuda::is_trivially_copyable_v<_Up>, "cuda::device::warp_shuffle: _Up must be trivially copyable");
}

// ---------------------------------------------------------------------------------------------------------------------
// PUBLIC API

// warp_shuffle_idx

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_idx(
  const _Tp& __data,
  const int __src_lane,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  ::cuda::device::__warp_shuffle_preconditions<_Width, _Tp, _Up>();
  // __src_lane is unrestricted because th final shuffle index is __src_lane % _Width
  // __src_lane is always in range [minLane, maxLane]

  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else if constexpr (__is_shuffle_bitcast_path_v<_Up>)
  {
    using __unsigned_t    = __shuffle_bitcast_storage_t<_Up>;
    const auto __value    = ::cuda::device::__shuffle_cast(__data);
    const auto __shuffled = ::__shfl_sync(__lane_mask, __value, __src_lane, _Width);
    const auto __narrowed = static_cast<__unsigned_t>(__shuffled);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), true};
  }
  else
  {
    auto __array          = ::cuda::device::__shuffle_cast(__data);
    constexpr int __ratio = __array.size();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::__shfl_sync(__lane_mask, __array[i], __src_lane, _Width);
    }
    return ::cuda::device::__make_shuffle_result<_Up>(__array, true);
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_idx(const _Tp& __data, const int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_idx(__data, __src_lane, 0xFFFFFFFFu, __width);
}

// warp_shuffle_xor

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_xor(
  const _Tp& __data,
  const int __xor_mask,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  ::cuda::device::__warp_shuffle_preconditions<_Width, _Tp, _Up>();
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __xor_mask, &__pred1),
                                                           "all active lanes must have the same xor_mask");))
  // 0 < __xor_mask < _Width => lane ^ xor_mask is always in the segment range

  if constexpr (_Width == 1)
  {
    _CCCL_ASSERT(__xor_mask == 0, "xor_mask must be 0 when Width == 1");
    return warp_shuffle_result<_Up>{__data, true};
  }
  else if constexpr (__is_shuffle_bitcast_path_v<_Up>)
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "xor_mask must be in the range [1, _Width)");
    using __unsigned_t    = __shuffle_bitcast_storage_t<_Up>;
    const auto __value    = ::cuda::device::__shuffle_cast(__data);
    const auto __shuffled = ::__shfl_xor_sync(__lane_mask, __value, __xor_mask, _Width);
    const auto __narrowed = static_cast<__unsigned_t>(__shuffled);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), true};
  }
  else
  {
    _CCCL_ASSERT(__xor_mask >= 1 && __xor_mask < _Width, "xor_mask must be in the range [1, _Width)");
    auto __array          = ::cuda::device::__shuffle_cast(__data);
    constexpr int __ratio = __array.size();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::__shfl_xor_sync(__lane_mask, __array[i], __xor_mask, _Width);
    }
    return ::cuda::device::__make_shuffle_result<_Up>(__array, true);
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_xor(const _Tp& __data, const int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_xor(__data, __src_lane, 0xFFFFFFFFu, __width);
}

// warp_shuffle_up

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_up(
  const _Tp& __data,
  const int __delta,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  ::cuda::device::__warp_shuffle_preconditions<_Width, _Tp, _Up>();
  _CCCL_ASSERT(__delta >= 0 && __delta < _Width, "delta must be in the range [0, _Width)");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __delta, &__pred1),
                                                           "all active lanes must have the same delta");))
  [[maybe_unused]] constexpr auto __clamp_segmask = (__warp_threads - _Width) << 8;
  [[maybe_unused]] bool __pred;

  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else if constexpr (__is_shuffle_bitcast_path_v<_Up>)
  {
    using __unsigned_t    = __shuffle_bitcast_storage_t<_Up>;
    const auto __value    = ::cuda::device::__shuffle_cast(__data);
    const auto __shuffled = ::cuda::ptx::shfl_sync_up(__value, __pred, __delta, __clamp_segmask, __lane_mask);
    const auto __narrowed = static_cast<__unsigned_t>(__shuffled);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), __pred};
  }
  else
  {
    auto __array          = ::cuda::device::__shuffle_cast(__data);
    constexpr int __ratio = __array.size();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_up(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    return ::cuda::device::__make_shuffle_result<_Up>(__array, __pred);
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_up(const _Tp& __data, int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_up(__data, __src_lane, 0xFFFFFFFFu, __width);
}

// warp_shuffle_down

template <int _Width = 32, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up> warp_shuffle_down(
  const _Tp& __data,
  const int __delta,
  const ::cuda::std::uint32_t __lane_mask     = 0xFFFFFFFFu,
  ::cuda::std::integral_constant<int, _Width> = {})
{
  ::cuda::device::__warp_shuffle_preconditions<_Width, _Tp, _Up>();
  _CCCL_ASSERT(__delta >= 0 && __delta < _Width, "__delta must be in the range [0, _Width)");
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               ([[maybe_unused]] int __pred1; _CCCL_ASSERT(::__match_all_sync(::__activemask(), __delta, &__pred1),
                                                           "all active lanes must have the same delta");))
  [[maybe_unused]] constexpr auto __clamp_segmask = (_Width - 1u) | ((__warp_threads - _Width) << 8);
  [[maybe_unused]] bool __pred;

  if constexpr (_Width == 1)
  {
    return warp_shuffle_result<_Up>{__data, true};
  }
  else if constexpr (__is_shuffle_bitcast_path_v<_Up>)
  {
    using __unsigned_t    = __shuffle_bitcast_storage_t<_Up>;
    const auto __value    = ::cuda::device::__shuffle_cast(__data);
    const auto __shuffled = ::cuda::ptx::shfl_sync_down(__value, __pred, __delta, __clamp_segmask, __lane_mask);
    const auto __narrowed = static_cast<__unsigned_t>(__shuffled);
    return warp_shuffle_result<_Up>{::cuda::std::bit_cast<_Up>(__narrowed), __pred};
  }
  else
  {
    auto __array          = ::cuda::device::__shuffle_cast(__data);
    constexpr int __ratio = __array.size();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < __ratio; ++i)
    {
      __array[i] = ::cuda::ptx::shfl_sync_down(__array[i], __pred, __delta, __clamp_segmask, __lane_mask);
    }
    return ::cuda::device::__make_shuffle_result<_Up>(__array, __pred);
  }
}

template <int _Width, typename _Tp, typename _Up = ::cuda::std::remove_cv_t<_Tp>>
[[nodiscard]] _CCCL_DEVICE_API warp_shuffle_result<_Up>
warp_shuffle_down(const _Tp& __data, const int __src_lane, ::cuda::std::integral_constant<int, _Width> __width)
{
  return ::cuda::device::warp_shuffle_down(__data, __src_lane, 0xFFFFFFFFu, __width);
}

_CCCL_END_NAMESPACE_CUDA_DEVICE

#    include <cuda/std/__cccl/epilogue.h>

#  endif // __cccl_ptx_isa >= 600
#endif // _CCCL_CUDA_COMPILATION()
#endif // _CUDA___WARP_WARP_SHUFFLE_H
