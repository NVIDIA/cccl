//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_CONFIG_H
#define _CUDA___FLOATING_POINT_CONFIG_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__floating_point/type_traits.h>
#  include <cuda/__fwd/fp.h>
#  include <cuda/std/__bit/integral.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/cfloat>

// Silence the warning about the use of long double in device code
_CCCL_NV_DIAG_SUPPRESS(20208)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// _FpConfig is a type that provides the configuration for the __fp type
// - static constexpr variables:
//   - __exp_nbits: number of bits in the exponent
//   - __mant_nbits: number of bits in the mantissa
//   - __is_signed: whether the floating point type is signed
//   - __has_inf: whether the floating point type has infinity
//   - __has_nan: whether the floating point type has quiet NaN
//   - __has_nans: whether the floating point type has signaling NaN
//   - __has_denorm: whether the floating point type has denormalized values
//   - __native_type: the native type that the floating point type is based on
//   - __is_iec559: whether the floating point type is based on IEC 559
// - type aliases:
//   - __host native_type: the host native type that the floating point type is based on (if no native type, use
//   __fp_no_native_type_tag)
//   - __device native_type: the device native type that the floating point type is based on (if no native type, use
//   __fp_no_native_type_tag)
// - static member functions:
//   - __min(): returns the minimum value for the floating point type
//   - __max(): returns the maximum value for the floating point type
//   - __inf(): returns the infinity value for the floating point type (if __has_inf is true)
//   - __nan(): returns the quiet NaN value for the floating point type (if __has_quiet_nan is true)
//   - __nans(): returns the signaling NaN value for the floating point type (if __has_signaling_nan is true)
//   - __is_inf(): returns whether the given value is infinity (if __has_inf is true)
//   - __is_nan(): returns whether the given value is NaN (if __has_nan or __has_nans is true)

struct __fp_no_native_type_tag
{};

struct __fp_invalid_config
{};

struct __fp_from_native_t
{};

_CCCL_INLINE_VAR constexpr __fp_from_native_t __fp_from_native{};

struct __fp_from_storage_t
{};

_CCCL_INLINE_VAR constexpr __fp_from_storage_t __fp_from_storage{};

struct __fp_iec559_config_base
{
  static constexpr bool __is_signed  = true;
  static constexpr bool __has_inf    = true;
  static constexpr bool __has_nan    = true;
  static constexpr bool __has_nans   = true;
  static constexpr bool __has_denorm = true;
  static constexpr bool __is_iec559  = true;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    _Tp __ret{};
    __ret.__set_exp(typename _Tp::__storage_type{1});
    return __ret;
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    _Tp __ret{};
    __ret.__set_exp(static_cast<typename _Tp::__storage_type>(~typename _Tp::__storage_type{1}));
    return __ret;
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __inf()
  {
    _Tp __ret{};
    __ret.__set_exp(_Tp::__exp_val_mask());
    return __ret;
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __nan()
  {
    _Tp __ret = __inf<_Tp>();
    __ret.__set_mant(typename _Tp::__storage_type{1} << (_Tp::__mant_nbits - 1));
    return __ret;
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __nans()
  {
    _Tp __ret = __inf<_Tp>();
    __ret.__set_mant(typename _Tp::__storage_type{1} << (_Tp::__mant_nbits - 2));
    return __ret;
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_inf(const _Tp& __val)
  {
    return __val.__get_exp() == _Tp::__exp_val_mask() && __val.__get_mant() == typename _Tp::__storage_type{0};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_nan(const _Tp& __val)
  {
    return __val.__get_exp() == _Tp::__exp_val_mask() && __val.__get_mant() != typename _Tp::__storage_type{0};
  }
};

struct __fp4_e2m1_config
{
  static constexpr size_t __exp_nbits  = 2;
  static constexpr size_t __mant_nbits = 1;
  static constexpr bool __is_signed    = true;
  static constexpr bool __has_inf      = false;
  static constexpr bool __has_nan      = false;
  static constexpr bool __has_nans     = false;
  static constexpr bool __has_denorm   = true;
  static constexpr bool __is_iec559    = false;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x08}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x7}};
  }
};

struct __fp6_e2m3_config
{
  static constexpr size_t __exp_nbits  = 2;
  static constexpr size_t __mant_nbits = 3;
  static constexpr bool __is_signed    = true;
  static constexpr bool __has_inf      = false;
  static constexpr bool __has_nan      = false;
  static constexpr bool __has_nans     = false;
  static constexpr bool __has_denorm   = true;
  static constexpr bool __is_iec559    = false;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x08}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x1f}};
  }
};

struct __fp6_e3m2_config
{
  static constexpr size_t __exp_nbits  = 3;
  static constexpr size_t __mant_nbits = 2;
  static constexpr bool __is_signed    = true;
  static constexpr bool __has_inf      = false;
  static constexpr bool __has_nan      = false;
  static constexpr bool __has_nans     = false;
  static constexpr bool __has_denorm   = true;
  static constexpr bool __is_iec559    = false;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x08}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x1f}};
  }
};

struct __fp8_e4m3_config
{
  static constexpr size_t __exp_nbits  = 4;
  static constexpr size_t __mant_nbits = 3;
  static constexpr bool __is_signed    = true;
  static constexpr bool __has_inf      = false;
  static constexpr bool __has_nan      = true;
  static constexpr bool __has_nans     = false;
  static constexpr bool __has_denorm   = true;
  static constexpr bool __is_iec559    = false;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x08}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x7e}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __nan()
  {
    return _Tp{__fp_from_storage, static_cast<typename _Tp::__storage_type>(0x7f)};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_nan(const _Tp& __val)
  {
    return (__val.__storage & static_cast<typename _Tp::__storage_type>(0x7f))
        == static_cast<typename _Tp::__storage_type>(0x7f);
  }
};

struct __fp8_e5m2_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = 5;
  static constexpr size_t __mant_nbits = 2;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;
};

struct __fp16_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = 5;
  static constexpr size_t __mant_nbits = 10;
  static constexpr bool __is_iec559    = false;

#  if __STDCPP_FLOAT16_T__ == 1
  using __host_native_type = ::std::float16_t;
#  elif _CCCL_COMPILER(GCC, >=, 7) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC, >=, 24, 9)
  using __host_native_type = _Float16;
#  else
  using __host_native_type = __fp_no_native_type_tag;
#  endif
#  if _CCCL_CUDA_COMPILER(CLANG, >=, 19) || _CCCL_CUDA_COMPILER(NVHPC, >=, 24, 9)
  using __device_native_type = _Float16;
#  else
  using __device_native_type = __fp_no_native_type_tag;
#  endif
};

struct __fp32_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = 8;
  static constexpr size_t __mant_nbits = 23;

  using __host_native_type   = float;
  using __device_native_type = float;
};

struct __fp64_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = 11;
  static constexpr size_t __mant_nbits = 52;

  using __host_native_type   = double;
  using __device_native_type = double;
};

#  if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
struct __fp_long_double_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = _CUDA_VSTD::bit_width(static_cast<size_t>(LDBL_MAX_EXP));
  static constexpr size_t __mant_nbits = LDBL_MANT_DIG;

  using __host_native_type   = long double;
  using __device_native_type = __fp_no_native_type_tag;
};
#  endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE

struct __fp128_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = 15;
  static constexpr size_t __mant_nbits = 112;

#  if __STDCPP_FLOAT128_T__ == 1
  using __host_native_type = ::std::float128_t;
#  elif (defined(__SIZEOF_FLOAT128__) || defined(__FLOAT128__)) && _CCCL_OS(LINUX)                        \
    && (_CCCL_COMPILER(GCC) || _CCCL_COMPILER(CLANG) || _CCCL_COMPILER(NVHPC)) && !defined(__CUDA_ARCH__) \
    && !_CCCL_COMPILER(NVRTC)
  using __host_native_type = __float128;
#  else
  using __host_native_type = __fp_no_native_type_tag;
#  endif
  using __device_native_type = __fp_no_native_type_tag;
};

struct __bf16_config : __fp_iec559_config_base
{
  static constexpr size_t __exp_nbits  = 8;
  static constexpr size_t __mant_nbits = 7;

#  if __STDCPP_BFLOAT16_T__ == 1
  using __host_native_type = ::std::bfloat16_t;
#  elif _CCCL_COMPILER(GCC, >=, 13) || (_CCCL_COMPILER(CLANG, >=, 15) && _CCCL_ARCH(X86_64)) \
    || (_CCCL_COMPILER(CLANG, >=, 11) && _CCCL_ARCH(ARM64))
  using __host_native_type = __bf16;
#  else
  using __host_native_type = __fp_no_native_type_tag;
#  endif
#  if _CCCL_CUDA_COMPILER(CLANG, >=, 17)
  using __device_native_type = __bf16;
#  else
  using __device_native_type = __fp_no_native_type_tag;
#  endif
};

struct __fp8_ue4m3_config
{
  static constexpr size_t __exp_nbits  = 4;
  static constexpr size_t __mant_nbits = 3;
  static constexpr bool __is_signed    = false;
  static constexpr bool __has_inf      = false;
  static constexpr bool __has_nan      = true;
  static constexpr bool __has_nans     = false;
  static constexpr bool __has_denorm   = true;
  static constexpr bool __is_iec559    = false;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x08}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x7e}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __nan()
  {
    return _Tp{__fp_from_storage, static_cast<typename _Tp::__storage_type>(0x7f)};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_nan(const _Tp& __val)
  {
    return (__val.__storage & static_cast<typename _Tp::__storage_type>(0x7f))
        == static_cast<typename _Tp::__storage_type>(0x7f);
  }
};

struct __fp8_ue8m0_config
{
  static constexpr size_t __exp_nbits  = 8;
  static constexpr size_t __mant_nbits = 0;
  static constexpr bool __is_signed    = false;
  static constexpr bool __has_inf      = false;
  static constexpr bool __has_nan      = true;
  static constexpr bool __has_nans     = false;
  static constexpr bool __has_denorm   = false;
  static constexpr bool __is_iec559    = false;

  using __host_native_type   = __fp_no_native_type_tag;
  using __device_native_type = __fp_no_native_type_tag;

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __min()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0x00}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __max()
  {
    return _Tp{__fp_from_storage, typename _Tp::__storage_type{0xff}};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __nan()
  {
    return _Tp{__fp_from_storage, static_cast<typename _Tp::__storage_type>(0xff)};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_extended_floating_point_v<_Tp>)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __is_nan(const _Tp& __val)
  {
    return __val.__storage == static_cast<typename _Tp::__storage_type>(0xff);
  }
};

template <class _Tp>
_LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_make_config_from()
{
  using _Up = _CUDA_VSTD::remove_cv_t<_Tp>;

  if constexpr (__is_cuda_extended_floating_point_v<_Up>)
  {
    return typename _Up::__config_type{};
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, float>)
  {
    return __fp32_config{};
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, double>)
  {
    return __fp64_config{};
  }
#  if !defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, long double>)
  {
    return __fp_long_double_config{};
  }
#  endif // !_LIBCUDACXX_HAS_NO_LONG_DOUBLE
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, typename __fp16_config::__host_native_type>
                     || _CUDA_VSTD::is_same_v<_Up, typename __fp16_config::__device_native_type>)
  {
    return __fp16_config{};
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, typename __fp128_config::__host_native_type>
                     || _CUDA_VSTD::is_same_v<_Up, typename __fp128_config::__device_native_type>)
  {
    return __fp128_config{};
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, typename __bf16_config::__host_native_type>
                     || _CUDA_VSTD::is_same_v<_Up, typename __bf16_config::__device_native_type>)
  {
    return __bf16_config{};
  }
#  if __STDCPP_FLOAT16_T__ == 1
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, ::std::float16_t>)
  {
    return __fp16_config{};
  }
#  endif // __STDCPP_FLOAT16_T__
#  if __STDCPP_FLOAT32_T__ == 1
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, ::std::float32_t>)
  {
    return __fp32_config{};
  }
#  endif // __STDCPP_FLOAT32_T__
#  if __STDCPP_FLOAT64_T__ == 1
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, ::std::float64_t>)
  {
    return __fp64_config{};
  }
#  endif // __STDCPP_FLOAT64_T__
#  if __STDCPP_FLOAT128_T__ == 1
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, ::std::float128_t>)
  {
    return __fp128_config{};
  }
#  endif // __STDCPP_FLOAT128_T__
#  if __STDCPP_BFLOAT16_T__ == 1
  else if constexpr (_CUDA_VSTD::is_same_v<_Up, ::std::bfloat16_t>)
  {
    return __bf16_config{};
  }
#  endif // __STDCPP_BFLOAT16_T__
  else
  {
    return __fp_invalid_config{};
  }
}

template <class _Tp>
using __fp_make_config_from_t = decltype(::cuda::__fp_make_config_from<_Tp>());

_LIBCUDACXX_END_NAMESPACE_CUDA

_CCCL_NV_DIAG_DEFAULT(20208)

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_CONFIG_H
