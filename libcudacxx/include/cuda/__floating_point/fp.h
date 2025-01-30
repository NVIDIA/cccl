//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_FP_H
#define _CUDA___FLOATING_POINT_FP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__floating_point/config.h>
#  include <cuda/__floating_point/conv_rank_order.h>
#  include <cuda/__floating_point/storage.h>
#  include <cuda/__floating_point/type_traits.h>
#  include <cuda/__fwd/fp.h>
#  include <cuda/std/__bit/bit_cast.h>
#  include <cuda/std/__bit/integral.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_constant_evaluated.h>
#  include <cuda/std/__type_traits/is_integral.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/make_signed.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/climits>

#  include <nv/target>

// Silence the warning about the use of long double in device code
_CCCL_NV_DIAG_SUPPRESS(20208)

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// Signed FP types
using fp4_e2m1 = __fp<__fp4_e2m1_config>;
using fp6_e2m3 = __fp<__fp6_e2m3_config>;
using fp6_e3m2 = __fp<__fp6_e3m2_config>;
using fp8_e4m3 = __fp<__fp8_e4m3_config>;
using fp8_e5m2 = __fp<__fp8_e5m2_config>;
using fp16     = __fp<__fp16_config>;
using bf16     = __fp<__bf16_config>;
using fp32     = __fp<__fp32_config>;
using fp64     = __fp<__fp64_config>;
// #  if !defined(_LIBCUDACXX_HAS_NO_INT128)
// using fp128 = __fp<__fp128_config>;
// #  endif // !_LIBCUDACXX_HAS_NO_INT128

// Unsigned FP types
using fp8_ue4m3 = __fp<__fp8_ue4m3_config>;
using fp8_ue8m0 = __fp<__fp8_ue8m0_config>;

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__fp_is_floating_point_v<_Tp>)
__fp(_Tp) -> __fp<__fp_make_config_from_t<_Tp>>;

template <class _Tp>
__fp(__fp_from_native_t, _Tp) -> __fp<__fp_make_config_from_t<_Tp>>;

template <class _To, class _From>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __fp_cast_is_implicit()
{
  constexpr auto __rank_order = ::cuda::__fp_make_conv_rank_order<_To, _From>();

  return __rank_order == __fp_conv_rank_order::__equal || __rank_order == __fp_conv_rank_order::__greater;
}

template <class _Config>
class __fp
{
public:
  template <class>
  friend class __fp;
  friend _Config;
  friend struct __fp_cast;
  friend struct __fp_ops;

  using __config_type = _Config;

  static constexpr size_t __exp_nbits  = __config_type::__exp_nbits;
  static constexpr size_t __mant_nbits = __config_type::__mant_nbits;
  static constexpr bool __is_signed    = __config_type::__is_signed;
  static constexpr size_t __nbits      = __exp_nbits + __mant_nbits + __is_signed;
  static constexpr bool __has_inf      = __config_type::__has_inf;
  static constexpr bool __has_nan      = __config_type::__has_nan;
  static constexpr bool __has_nans     = __config_type::__has_nans;
  static constexpr bool __has_denorm   = __config_type::__has_denorm;

  using __storage_type       = __fp_storage_t<__nbits>;
  using __host_native_type   = typename __config_type::__host_native_type;
  using __device_native_type = typename __config_type::__device_native_type;

  static constexpr bool __has_host_native_type = !_CUDA_VSTD::is_same_v<__host_native_type, __fp_no_native_type_tag>;
  static constexpr bool __has_device_native_type =
    !_CUDA_VSTD::is_same_v<__device_native_type, __fp_no_native_type_tag>;

  static_assert(!__has_host_native_type || sizeof(__storage_type) == sizeof(__host_native_type));
  static_assert(!__has_device_native_type || sizeof(__storage_type) == sizeof(__device_native_type));

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __min()
  {
    return __config_type::template __min<__fp>();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __max()
  {
    return __config_type::template __max<__fp>();
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __lowest()
  {
    __fp __ret{};
    if constexpr (__is_signed)
    {
      __ret = __max();
      __ret.__set_sign(true);
    }
    return __ret;
  }
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __epsilon()
  {
    // TODO: implement epsilon
    return __fp{};
  }
  _CCCL_TEMPLATE(bool _HasInf = __has_inf)
  _CCCL_REQUIRES(_HasInf)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __inf()
  {
    return __config_type::template __inf<__fp>();
  }
  _CCCL_TEMPLATE(bool _HasNan = __has_nan)
  _CCCL_REQUIRES(_HasNan)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __nan()
  {
    return __config_type::template __nan<__fp>();
  }
  _CCCL_TEMPLATE(bool _HasNans = __has_nans)
  _CCCL_REQUIRES(_HasNans)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __nans()
  {
    return __config_type::template __nans<__fp>();
  }
  _CCCL_TEMPLATE(bool _HasDenorm = __has_denorm)
  _CCCL_REQUIRES(_HasDenorm)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __denorm_min()
  {
    return __fp{__fp_from_storage, __storage_type{1}};
  }

  _CCCL_HIDE_FROM_ABI constexpr __fp() = default;

  _CCCL_HIDE_FROM_ABI constexpr __fp(const __fp&) = default;

#  if defined(_CCCL_NO_CONDITIONAL_EXPLICIT)
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__fp_is_floating_point_v<_Tp> _CCCL_AND __fp_cast_is_implicit<__fp, _Tp>())
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp(const _Tp& __v) noexcept
      : __fp{__construct_from(__v)}
  {}

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__fp_is_floating_point_v<_Tp> _CCCL_AND(!__fp_cast_is_implicit<__fp, _Tp>()))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __fp(const _Tp& __v) noexcept
      : __fp{__construct_from(__v)}
  {}
#  else // ^^^ _CCCL_NO_CONDITIONAL_EXPLICIT ^^^ / vvv !_CCCL_NO_CONDITIONAL_EXPLICIT vvv
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__fp_is_floating_point_v<_Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI explicit(!__fp_cast_is_implicit<__fp, _Tp>()) constexpr __fp(const _Tp& __v) noexcept
      : __fp{__construct_from(__v)}
  {}
#  endif // ^^^ !_CCCL_NO_CONDITIONAL_EXPLICIT ^^^

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp(const _Tp& __v) noexcept
      : __fp{__construct_from(__v)}
  {}

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::is_same_v<__fp_make_config_from_t<_Tp>, __config_type>)
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __fp(__fp_from_native_t, const _Tp& __v) noexcept
      : __fp{__fp_from_storage, _CUDA_VSTD::bit_cast<__storage_type>(__v)}
  {}

  template <class _Tp>
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr __fp(__fp_from_storage_t, const _Tp& __v) noexcept
      : __storage_{__v}
  {
    static_assert(_CUDA_VSTD::is_same_v<_Tp, __storage_type>);
  }

#  if defined(_CCCL_NO_CONDITIONAL_EXPLICIT)
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__fp_is_floating_point_v<_Tp> _CCCL_AND __fp_cast_is_implicit<_Tp, __fp>())
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr operator _Tp() const noexcept
  {
    return __cast_to<_Tp>();
  }

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__fp_is_floating_point_v<_Tp> _CCCL_AND(!__fp_cast_is_implicit<_Tp, __fp>()))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr operator _Tp() const noexcept
  {
    return __cast_to<_Tp>();
  }
#  else // ^^^ _CCCL_NO_CONDITIONAL_EXPLICIT ^^^ / vvv !_CCCL_NO_CONDITIONAL_EXPLICIT vvv
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__fp_is_floating_point_v<_Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI explicit(!__fp_cast_is_implicit<_Tp, __fp>()) constexpr operator _Tp() const noexcept
  {
    return __cast_to<_Tp>();
  }
#  endif // ^^^ !_CCCL_NO_CONDITIONAL_EXPLICIT ^^^

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Tp>)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr operator _Tp() const noexcept
  {
    return __cast_to<_Tp>();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __has_native_type() noexcept
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __has_host_native_type;), (return __has_device_native_type;))
  }

  // private:
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp __construct_from(const _Tp& __v) noexcept
  {
#  if defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
    static_assert(!_CUDA_VSTD::is_same_v<_Tp, long double>, "long double is not supported");
#  endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE

    // todo: improve the implementation
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __construct_from_host<_Tp>(__v);), (return __construct_from_device<_Tp>(__v);))
  }
#  if !_CCCL_COMPILER(NVRTC)
  template <class _Tp, class _FpCast = class __fp_cast>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST static constexpr __fp __construct_from_host(const _Tp& __v) noexcept
  {
    if constexpr (__has_host_native_type)
    {
      using _TpConfig = __fp_make_config_from_t<_Tp>;

      if constexpr (_CUDA_VSTD::is_same_v<_TpConfig, __config_type>)
      {
        return __fp{__fp_from_native, __v};
      }
      else if constexpr (!_CUDA_VSTD::is_same_v<_TpConfig, __fp_invalid_config>)
      {
        using _TpFp = __fp<_TpConfig>;
        if constexpr (_TpFp::__has_host_native_type)
        {
          return __fp{__fp_from_native,
                      static_cast<__host_native_type>(_CUDA_VSTD::bit_cast<typename _TpFp::__host_native_type>(__v))};
        }
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<_Tp>)
      {
        return __fp{__fp_from_native, static_cast<__host_native_type>(__v)};
      }
    }
    return _FpCast::template __cast<__fp>(__v);
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  template <class _Tp, class _FpCast = class __fp_cast>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static constexpr __fp __construct_from_device(const _Tp& __v) noexcept
  {
    if constexpr (__has_device_native_type)
    {
      using _TpConfig = __fp_make_config_from_t<_Tp>;

      if constexpr (_CUDA_VSTD::is_same_v<_TpConfig, __config_type>)
      {
        return __fp{__fp_from_native, __v};
      }
      else if constexpr (!_CUDA_VSTD::is_same_v<_TpConfig, __fp_invalid_config>)
      {
        using _TpFp = __fp<_TpConfig>;
        if constexpr (_TpFp::__has_device_native_type)
        {
          return __fp{
            __fp_from_native,
            static_cast<__device_native_type>(_CUDA_VSTD::bit_cast<typename _TpFp::__device_native_type>(__v))};
        }
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<_Tp>)
      {
        return __fp{__fp_from_native, static_cast<__device_native_type>(__v)};
      }
    }
    return _FpCast::template __cast<__fp>(__v);
  }

  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp __cast_to() const noexcept
  {
    using _Up = _CUDA_VSTD::remove_cv_t<_Tp>;

#  if defined(_LIBCUDACXX_HAS_NO_LONG_DOUBLE)
    static_assert(!_CUDA_VSTD::is_same_v<_Up, long double>, "long double is not supported");
#  endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE

    // todo: improve the implementation
    NV_IF_ELSE_TARGET(NV_IS_HOST, (return __cast_to_host<_Up>();), (return __cast_to_device<_Up>();))
  }

#  if !_CCCL_COMPILER(NVRTC)
  template <class _Tp, class _FpCast = class __fp_cast>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST constexpr _Tp __cast_to_host() const noexcept
  {
    if constexpr (__has_host_native_type)
    {
      using _TpConfig = __fp_make_config_from_t<_Tp>;

      if constexpr (_CUDA_VSTD::is_same_v<_TpConfig, __config_type>)
      {
        return _CUDA_VSTD::bit_cast<_Tp>(__storage_);
      }
      else if constexpr (!_CUDA_VSTD::is_same_v<_TpConfig, __fp_invalid_config>)
      {
        using _TpFp = __fp<_TpConfig>;
        if constexpr (_TpFp::__has_host_native_type)
        {
          return _CUDA_VSTD::bit_cast<_Tp>(static_cast<typename _TpFp::__host_native_type>(__host_native()));
        }
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<_Tp>)
      {
        return static_cast<_Tp>(__host_native());
      }
    }
    return _FpCast::template __cast<_Tp>(*this);
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  template <class _Tp, class _FpCast = class __fp_cast>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr _Tp __cast_to_device() const noexcept
  {
    if constexpr (__has_device_native_type)
    {
      using _TpConfig = __fp_make_config_from_t<_Tp>;

      if constexpr (_CUDA_VSTD::is_same_v<_TpConfig, __config_type>)
      {
        return _CUDA_VSTD::bit_cast<_Tp>(__storage_);
      }
      else if constexpr (!_CUDA_VSTD::is_same_v<_TpConfig, __fp_invalid_config>)
      {
        using _TpFp = __fp<_TpConfig>;
        if constexpr (_TpFp::__has_device_native_type)
        {
          return _CUDA_VSTD::bit_cast<_Tp>(static_cast<typename _TpFp::__device_native_type>(__device_native()));
        }
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<_Tp>)
      {
        return static_cast<_Tp>(__device_native());
      }
    }
    return _FpCast::template __cast<_Tp>(*this);
  }

#  if !_CCCL_COMPILER(NVRTC)
  _CCCL_TEMPLATE(bool _HasNativeType = __has_host_native_type)
  _CCCL_REQUIRES(_HasNativeType)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_HOST constexpr __host_native_type __host_native() const noexcept
  {
    return _CUDA_VSTD::bit_cast<__host_native_type>(__storage_);
  }
#  endif // !_CCCL_COMPILER(NVRTC)

  _CCCL_TEMPLATE(bool _HasNativeType = __has_device_native_type)
  _CCCL_REQUIRES(_HasNativeType)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE constexpr __device_native_type __device_native() const noexcept
  {
    return _CUDA_VSTD::bit_cast<__device_native_type>(__storage_);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __storage_type __mask() noexcept
  {
    return __sign_mask() | __exp_mask() | __mant_mask();
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __sign_shift() noexcept
  {
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    return (sizeof(__storage_type) * CHAR_BIT) - static_cast<size_t>(__is_signed);
#  else
    // return CHAR_BIT - static_cast<size_t>(__is_signed);
#  endif
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __storage_type __sign_mask() noexcept
  {
    return (__is_signed) ? __storage_type(1) << __sign_shift() : 0;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __storage_type __exp_val_mask() noexcept
  {
    return ((__storage_type(1) << __exp_nbits) - 1);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __exp_shift() noexcept
  {
    return __sign_shift() - __exp_nbits;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __storage_type __exp_mask() noexcept
  {
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    return static_cast<__storage_type>(__exp_val_mask() << __exp_shift());
#  else
    // return __exp_val_mask() << (sizeof(__storage_type) * CHAR_BIT - (__exp_nbits + __is_signed));
#  endif
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __storage_type __mant_val_mask() noexcept
  {
    return static_cast<__storage_type>((__storage_type(1) << __mant_nbits) - 1);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr size_t __mant_shift() noexcept
  {
    return __exp_shift() - __mant_nbits;
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __storage_type __mant_mask() noexcept
  {
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    return static_cast<__storage_type>(__mant_val_mask() << __mant_shift());
#  else
    // return __mant_val_mask() << (sizeof(__storage_type) * CHAR_BIT - (__mant_nbits + __exp_nbits + __is_signed));
#  endif
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __get_sign() const noexcept
  {
    return static_cast<bool>(__storage_ & __sign_mask());
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __storage_type __get_exp() const noexcept
  {
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    return static_cast<__storage_type>((__storage_ & __exp_mask()) >> __exp_shift());
#  else
    // return (__storage_ & __exp_mask()) >> (sizeof(__storage_type) * CHAR_BIT - (__exp_nbits + __is_signed));
#  endif
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __storage_type __get_mant() const noexcept
  {
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    return static_cast<__storage_type>(static_cast<__storage_type>(__storage_ & __mant_mask()) >> __mant_shift());
#  else
    // return (__storage_ & __mant_mask()) >> (sizeof(__storage_type) * CHAR_BIT - (__mant_nbits + __exp_nbits +
    // __is_signed));
#  endif
  }

  _CCCL_TEMPLATE(bool _IsSigned = __is_signed)
  _CCCL_REQUIRES(_IsSigned)
  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __set_sign(bool __sign) noexcept
  {
    __storage_ &= ~__sign_mask();
    __storage_ |= static_cast<__storage_type>(__sign) << __sign_shift();
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __set_exp(__storage_type __exp) noexcept
  {
    __storage_ &= ~__exp_mask();
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    __storage_ |= (__exp & __exp_val_mask()) << __exp_shift();
#  else
    // __storage_ |= (__exp & ((__storage_type(1) << __exp_nbits) - 1)) << (sizeof(__storage_type) * CHAR_BIT -
    // (__exp_nbits + __is_signed));
#  endif
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr void __set_mant(__storage_type __mant) noexcept
  {
    __storage_ &= ~__mant_mask();
#  if defined(_LIBCUDACXX_LITTLE_ENDIAN)
    __storage_ |= (__mant & __mant_val_mask()) << __mant_shift();
#  else
    // __storage_ |= (__mant & ((__storage_type(1) << __mant_nbits) - 1)) << (sizeof(__storage_type) * CHAR_BIT -
    // (__mant_nbits + __exp_nbits + __is_signed));
#  endif
  }

  _CCCL_TEMPLATE(bool _HasInf = __has_inf)
  _CCCL_REQUIRES(_HasInf)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __is_inf() const noexcept
  {
    return __config_type::template __is_inf<__fp>(*this);
  }

  _CCCL_TEMPLATE(bool _HasNan = __has_nan)
  _CCCL_REQUIRES(_HasNan)
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __is_nan() const noexcept
  {
    return __config_type::template __is_nan<__fp>(*this);
  }

  __storage_type __storage_;
};

_LIBCUDACXX_END_NAMESPACE_CUDA

_CCCL_NV_DIAG_DEFAULT(20208)

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_FP_H
