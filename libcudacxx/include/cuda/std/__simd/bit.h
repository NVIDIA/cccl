//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_BIT_H
#define _CUDA_STD___SIMD_BIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/byteswap.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__bit/integral.h>
#include <cuda/std/__bit/popcount.h>
#include <cuda/std/__bit/rotate.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__simd/basic_vec.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_unsigned_integer.h>
#include <cuda/std/__type_traits/make_signed.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

//----------------------------------------------------------------------------------------------------------------------
// [simd.bit] element-wise helpers

template <typename _Vp>
struct __simd_byteswap_generator
{
  using __result_t = typename _Vp::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return ::cuda::std::byteswap(__v_[_Idx::value]);
  }
};

template <typename _Vp>
struct __simd_bit_ceil_generator
{
  using __result_t = typename _Vp::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return ::cuda::std::bit_ceil(__v_[_Idx::value]);
  }
};

template <typename _Vp>
struct __simd_bit_floor_generator
{
  using __result_t = typename _Vp::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return ::cuda::std::bit_floor(__v_[_Idx::value]);
  }
};

template <typename _Vp>
struct __simd_has_single_bit_generator
{
  using __result_t = bool;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return ::cuda::std::has_single_bit(__v_[_Idx::value]);
  }
};

template <typename _Vp0, typename _Vp1>
struct __simd_rotl_generator
{
  using __result_t = typename _Vp0::value_type;
  const _Vp0& __v0_;
  const _Vp1& __v1_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::std::in_range<int>(__v1_[_Idx::value]), "rotl: count is out of range");
    return ::cuda::std::rotl(__v0_[_Idx::value], static_cast<int>(__v1_[_Idx::value]));
  }
};

template <typename _Vp0, typename _Vp1>
struct __simd_rotr_generator
{
  using __result_t = typename _Vp0::value_type;
  const _Vp0& __v0_;
  const _Vp1& __v1_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    _CCCL_ASSERT(::cuda::std::in_range<int>(__v1_[_Idx::value]), "rotr: count is out of range");
    return ::cuda::std::rotr(__v0_[_Idx::value], static_cast<int>(__v1_[_Idx::value]));
  }
};

template <typename _Vp>
struct __simd_rotl_scalar_generator
{
  using __result_t = typename _Vp::value_type;
  const _Vp& __v_;
  const int __s_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return ::cuda::std::rotl(__v_[_Idx::value], __s_);
  }
};

template <typename _Vp>
struct __simd_rotr_scalar_generator
{
  using __result_t = typename _Vp::value_type;
  const _Vp& __v_;
  const int __s_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return ::cuda::std::rotr(__v_[_Idx::value], __s_);
  }
};

template <typename _Vp, typename _Result>
struct __simd_bit_width_generator
{
  using __result_t = typename _Result::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return static_cast<__result_t>(::cuda::std::bit_width(__v_[_Idx::value]));
  }
};

template <typename _Vp, typename _Result>
struct __simd_countl_zero_generator
{
  using __result_t = typename _Result::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return static_cast<__result_t>(::cuda::std::countl_zero(__v_[_Idx::value]));
  }
};

template <typename _Vp, typename _Result>
struct __simd_countl_one_generator
{
  using __result_t = typename _Result::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return static_cast<__result_t>(::cuda::std::countl_one(__v_[_Idx::value]));
  }
};

template <typename _Vp, typename _Result>
struct __simd_countr_zero_generator
{
  using __result_t = typename _Result::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return static_cast<__result_t>(::cuda::std::countr_zero(__v_[_Idx::value]));
  }
};

template <typename _Vp, typename _Result>
struct __simd_countr_one_generator
{
  using __result_t = typename _Result::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return static_cast<__result_t>(::cuda::std::countr_one(__v_[_Idx::value]));
  }
};

template <typename _Vp, typename _Result>
struct __simd_popcount_generator
{
  using __result_t = typename _Result::value_type;
  const _Vp& __v_;

  template <typename _Idx>
  [[nodiscard]] _CCCL_API constexpr __result_t operator()(_Idx) const noexcept
  {
    return static_cast<__result_t>(::cuda::std::popcount(__v_[_Idx::value]));
  }
};

//----------------------------------------------------------------------------------------------------------------------
// [simd.bit], bit manipulation

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vp = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(is_integral_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vp byteswap(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Vp{__simd_byteswap_generator<_Vp>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vp = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vp bit_ceil(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Vp{__simd_bit_ceil_generator<_Vp>{__v}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vp = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vp bit_floor(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Vp{__simd_bit_floor_generator<_Vp>{__v}};
}

_CCCL_TEMPLATE(
  typename _Tp, typename _Abi, typename _Vp = basic_vec<_Tp, _Abi>, typename _Result = typename _Vp::mask_type)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result has_single_bit(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_has_single_bit_generator<_Vp>{__v}};
}

template <typename _Tp0, typename _Abi0, typename _Tp1, typename _Abi1>
inline constexpr bool __simd_is_valid_rotate_v =
  __cccl_is_unsigned_integer_v<_Tp0> //
  && is_integral_v<_Tp1> //
  && (__simd_size_v<_Tp0, _Abi0> == __simd_size_v<_Tp1, _Abi1>) //
  &&(sizeof(_Tp0) == sizeof(_Tp1));

_CCCL_TEMPLATE(typename _Tp0,
               typename _Abi0,
               typename _Tp1,
               typename _Abi1,
               typename _Vp0 = basic_vec<_Tp0, _Abi0>,
               typename _Vp1 = basic_vec<_Tp1, _Abi1>)
_CCCL_REQUIRES(__simd_is_valid_rotate_v<_Tp0, _Abi0, _Tp1, _Abi1>)
[[nodiscard]] _CCCL_API constexpr _Vp0
rotl(const basic_vec<_Tp0, _Abi0>& __v0, const basic_vec<_Tp1, _Abi1>& __v1) noexcept
{
  return _Vp0{__simd_rotl_generator<_Vp0, _Vp1>{__v0, __v1}};
}

_CCCL_TEMPLATE(typename _Tp0,
               typename _Abi0,
               typename _Tp1,
               typename _Abi1,
               typename _Vp0 = basic_vec<_Tp0, _Abi0>,
               typename _Vp1 = basic_vec<_Tp1, _Abi1>)
_CCCL_REQUIRES(__simd_is_valid_rotate_v<_Tp0, _Abi0, _Tp1, _Abi1>)
[[nodiscard]] _CCCL_API constexpr _Vp0
rotr(const basic_vec<_Tp0, _Abi0>& __v0, const basic_vec<_Tp1, _Abi1>& __v1) noexcept
{
  return _Vp0{__simd_rotr_generator<_Vp0, _Vp1>{__v0, __v1}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vp = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vp rotl(const basic_vec<_Tp, _Abi>& __v, const int __s) noexcept
{
  return _Vp{__simd_rotl_scalar_generator<_Vp>{__v, __s}};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Vp = basic_vec<_Tp, _Abi>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Vp rotr(const basic_vec<_Tp, _Abi>& __v, const int __s) noexcept
{
  return _Vp{__simd_rotr_scalar_generator<_Vp>{__v, __s}};
}

template <typename _Tp, typename _Vp>
using __simd_bit_count_result_t = rebind_t<make_signed_t<_Tp>, _Vp>;

_CCCL_TEMPLATE(typename _Tp,
               typename _Abi,
               typename _Vp     = basic_vec<_Tp, _Abi>,
               typename _Result = __simd_bit_count_result_t<_Tp, _Vp>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result bit_width(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_bit_width_generator<_Vp, _Result>{__v}};
}

_CCCL_TEMPLATE(typename _Tp,
               typename _Abi,
               typename _Vp     = basic_vec<_Tp, _Abi>,
               typename _Result = __simd_bit_count_result_t<_Tp, _Vp>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result countl_zero(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_countl_zero_generator<_Vp, _Result>{__v}};
}

_CCCL_TEMPLATE(typename _Tp,
               typename _Abi,
               typename _Vp     = basic_vec<_Tp, _Abi>,
               typename _Result = __simd_bit_count_result_t<_Tp, _Vp>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result countl_one(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_countl_one_generator<_Vp, _Result>{__v}};
}

_CCCL_TEMPLATE(typename _Tp,
               typename _Abi,
               typename _Vp     = basic_vec<_Tp, _Abi>,
               typename _Result = __simd_bit_count_result_t<_Tp, _Vp>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result countr_zero(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_countr_zero_generator<_Vp, _Result>{__v}};
}

_CCCL_TEMPLATE(typename _Tp,
               typename _Abi,
               typename _Vp     = basic_vec<_Tp, _Abi>,
               typename _Result = __simd_bit_count_result_t<_Tp, _Vp>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result countr_one(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_countr_one_generator<_Vp, _Result>{__v}};
}

_CCCL_TEMPLATE(typename _Tp,
               typename _Abi,
               typename _Vp     = basic_vec<_Tp, _Abi>,
               typename _Result = __simd_bit_count_result_t<_Tp, _Vp>)
_CCCL_REQUIRES(__cccl_is_unsigned_integer_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr _Result popcount(const basic_vec<_Tp, _Abi>& __v) noexcept
{
  return _Result{__simd_popcount_generator<_Vp, _Result>{__v}};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_BIT_H
