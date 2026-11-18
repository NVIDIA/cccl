//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___SIMD_PERMUTE_DYNAMIC_H
#define _CUDA_STD___SIMD_PERMUTE_DYNAMIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/simd.h>
#include <cuda/std/__simd/abi.h>
#include <cuda/std/__simd/exposition.h>
#include <cuda/std/__simd/type_traits.h>
#include <cuda/std/__type_traits/is_integral.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_SIMD

// [simd.permute.dynamic]

template <typename _Vp, typename _Ip>
struct __permute_dynamic_generator
{
  using __value_type _CCCL_NODEBUG = typename _Vp::value_type;

  const _Vp& __v_;
  const _Ip& __indices_;

  template <__simd_size_type _Idx>
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __value_type operator()(__simd_size_constant<_Idx>) const noexcept
  {
    const auto __src = static_cast<__simd_size_type>(__indices_[_Idx]);
    _CCCL_ASSERT(::cuda::in_range(__src, __simd_size_type{0}, _Vp::size()),
                 "cuda::std::simd::permute: indices[i] must be in [0, V::size())");
    return __v_[__src];
  }
};

template <typename _Vp, typename _Ip>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __permute_dynamic_generator<_Vp, _Ip>
__make_permute_dynamic_generator(const _Vp& __v, const _Ip& __indices) noexcept
{
  return __permute_dynamic_generator<_Vp, _Ip>{__v, __indices};
}

_CCCL_TEMPLATE(typename _Tp, typename _Abi, typename _Up, typename _UAbi)
_CCCL_REQUIRES(is_integral_v<_Up>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr resize_t<__simd_size_v<_Up, _UAbi>, basic_vec<_Tp, _Abi>>
permute(const basic_vec<_Tp, _Abi>& __v, const basic_vec<_Up, _UAbi>& __indices)
{
  using __result_t = resize_t<__simd_size_v<_Up, _UAbi>, basic_vec<_Tp, _Abi>>;
  return __result_t{::cuda::std::simd::__make_permute_dynamic_generator(__v, __indices)};
}

_CCCL_TEMPLATE(size_t _Bytes, typename _Abi, typename _Up, typename _UAbi)
_CCCL_REQUIRES(is_integral_v<_Up>)
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr resize_t<__simd_size_v<_Up, _UAbi>, basic_mask<_Bytes, _Abi>>
permute(const basic_mask<_Bytes, _Abi>& __v, const basic_vec<_Up, _UAbi>& __indices)
{
  using __result_t = resize_t<__simd_size_v<_Up, _UAbi>, basic_mask<_Bytes, _Abi>>;
  return __result_t{::cuda::std::simd::__make_permute_dynamic_generator(__v, __indices)};
}

_CCCL_END_NAMESPACE_CUDA_STD_SIMD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___SIMD_PERMUTE_DYNAMIC_H
