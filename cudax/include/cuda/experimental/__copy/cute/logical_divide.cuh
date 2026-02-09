//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_LOGICAL_DIVIDE_H
#define __CUDAX_COPY_CUTE_LOGICAL_DIVIDE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/cute/complement.cuh>

#  include <cute/layout.hpp>
#  include <cute/tensor_impl.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Run-time version of CuTe `logical_divide` for layouts that may have dynamic strides in the tiler.
 */
template <class _LShape, class _LStride, class _Shape, class _Stride>
[[nodiscard]] _CCCL_HOST_API auto __logical_divide(
  const ::cute::Layout<_LShape, _LStride>& __layout, const ::cute::Layout<_Shape, _Stride>& __tiler) noexcept
{
  if constexpr (::cute::is_static<_Stride>::value)
  {
    return ::cute::logical_divide(__layout, __tiler);
  }
  else
  {
    const auto __codomain_size = static_cast<::cuda::std::int64_t>(::cute::size(__layout));
    const auto __complement    = ::cuda::experimental::__complement(__tiler, __codomain_size);
    return ::cute::composition(__layout, ::cute::make_layout(__tiler, __complement));
  }
}

/**
 * @brief Overload of `__logical_divide` for CuTe tensors.
 */
template <class _Engine, class _Layout, class _Shape, class _Stride>
[[nodiscard]] _CCCL_HOST_API auto __logical_divide(
  const ::cute::Tensor<_Engine, _Layout>& __tensor, const ::cute::Layout<_Shape, _Stride>& __tiler) noexcept
{
  return ::cute::make_tensor(
    __tensor.data(), ::cuda::experimental::__logical_divide(__tensor.layout(), __tiler));
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_LOGICAL_DIVIDE_H
