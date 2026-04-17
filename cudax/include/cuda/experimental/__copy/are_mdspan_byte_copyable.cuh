//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_ARE_MDSPAN_BYTE_COPYABLE_H
#define _CUDAX__COPY_ARE_MDSPAN_BYTE_COPYABLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__mdspan/default_accessor.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/is_trivially_copyable.h>
#  include <cuda/std/__type_traits/remove_cv.h>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _TpIn,
          typename _TpOut,
          typename _LayoutPolicyIn,
          typename _LayoutPolicyOut,
          typename _AccessorPolicyIn,
          typename _AccessorPolicyOut>
constexpr bool __are_mdspan_byte_copyable() noexcept
{
  using __default_accessor_in  = ::cuda::std::default_accessor<_TpIn>;
  using __default_accessor_out = ::cuda::std::default_accessor<_TpOut>;

  return ::cuda::std::is_same_v<::cuda::std::remove_cv_t<_TpIn>, ::cuda::std::remove_cv_t<_TpOut>>
      && ::cuda::std::is_trivially_copyable_v<_TpIn> //
      && ::cuda::std::is_convertible_v<_AccessorPolicyIn, __default_accessor_in>
      && ::cuda::std::is_convertible_v<_AccessorPolicyOut, __default_accessor_out>;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // _CUDAX__COPY_ARE_MDSPAN_BYTE_COPYABLE_H
