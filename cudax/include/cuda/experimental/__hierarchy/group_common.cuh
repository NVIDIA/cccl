//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_COMMON_CUH
#define _CUDA_EXPERIMENTAL___GROUP_COMMON_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/span>

#include <cuda/experimental/__hierarchy/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <::cuda::std::size_t _Np>
struct group_by_t
{
  static_assert(_Np > 0, "_Np must be greater than 0");

  _CCCL_HIDE_FROM_ABI explicit group_by_t() = default;
};

template <::cuda::std::size_t _Np>
_CCCL_GLOBAL_CONSTANT group_by_t<_Np> group_by;

class group_as
{
  ::cuda::std::span<const unsigned> __vs_;

public:
  _CCCL_API explicit constexpr group_as(::cuda::std::span<const unsigned> __vs) noexcept
      : __vs_{__vs}
  {}

  [[nodiscard]] _CCCL_API constexpr ::cuda::std::span<const unsigned> get() const noexcept
  {
    return __vs_;
  }
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_COMMON_CUH
