//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ADDRESS_SPACE_H
#define _CUDA___MEMORY_ADDRESS_SPACE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <cuda/std/__utility/to_underlying.h>

#  include <nv/target>

#  include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_DEVICE

enum class address_space
{
  global,
  shared,
  constant,
  local,
  grid_constant,
  cluster_shared,
  __max,
};

[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr bool __cccl_is_valid_address_space(address_space __space) noexcept
{
  const auto __v = _CUDA_VSTD::to_underlying(__space);
  return __v >= 0 && __v < _CUDA_VSTD::to_underlying(address_space::__max);
}

[[nodiscard]] _CCCL_FORCEINLINE _CCCL_VISIBILITY_HIDDEN _CCCL_DEVICE bool
is_address_from(address_space __space, const void* __ptr)
{
  _CCCL_ASSERT(__ptr != nullptr, "invalid pointer");
  _CCCL_ASSERT(_CUDA_DEVICE::__cccl_is_valid_address_space(__space), "invalid address space");

  // NVCC and NVRTC < 12.3 have problems tracking the address space of pointers, fallback to inline PTX for them
  switch (__space)
  {
    case address_space::global:
#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3)
    {
      unsigned __ret;
      asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  isspacep.global p, %1;\n\t"
        "  selp.u32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(__ret)
        : "l"(__ptr));
      return __ret;
    }
#  else // ^^^ _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^ /
        // vvv !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) vvv
      return ::__isGlobal(__ptr);
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^
    case address_space::shared:
#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3)
    {
      unsigned __ret;
      asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  isspacep.shared p, %1;\n\t"
        "  selp.u32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(__ret)
        : "l"(__ptr));
      return __ret;
    }
#  else // ^^^ _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^ /
        // vvv !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) vvv
      return ::__isShared(__ptr);
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^
    case address_space::constant:
#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3)
    {
      unsigned __ret;
      asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  isspacep.constant p, %1;\n\t"
        "  selp.u32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(__ret)
        : "l"(__ptr));
      return __ret;
    }
#  else // ^^^ _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^ /
        // vvv !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) vvv
      return ::__isConstant(__ptr);
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^
    case address_space::local:
#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3)
    {
      unsigned __ret;
      asm volatile(
        "{\n\t"
        "  .reg .pred p;\n\t"
        "  isspacep.local p, %1;\n\t"
        "  selp.u32 %0, 1, 0, p;\n\t"
        "}\n\t"
        : "=r"(__ret)
        : "l"(__ptr));
      return __ret;
    }
#  else // ^^^ _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^ /
        // vvv !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) vvv
      return ::__isLocal(__ptr);
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^
    case address_space::grid_constant:
#  if _CCCL_HAS_GRID_CONSTANT()
#    if _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3)
    {
      unsigned __ret;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_70,
        (asm volatile("{\n\t"
                      "  .reg .pred p;\n\t"
                      "  isspacep.param p, %1;\n\t"
                      "  selp.u32 %0, 1, 0, p;\n\t"
                      "}\n\t" : "=r"(__ret) : "l"(__ptr));
         return __ret;),
        (return false;))
    }
#    else // ^^^ _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^ /
          // vvv !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) vvv
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_70, (return ::__isGridConstant(__ptr);), (return false;))
#    endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^
#  else // ^^^ _CCCL_HAS_GRID_CONSTANT() ^^^ / vvv !_CCCL_HAS_GRID_CONSTANT() vvv
      return false;
#  endif // ^^^ !_CCCL_HAS_GRID_CONSTANT() ^^^
    case address_space::cluster_shared:
#  if _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3)
    {
      unsigned __ret;
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_90,
        (asm volatile("{\n\t"
                      "  .reg .pred p;\n\t"
                      "  isspacep.shared::cluster p, %1;\n\t"
                      "  selp.u32 %0, 1, 0, p;\n\t"
                      "}\n\t" : "=r"(__ret) : "l"(__ptr));
         return __ret;),
        (return false;))
    }
#  else // ^^^ _CCCL_CUDA_COMPILER(NVCC, <, 12, 3) || _CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^ /
        // vvv !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) vvv
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (return ::__isClusterShared(__ptr);), (return false;))
#  endif // ^^^ !_CCCL_CUDA_COMPILER(NVCC, <, 12, 3) && !_CCCL_CUDA_COMPILER(NVRTC, <, 12, 3) ^^^
    default:
      return false;
  }
}

_LIBCUDACXX_END_NAMESPACE_CUDA_DEVICE

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDA_COMPILATION()

#endif // _CUDA___MEMORY_ADDRESS_SPACE_H
