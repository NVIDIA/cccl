//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_ALIGN_UP_H
#define _CUDA___MEMORY_ALIGN_UP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/runtime_assume_aligned.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HAS_BUILTIN(__builtin_align_up)
#  define _CCCL_BUILTIN_ALIGN_UP(...) __builtin_align_up(__VA_ARGS__)
#endif // _CCCL_HAS_BUILTIN(__builtin_align_up)

// nvcc doesn't support this builtin in device code, clang-cuda crashes
#if (_CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(CLANG)) && _CCCL_DEVICE_COMPILATION()
#  undef _CCCL_BUILTIN_ALIGN_UP
#endif // (_CCCL_CUDA_COMPILER(NVCC) || _CCCL_CUDA_COMPILER(CLANG)) && _CCCL_DEVICE_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Tp>
[[nodiscard]] _CCCL_API inline _Tp* align_up(_Tp* __ptr, ::cuda::std::size_t __alignment) noexcept
{
  using ::cuda::std::uintptr_t;
  _CCCL_ASSERT(::cuda::is_power_of_two(__alignment), "alignment must be a power of two");
  if constexpr (!::cuda::std::is_void_v<_Tp>)
  {
    _CCCL_ASSERT(__alignment >= alignof(_Tp), "wrong alignment");
    _CCCL_ASSERT(reinterpret_cast<uintptr_t>(__ptr) % alignof(_Tp) == 0, "ptr is not aligned");
    if (__alignment == alignof(_Tp))
    {
      return __ptr;
    }
  }
#if defined(_CCCL_BUILTIN_ALIGN_UP)
  return (_Tp*) _CCCL_BUILTIN_ALIGN_UP(__ptr, __alignment);
#else // ^^^ _CCCL_BUILTIN_ALIGN_UP ^^^ / vvv !_CCCL_BUILTIN_ALIGN_UP vvv
  // all code below is translated to LOP3.LUT + IADD.64 instructions
  using _Up                = ::cuda::std::remove_cv_t<_Tp>;
  const auto __char_ptr    = reinterpret_cast<char*>(const_cast<_Up*>(__ptr));
  const auto __tmp         = static_cast<uintptr_t>(__alignment - 1);
  const auto __aligned_ptr = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(__ptr) + __tmp) & ~__tmp);
  // __aligned_ptr and __ptr must be pointers (not values) to apply the optimization
  const auto __diff = static_cast<::cuda::std::size_t>(__aligned_ptr - __char_ptr);
  const auto __ret  = reinterpret_cast<_Tp*>(__char_ptr + __diff);
  return ::cuda::std::__runtime_assume_aligned(__ret, __alignment);
#endif // ^^^ !_CCCL_BUILTIN_ALIGN_UP ^^^
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_ALIGN_UP_H
