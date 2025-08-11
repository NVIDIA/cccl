//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_DISCARD_MEMORY_H
#define _CUDA___MEMORY_DISCARD_MEMORY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/address_space.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

_CCCL_API inline void discard_memory([[maybe_unused]] volatile void* __ptr, [[maybe_unused]] size_t __nbytes) noexcept
{
  // The discard PTX instruction is only available with PTX ISA 7.4 and later
#if __cccl_ptx_isa >= 740ULL
  // clang-format off
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (_CCCL_ASSERT(__ptr != nullptr, "null pointer passed to discard_memory");
    if (!_CUDA_DEVICE::is_address_from(__ptr, _CUDA_DEVICE::address_space::global)) {
      return;
    }
    constexpr size_t __line_size = 128;
    auto __p                     = reinterpret_cast<char*>(const_cast<void*>(__ptr));
    auto __end_p                 = __p + __nbytes;
    // Trim the first block and last block if they're not 128 bytes aligned
    const auto __misalignment  = reinterpret_cast<uintptr_t>(__p) % __line_size;
    const auto __start_aligned = __misalignment == 0 ? __p : __p + (__line_size - __misalignment);
    const auto __end_aligned   = __end_p - (reinterpret_cast<uintptr_t>(__end_p) % __line_size);
    for (auto __i = __start_aligned; __i < __end_aligned; __i += __line_size) {
      asm volatile("discard.global.L2 [%0], 128;" ::"l"(__i) :);
    }))
  // clang-format off
#endif // __cccl_ptx_isa >= 740ULL
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_DISCARD_MEMORY_H
