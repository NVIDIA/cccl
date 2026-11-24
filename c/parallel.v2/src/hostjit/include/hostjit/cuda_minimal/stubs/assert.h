//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

// Minimal freestanding-mode stub for <assert.h>.
//
// CUDA toolkit headers pulled in via libcudacxx's __floating_point/cuda_fp_types.h
// (e.g. cuda_fp8.hpp) include <assert.h> unconditionally. In the JIT compile
// environment we have no libc; treat assert(expr) as a no-op. This matches the
// effect of `-DNDEBUG`, which CCCL/CUB device code already expects.
#ifndef _HOSTJIT_ASSERT_H
#define _HOSTJIT_ASSERT_H

#ifdef __cplusplus
extern "C" {
#endif

#undef assert
#define assert(expr) ((void) 0)

#if defined(__CUDA__) && defined(__clang__) && defined(__cplusplus)
// Keep host verification independent of libc in the freestanding environment.
// Clang compiles HostJIT code on every platform and provides the trap builtin.
// This is the host counterpart of the __device__ __assert_fail shim in
// __clang_cuda_runtime_wrapper.h. Hidden visibility binds the JIT library's
// calls to this definition at link time; with default visibility the dynamic
// linker would let the host process's glibc __assert_fail interpose it on
// Linux, and the library would depend on libc after all.
__attribute__((host, noreturn, visibility("hidden"))) inline void
__assert_fail(const char*, const char*, unsigned, const char*) noexcept
{
  __builtin_trap();
}
#endif

#ifdef __cplusplus
}
#endif

#endif // _HOSTJIT_ASSERT_H
