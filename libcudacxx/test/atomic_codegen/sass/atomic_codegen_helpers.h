//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_TEST_ATOMIC_CODEGEN_SASS_ATOMIC_CODEGEN_HELPERS_H
#define _LIBCUDACXX_TEST_ATOMIC_CODEGEN_SASS_ATOMIC_CODEGEN_HELPERS_H

#include <cuda/atomic>

struct __half;
struct __nv_bfloat16;

using f16  = __half;
using bf16 = __nv_bfloat16;

inline constexpr auto tsb = cuda::thread_scope_block;
inline constexpr auto tsd = cuda::thread_scope_device;
inline constexpr auto tss = cuda::thread_scope_system;

inline constexpr auto mor  = cuda::std::memory_order_relaxed;
inline constexpr auto moa  = cuda::std::memory_order_acquire;
inline constexpr auto more = cuda::std::memory_order_release;
inline constexpr auto moar = cuda::std::memory_order_acq_rel;
inline constexpr auto mosc = cuda::std::memory_order_seq_cst;

#if _CCCL_HAS_INT128()
using i128 = __int128_t;
using u128 = __uint128_t;
#endif // _CCCL_HAS_INT128()

template <typename T, cuda::thread_scope Scope>
using ca = cuda::atomic<T, Scope>;

template <typename T, cuda::thread_scope Scope>
using car = cuda::atomic_ref<T, Scope>;

template <typename T, cuda::thread_scope>
using csa = cuda::std::atomic<T>;

template <typename T, cuda::thread_scope>
using csar = cuda::std::atomic_ref<T>;

#endif // _LIBCUDACXX_TEST_ATOMIC_CODEGEN_SASS_ATOMIC_CODEGEN_HELPERS_H
