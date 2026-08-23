//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// P3323R1: atomic_ref<volatile T> is ill-formed when T is not always lock-free.

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
// UNSUPPORTED: force-tile

#include <cuda/atomic>

struct alignas(16) not_always_lock_free
{
  long long data[2];
};

volatile not_always_lock_free value{};

// expected-error@*:* {{requires T to be always lock-free}}
cuda::std::atomic_ref<volatile not_always_lock_free> std_ref(value);
// expected-error@*:* {{requires T to be always lock-free}}
cuda::atomic_ref<volatile not_always_lock_free> cuda_ref(value);
