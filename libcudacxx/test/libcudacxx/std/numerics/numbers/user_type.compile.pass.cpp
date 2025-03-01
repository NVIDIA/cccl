//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/numbers>

#include <cuda/std/numbers>

struct UserType
{
  int value;
};

template <>
UserType cuda::std::numbers::e_v<UserType>{};

template <>
UserType cuda::std::numbers::log2e_v<UserType>{};

template <>
UserType cuda::std::numbers::log10e_v<UserType>{};

template <>
UserType cuda::std::numbers::pi_v<UserType>{};

template <>
UserType cuda::std::numbers::inv_pi_v<UserType>{};

template <>
UserType cuda::std::numbers::inv_sqrtpi_v<UserType>{};

template <>
UserType cuda::std::numbers::ln2_v<UserType>{};

template <>
UserType cuda::std::numbers::ln10_v<UserType>{};

template <>
UserType cuda::std::numbers::sqrt2_v<UserType>{};

template <>
UserType cuda::std::numbers::sqrt3_v<UserType>{};

template <>
UserType cuda::std::numbers::inv_sqrt3_v<UserType>{};

template <>
UserType cuda::std::numbers::egamma_v<UserType>{};

template <>
UserType cuda::std::numbers::phi_v<UserType>{};

int main(int, char**)
{
  return 0;
}
