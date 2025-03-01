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

// Initializing the primary template is ill-formed.
int log2e{cuda::std::numbers::log2e_v<int>}; // expected-error-re@numbers:* {{[math.constants] A program that
                                             // instantiates a primary template of a mathematical constant variable
                                             // template is ill-formed.}}
int log10e{cuda::std::numbers::log10e_v<int>};
int pi{cuda::std::numbers::pi_v<int>};
int inv_pi{cuda::std::numbers::inv_pi_v<int>};
int inv_sqrtpi{cuda::std::numbers::inv_sqrtpi_v<int>};
int ln2{cuda::std::numbers::ln2_v<int>};
int ln10{cuda::std::numbers::ln10_v<int>};
int sqrt2{cuda::std::numbers::sqrt2_v<int>};
int sqrt3{cuda::std::numbers::sqrt3_v<int>};
int inv_sqrt3{cuda::std::numbers::inv_sqrt3_v<int>};
int egamma{cuda::std::numbers::egamma_v<int>};
int phi{cuda::std::numbers::phi_v<int>};
