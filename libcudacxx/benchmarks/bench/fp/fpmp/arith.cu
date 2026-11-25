//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp arithmetic: the pair operations and the roots. The accuracy inputs keep add and sub
// (and fma) away from cancellation, about which the arithmetic promises nothing.

#include "fpmp_bench.cuh"

FPB_OP2(add, a + b, a + b, (fpb::logr(1e-3, 1e3)), (fpb::logr(1e-3, 1e3)), FPB_PERF_DEFAULT)
FPB_OP2(sub, a - b, a - b, (fpb::logr(2.0, 16.0)), (fpb::logr(0.125, 0.5)), FPB_PERF_DEFAULT)
FPB_OP2(mul, a* b, a* b, (fpb::logr(1e-8, 1e8, true)), (fpb::logr(1e-8, 1e8, true)), FPB_PERF_DEFAULT)
FPB_OP2(div, a / b, a / b, (fpb::logr(1e-8, 1e8, true)), (fpb::logr(1e-8, 1e8, true)), FPB_PERF_DEFAULT)
FPB_OP3(fma,
        (cuda::fma(a, b, c)),
        (::fma(a, b, c)),
        (fpb::logr(1e-4, 1e4)),
        (fpb::logr(1e-4, 1e4)),
        (fpb::logr(1e-4, 1e4)),
        FPB_PERF_DEFAULT)
FPB_OP1(sqrt, cuda::sqrt(a), ::sqrt(a), (fpb::logr(1e-16, 1e16)), FPB_PERF_DEFAULT)
FPB_OP1(rsqrt, cuda::rsqrt(a), ::rsqrt(a), (fpb::logr(1e-16, 1e16)), FPB_PERF_DEFAULT)

using ops =
  nvbench::type_list<fpb::op_add, fpb::op_sub, fpb::op_mul, fpb::op_div, fpb::op_fma, fpb::op_sqrt, fpb::op_rsqrt>;
FPB_REGISTER(ops);
