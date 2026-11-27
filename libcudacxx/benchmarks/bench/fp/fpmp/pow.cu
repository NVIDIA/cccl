//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp powers, cube roots and distances (sqrt and rsqrt are in arith.cu).

#include "fpmp_bench.cuh"

FPB_OP2(
  pow, (cuda::pow(a, b)), (::pow(a, b)), (fpb::logr(0.01, 100.0)), (fpb::lin(-10.0, 10.0)), (fpb::perf(0.8f, 7e-7f)))
FPB_OP1(cbrt, cuda::cbrt(a), ::cbrt(a), (fpb::logr(1e-16, 1e16, true)), FPB_PERF_DEFAULT)
FPB_OP1(rcbrt, cuda::rcbrt(a), ::rcbrt(a), (fpb::logr(1e-16, 1e16, true)), FPB_PERF_DEFAULT)
FPB_OP2(hypot,
        (cuda::hypot(a, b)),
        (::hypot(a, b)),
        (fpb::logr(1e-8, 1e8, true)),
        (fpb::logr(1e-8, 1e8, true)),
        FPB_PERF_DEFAULT)
FPB_OP2(rhypot,
        (cuda::rhypot(a, b)),
        (::rhypot(a, b)),
        (fpb::logr(1e-8, 1e8, true)),
        (fpb::logr(1e-8, 1e8, true)),
        FPB_PERF_DEFAULT)

using ops = nvbench::type_list<fpb::op_pow, fpb::op_cbrt, fpb::op_rcbrt, fpb::op_hypot, fpb::op_rhypot>;
FPB_REGISTER(ops);
