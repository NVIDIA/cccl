//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp hyperbolic functions and their inverses.

#include "fpmp_bench.cuh"

FPB_OP1(sinh, cuda::sinh(a), ::sinh(a), (fpb::lin(-64.0, 64.0)), (fpb::perf(0.0f, 4e-6f)))
FPB_OP1(cosh, cuda::cosh(a), ::cosh(a), (fpb::lin(-64.0, 64.0)), FPB_PERF_DEFAULT)
FPB_OP1(tanh, cuda::tanh(a), ::tanh(a), (fpb::lin(-20.0, 20.0)), (fpb::perf(0.01f, 1e-5f)))
FPB_OP1(asinh, cuda::asinh(a), ::asinh(a), (fpb::logr(1e-6, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP1(acosh, cuda::acosh(a), ::acosh(a), (fpb::logr(1.0, 1e6)), FPB_PERF_DEFAULT)
FPB_OP1(atanh, cuda::atanh(a), ::atanh(a), (fpb::lin(-0.99, 0.99)), (fpb::perf(0.1f, 7e-7f)))

using ops = nvbench::type_list<fpb::op_sinh, fpb::op_cosh, fpb::op_tanh, fpb::op_asinh, fpb::op_acosh, fpb::op_atanh>;
FPB_REGISTER(ops);
