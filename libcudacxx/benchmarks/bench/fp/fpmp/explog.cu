//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp exponentials and logarithms. The accuracy ranges are the cccl-fpdev work ranges of
// the fp32mp2 cells; exp10's performance input stays below its fp32 overflow at 38.5.

#include "fpmp_bench.cuh"

FPB_OP1(exp, cuda::exp(a), ::exp(a), (fpb::lin(-64.0, 64.0)), FPB_PERF_DEFAULT)
FPB_OP1(exp2, cuda::exp2(a), ::exp2(a), (fpb::lin(-90.0, 90.0)), FPB_PERF_DEFAULT)
FPB_OP1(exp10, cuda::exp10(a), ::exp10(a), (fpb::lin(-28.0, 38.0)), (fpb::perf(-19.0f, 7.2e-5f)))
FPB_OP1(expm1, cuda::expm1(a), ::expm1(a), (fpb::lin(-64.0, 64.0)), FPB_PERF_DEFAULT)
FPB_OP1(log, cuda::log(a), ::log(a), (fpb::logr(1e-16, 1e16)), FPB_PERF_DEFAULT)
FPB_OP1(log2, cuda::log2(a), ::log2(a), (fpb::logr(1e-16, 1e16)), FPB_PERF_DEFAULT)
FPB_OP1(log10, cuda::log10(a), ::log10(a), (fpb::logr(1e-16, 1e16)), FPB_PERF_DEFAULT)
FPB_OP1(log1p, cuda::log1p(a), ::log1p(a), (fpb::logr(1e-10, 1e6)), FPB_PERF_DEFAULT)

using ops = nvbench::
  type_list<fpb::op_exp, fpb::op_exp2, fpb::op_exp10, fpb::op_expm1, fpb::op_log, fpb::op_log2, fpb::op_log10, fpb::op_log1p>;
FPB_REGISTER(ops);
