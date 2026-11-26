//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp trigonometric functions in radians and in units of pi (sincos and sincospi return two
// values and are not measured here).

#include "fpmp_bench.cuh"

FPB_OP1(sin, cuda::sin(a), ::sin(a), (fpb::lin(-1e4, 1e4)), FPB_PERF_DEFAULT)
FPB_OP1(cos, cuda::cos(a), ::cos(a), (fpb::lin(-1e4, 1e4)), FPB_PERF_DEFAULT)
FPB_OP1(tan, cuda::tan(a), ::tan(a), (fpb::lin(-1e4, 1e4)), FPB_PERF_DEFAULT)
FPB_OP1(sinpi, cuda::sinpi(a), ::sinpi(a), (fpb::lin(-1e4, 1e4)), FPB_PERF_DEFAULT)
FPB_OP1(cospi, cuda::cospi(a), ::cospi(a), (fpb::lin(-1e4, 1e4)), FPB_PERF_DEFAULT)

using ops = nvbench::type_list<fpb::op_sin, fpb::op_cos, fpb::op_tan, fpb::op_sinpi, fpb::op_cospi>;
FPB_REGISTER(ops);
