//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp inverse trigonometric functions. asin and acos take their performance input inside
// [-1, 1] (the default 1 + tid * 1e-4 would time the NaN return).

#include "fpmp_bench.cuh"

FPB_OP1(asin, cuda::asin(a), ::asin(a), (fpb::lin(-1.0, 1.0)), (fpb::perf(-0.9995f, 3.8e-6f)))
FPB_OP1(acos, cuda::acos(a), ::acos(a), (fpb::lin(-1.0, 1.0)), (fpb::perf(-0.9995f, 3.8e-6f)))
FPB_OP1(atan, cuda::atan(a), ::atan(a), (fpb::logr(1e-8, 1e8, true)), FPB_PERF_DEFAULT)
FPB_OP2(atan2,
        (cuda::atan2(a, b)),
        (::atan2(a, b)),
        (fpb::logr(1e-8, 1e8, true)),
        (fpb::logr(1e-8, 1e8, true)),
        FPB_PERF_DEFAULT)

using ops = nvbench::type_list<fpb::op_asin, fpb::op_acos, fpb::op_atan, fpb::op_atan2>;
FPB_REGISTER(ops);
