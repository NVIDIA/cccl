//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp rounding functions and remainders. These are exact on the pair (the remainders on the
// pair rounded to 53 bits at the default level), so u2 reads 0 unless something broke.

#include "fpmp_bench.cuh"

FPB_OP1(floor, cuda::floor(a), ::floor(a), (fpb::logr(1e-3, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP1(ceil, cuda::ceil(a), ::ceil(a), (fpb::logr(1e-3, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP1(round, cuda::round(a), ::round(a), (fpb::logr(1e-3, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP1(trunc, cuda::trunc(a), ::trunc(a), (fpb::logr(1e-3, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP1(rint, cuda::rint(a), ::rint(a), (fpb::logr(1e-3, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP1(nearbyint, cuda::nearbyint(a), ::nearbyint(a), (fpb::logr(1e-3, 1e6, true)), FPB_PERF_DEFAULT)
FPB_OP2(
  fmod, (cuda::fmod(a, b)), (::fmod(a, b)), (fpb::lin(-1000.0, 1000.0)), (fpb::lin(-1000.0, 1000.0)), FPB_PERF_DEFAULT)
FPB_OP2(remainder,
        (cuda::remainder(a, b)),
        (::remainder(a, b)),
        (fpb::lin(-1000.0, 1000.0)),
        (fpb::lin(-1000.0, 1000.0)),
        FPB_PERF_DEFAULT)

using ops =
  nvbench::type_list<fpb::op_floor,
                     fpb::op_ceil,
                     fpb::op_round,
                     fpb::op_trunc,
                     fpb::op_rint,
                     fpb::op_nearbyint,
                     fpb::op_fmod,
                     fpb::op_remainder>;
FPB_REGISTER(ops);
