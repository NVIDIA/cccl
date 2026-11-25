//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// fpmp error functions, their inverses, the normal distribution and the Boys function. The
// performance inputs of erfinv and erfcinv stay inside their domains (the suite's
// 0.01 + tid * 1e-5 leaves them at tid > 10^5). boys_f0 has no double counterpart in CUDA's
// math library; its reference is the closed form sqrt(pi / x) erf(sqrt(x)) / 2.

#include "fpmp_bench.cuh"

FPB_OP1(erf, cuda::erf(a), ::erf(a), (fpb::lin(-6.0, 6.0)), (fpb::perf(0.01f, 1e-5f)))
FPB_OP1(erfc, cuda::erfc(a), ::erfc(a), (fpb::lin(-6.0, 6.0)), (fpb::perf(0.01f, 1e-5f)))
FPB_OP1(erfcx, cuda::erfcx(a), ::erfcx(a), (fpb::lin(-9.0, 60.0)), (fpb::perf(0.5f, 5e-5f)))
FPB_OP1(erfinv, cuda::erfinv(a), ::erfinv(a), (fpb::lin(-1.0 + 1e-7, 1.0 - 1e-7)), (fpb::perf(-0.9995f, 3.8e-6f)))
FPB_OP1(erfcinv, cuda::erfcinv(a), ::erfcinv(a), (fpb::lin(1e-7, 2.0 - 1e-7)), (fpb::perf(0.0005f, 3.8e-6f)))
FPB_OP1(normcdf, cuda::normcdf(a), ::normcdf(a), (fpb::lin(-11.5, 8.4)), (fpb::perf(-6.0f, 1e-4f)))
FPB_OP1(normcdfinv, cuda::normcdfinv(a), ::normcdfinv(a), (fpb::lin(1e-6, 1.0 - 1e-6)), (fpb::perf(0.5f, 1e-7f)))
FPB_OP1(boys_f0,
        cuda::boys_f0(a),
        (0.5 * ::sqrt(3.14159265358979323846 / a) * ::erf(::sqrt(a))),
        (fpb::logr(1e-10, 50.0)),
        (fpb::perf(0.01f, 6e-5f)))

using ops =
  nvbench::type_list<fpb::op_erf,
                     fpb::op_erfc,
                     fpb::op_erfcx,
                     fpb::op_erfinv,
                     fpb::op_erfcinv,
                     fpb::op_normcdf,
                     fpb::op_normcdfinv,
                     fpb::op_boys_f0>;
FPB_REGISTER(ops);
