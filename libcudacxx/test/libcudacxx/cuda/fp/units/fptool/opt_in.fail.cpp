// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: <cuda/fptool> refuses to compile without CCCL_ENABLE_FPTOOL.
//
//  The feature is opt-in because both of its types carry mutable state at
//  namespace scope - fp_custom's runtime field sizes and fpmp2_stat's counter
//  record - and that state has vague linkage, so one copy is shared by every
//  translation unit that includes the header. An #include left behind after an
//  experiment would therefore put the state, and the counter traffic with it,
//  into a shipping binary. See <cuda/__fp/fptool_common.h>.
//
//  This is the counterpart to the other tests in this directory, which all pass
//  -DCCCL_ENABLE_FPTOOL and would silently keep passing if the gate were
//  removed. Note that the flag is deliberately NOT passed here.
//
//===----------------------------------------------------------------------===//

#include <cuda/fptool>

int main(int, char**)
{
  return 0;
}
