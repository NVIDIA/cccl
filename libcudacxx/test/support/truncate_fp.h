//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

TEST_HOST_DEVICE
inline long double truncate_fp(long double val) {
  volatile long double sink = val;
  return sink;
}

TEST_HOST_DEVICE
inline double truncate_fp(double val) {
  volatile double sink = val;
  return sink;
}

TEST_HOST_DEVICE
inline float truncate_fp(float val) {
  volatile float sink = val;
  return sink;
}
