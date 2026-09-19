//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Strictness of the CUDASTF_FAKE_LOCALITY_DOMAINS override
 *
 * Requests a fake topology that no device can provide (vastly more domains
 * than there are multiprocessors, so the test adapts to any hardware) and
 * checks that locality-domain queries and factories throw a typed exception
 * instead of silently reporting a smaller topology.
 */

#include <cuda/experimental/stf.cuh>

#include <cstdlib>
#include <stdexcept>

using namespace cuda::experimental::stf;

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr,
          "Green contexts are not supported by this version of CUDA: the fake topology "
          "override is inactive, test waived.\n");
  return 0;
#else // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^ / vvv _CCCL_CTK_AT_LEAST(12, 4) vvv
  // Far beyond any device's multiprocessor count, hence never fulfillable
  // (each fake domain needs at least one SM group).
#  if _CCCL_COMPILER(MSVC)
  EXPECT(_putenv_s("CUDASTF_FAKE_LOCALITY_DOMAINS", "1000000") == 0);
#  else // ^^^ MSVC ^^^ / vvv POSIX vvv
  EXPECT(setenv("CUDASTF_FAKE_LOCALITY_DOMAINS", "1000000", 1) == 0);
#  endif // !MSVC

  const int dev = 0;

  int ndevs = 0;
  if (cudaGetDeviceCount(&ndevs) != cudaSuccess || ndevs == 0)
  {
    fprintf(stderr, "No CUDA device: test waived.\n");
    return 0;
  }

  // The count query must throw, not clamp
  bool threw = false;
  try
  {
    const unsigned int n = locality_domain_count(dev);
    fprintf(stderr, "Unexpectedly got a count of %u domains.\n", n);
  }
  catch (const ::std::runtime_error& e)
  {
    threw = true;
    fprintf(stderr, "Got expected error: %s\n", e.what());
  }
  EXPECT(threw);

  // The place factories must throw as well, on every build type
  threw = false;
  try
  {
    exec_place ep = exec_place::locality_domain(dev, 0);
    (void) ep;
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);

  threw = false;
  try
  {
    data_place dp = data_place::locality_domain(dev, 0);
    (void) dp;
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);

  // Invalid device ordinals are rejected (device validation precedes
  // topology fulfillment)
  bool threw_invalid = false;
  try
  {
    locality_domain_count(-1);
  }
  catch (...)
  {
    threw_invalid = true;
  }
  EXPECT(threw_invalid);

  bool threw_oob = false;
  try
  {
    locality_domain_count(ndevs);
  }
  catch (...)
  {
    threw_oob = true;
  }
  EXPECT(threw_oob);

  return 0;
#endif // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^
}
