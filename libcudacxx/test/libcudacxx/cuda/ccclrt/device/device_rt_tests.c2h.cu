//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/iterator>

#include <testing.cuh>

C2H_CCCLRT_TEST("init", "[device]")
{
  cuda::device_ref dev{0};
  dev.init();
  CCCLRT_REQUIRE(cuda::__driver::__isPrimaryCtxActive(cuda::__driver::__deviceGet(0)));
}

C2H_CCCLRT_TEST("global devices vector", "[device]")
{
  CCCLRT_REQUIRE(cuda::devices.size() > 0);
  CCCLRT_REQUIRE(cuda::devices.begin() != cuda::devices.end());
  CCCLRT_REQUIRE(cuda::devices.begin() == cuda::devices.begin());
  CCCLRT_REQUIRE(cuda::devices.end() == cuda::devices.end());
  CCCLRT_REQUIRE(cuda::devices.size() == static_cast<size_t>(cuda::devices.end() - cuda::devices.begin()));

  CCCLRT_REQUIRE(0 == cuda::devices[0].get());
  CCCLRT_REQUIRE(cuda::device_ref{0} == cuda::devices[0]);

  CCCLRT_REQUIRE(0 == (*cuda::devices.begin()).get());
  CCCLRT_REQUIRE(cuda::device_ref{0} == *cuda::devices.begin());

  CCCLRT_REQUIRE(0 == cuda::devices.begin()->get());
  CCCLRT_REQUIRE(0 == cuda::devices.begin()[0].get());

  if (cuda::devices.size() > 1)
  {
    CCCLRT_REQUIRE(1 == cuda::devices[1].get());
    CCCLRT_REQUIRE(cuda::device_ref{0} != cuda::devices[1].get());

    CCCLRT_REQUIRE(1 == (*cuda::std::next(cuda::devices.begin())).get());
    CCCLRT_REQUIRE(1 == cuda::std::next(cuda::devices.begin())->get());
    CCCLRT_REQUIRE(1 == cuda::devices.begin()[1].get());

    CCCLRT_REQUIRE(
      cuda::devices.size() - 1 == static_cast<cuda::std::size_t>((*cuda::std::prev(cuda::devices.end())).get()));
    CCCLRT_REQUIRE(
      cuda::devices.size() - 1 == static_cast<cuda::std::size_t>(cuda::std::prev(cuda::devices.end())->get()));
    CCCLRT_REQUIRE(cuda::devices.size() - 1 == static_cast<cuda::std::size_t>(cuda::devices.end()[-1].get()));

    auto peers = cuda::devices[0].peers();
    for (auto peer : peers)
    {
      CCCLRT_REQUIRE(cuda::devices[0].has_peer_access_to(peer));
      CCCLRT_REQUIRE(peer.has_peer_access_to(cuda::devices[0]));
    }
  }
}
