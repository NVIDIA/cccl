//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_NCCL_TEST_COMMON_H
#define CUDAX_TEST_MULTI_NCCL_TEST_COMMON_H

#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>
#include <cuda/experimental/stream.cuh>

#include <vector>

#include <nccl.h>

#include <c2h/catch2_test_helper.h>

namespace cudax = ::cuda::experimental;

namespace nccl_test_util
{
// One stream per rank, each current on its own device.
[[nodiscard]] inline std::vector<cudax::stream> make_streams()
{
  return {cuda::devices.begin(), cuda::devices.end()};
}

class nccl_communicator : public cudax::nccl_communicator_ref
{
public:
  using nccl_communicator_ref::nccl_communicator_ref;

  // These are never allowed
  nccl_communicator(const nccl_communicator&)            = delete;
  nccl_communicator& operator=(const nccl_communicator&) = delete;
  // These should be allowed once we make this a proper class, but need to be able to reach
  // into nccl_communicator_ref to work properly
  nccl_communicator(nccl_communicator&&)            = delete;
  nccl_communicator& operator=(nccl_communicator&&) = delete;

  ~nccl_communicator()
  {
    static_cast<void>(ncclCommDestroy(native_handle()));
  }
};

[[nodiscard]] inline const std::vector<nccl_communicator>& nccl_comms()
{
  static const auto comms = []() -> std::vector<nccl_communicator> {
    if (cuda::devices.size() == 0)
    {
      SKIP("No CUDA devices visible");
    }

    std::vector<int> devs;

    devs.reserve(cuda::devices.size());
    for (auto d : cuda::devices)
    {
      devs.emplace_back(d.get());
    }

    std::vector<ncclComm_t> raw_comms(devs.size());

    const ncclResult_t result = ncclCommInitAll(raw_comms.data(), static_cast<int>(devs.size()), devs.data());

    INFO("NCCL: " << ncclGetErrorString(result));
    REQUIRE(result == ncclSuccess);

    return {raw_comms.begin(), raw_comms.end()};
  }();

  return comms;
}

// Caches a single-process, multi-GPU NCCL communicator world for the life of the entire test
// suite.
template <class = void>
class nccl_comm_fixture
{
public:
  [[nodiscard]] cuda::std::span<cudax::nccl_communicator_ref> communicators()
  {
    return wrappers_;
  }

private:
  std::vector<cudax::nccl_communicator_ref> wrappers_{nccl_comms().begin(), nccl_comms().end()};
};

#define MULTI_GPU_TEST(NAME, ...) \
  C2H_TEST_WITH_FIXTURE(::nccl_test_util::nccl_comm_fixture, NAME, "[multi_gpu][nccl]", __VA_ARGS__)
} // namespace nccl_test_util

#endif // CUDAX_TEST_MULTI_GPU_NCCL_TEST_COMMON_H
