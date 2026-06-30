//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_COMMUNICATORS_NCCL_TEST_HELPERS_CUH
#define CUDAX_TEST_MULTI_GPU_COMMUNICATORS_NCCL_TEST_HELPERS_CUH

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

// Caches a single-process, multi-GPU NCCL communicator world for the life of one test
// case. Setup runs in the constructor and teardown in the destructor. The template parameter
// is required because the C2H fixture macros expand to a templated TEST_CASE_METHOD.
template <class Dummy = void>
class nccl_comm_fixture
{
public:
  nccl_comm_fixture()
  {
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

    comms_.resize(devs.size());

    const ncclResult_t result = ncclCommInitAll(comms_.data(), devs.size(), devs.data());

    INFO("NCCL: " << ncclGetErrorString(result));
    REQUIRE(result == ncclSuccess);

    wrappers_.reserve(comms_.size());
    for (cuda::std::size_t i = 0; i < comms_.size(); ++i)
    {
      wrappers_.emplace_back(comms_[i], cudax::logical_device{devs[i]});
    }
  }

  ~nccl_comm_fixture()
  {
    wrappers_.clear();
    // Wrappers are non-owning; they are declared after comms_ so they destruct first, before we
    // destroy the handles they refer to.
    for (auto comm : comms_)
    {
      if (comm != nullptr)
      {
        static_cast<void>(ncclCommDestroy(comm));
      }
    }
  }

  [[nodiscard]] cuda::std::span<cudax::nccl_communicator_ref> communicators()
  {
    return wrappers_;
  }

  [[nodiscard]] cuda::std::span<const ncclComm_t> handles() const
  {
    return comms_;
  }

private:
  std::vector<ncclComm_t> comms_{};
  std::vector<cudax::nccl_communicator_ref> wrappers_{};
};

#define NCCL_COMM_TEST(NAME, ...) \
  C2H_TEST_WITH_FIXTURE(::nccl_test_util::nccl_comm_fixture, NAME, "[multi_gpu][nccl]", __VA_ARGS__)
} // namespace nccl_test_util

#endif // CUDAX_TEST_MULTI_GPU_COMMUNICATORS_NCCL_TEST_HELPERS_CUH
