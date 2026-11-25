// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/stream>

#include <cuda/experimental/mgmn/nccl_communicators>
#include <cuda/experimental/mgmn/sort>

#include <cstdlib>

#include <testing.cuh>

#include <c2h/catch2_test_helper.h>

namespace optional_link_test
{
// This target deliberately does not link NCCL. Keep a complete sort
// instantiation in the binary, including its device-capability branch.
void instantiate(cudax::mgmn::nccl_communicator_ref comm, cuda::stream_ref stream, int* keys)
{
  cudax::mgmn::sort(cudax::distributed, comm, stream, keys, 1);
}
} // namespace optional_link_test

TEST_CASE("sort, NCCL remains a runtime dependency", "[multi_gpu][nccl]")
{
  auto* volatile implementation = &optional_link_test::instantiate;
  REQUIRE(implementation != nullptr);
#if defined(_CUDAX_HAS_NCCL_DEVICE_API)
  if (std::getenv("CUDAX_TEST_EXPECT_DEVICE_API"))
  {
    REQUIRE(cuda::experimental::__nccl::__get_device_api().__supports_lsa());
    REQUIRE(cuda::experimental::__nccl::__get_device_api().__supports_gin());
  }
  // Allows a compatibility runner to supply an older/stub libnccl without
  // changing the production loader or adding a force-host option to sort.
  if (std::getenv("CUDAX_TEST_EXPECT_NO_DEVICE_API"))
  {
    REQUIRE_FALSE(cuda::experimental::__nccl::__get_device_api().__supports_lsa());
    REQUIRE_FALSE(cuda::experimental::__nccl::__get_device_api().__supports_gin());
  }
#else
  if (std::getenv("CUDAX_TEST_EXPECT_NO_DEVICE_API"))
  {
    SKIP("Optional device-symbol lookup requires the device SDK headers");
  }
#endif
}
