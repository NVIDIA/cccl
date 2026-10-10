#include <thrust/execution_policy.h>
#include <thrust/system/cuda/detail/util.h>

#include <thread>

#include <unittest/unittest.h>

void verify_stream()
{
  auto exec   = thrust::device;
  auto stream = thrust::cuda_cub::stream(exec);
  REQUIRE(stream == cudaStreamLegacy);
}

TEST_CASE("TestLegacyDefaultStream", "[stream_legacy]")
{
  verify_stream();

  std::thread t(verify_stream);
  t.join();
}
