#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/equal.h>
#include <thrust/sequence.h>

#include <cuda/devices>
#include <cuda/iterator>
#include <cuda/stream>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

#define CCCL_DISABLE_STREAM_DEVICE_CHECK

// %PARAM% TEST_LAUNCH lid 0:1:2

struct incrementer_t
{
  int* d_counts;

  template <class OffsetT>
  __device__ void operator()(OffsetT i)
  {
    atomicAdd(d_counts + i, 1); // Check if `i` was served more than once
  }
};

C2H_TEST("Device for each n with stream from another device works when using CCCL_DISABLE_STREAM_DEVICE_CHECK",
         "[for][device]")
{
  if (cuda::devices.size() < 2)
  {
    SKIP("Test requires at least 2 CUDA devices");
  }

  cuda::stream stream_on_device_1_wrapper(cuda::devices[1]);
  cudaStream_t stream_on_device_1 = stream_on_device_1_wrapper.release();
  // Copy of the above test but specifying stream from another device
  using offset_t               = int;
  constexpr offset_t max_items = 5000000;
  constexpr offset_t min_items = 1;
  const offset_t num_items     = GENERATE_COPY(
    take(3, random(min_items, max_items)),
    values({
      min_items,
      max_items,
    }));

  const auto it = cuda::counting_iterator<int>{0};
  c2h::device_vector<int> counts(num_items);

  auto result = cub::DeviceFor::ForEach(
    it, it + num_items, incrementer_t{thrust::raw_pointer_cast(counts.data())}, stream_on_device_1);
  REQUIRE(result == cudaSuccess);

  const auto num_of_once_marked_items = static_cast<offset_t>(thrust::count(counts.begin(), counts.end(), 1));
  REQUIRE(num_of_once_marked_items == num_items);

  REQUIRE(cudaStreamDestroy(stream_on_device_1) == cudaSuccess);
}
