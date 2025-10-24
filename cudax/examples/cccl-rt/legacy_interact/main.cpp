#define _CCCL_IMPLICIT_SYSTEM_HEADER_GCC

#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/std/ranges>
#include <cuda/std/source_location>
#include <cuda/stream>

#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <cstdio>

#include "legculib.h"

namespace cudax = cuda::experimental;

//! @brief Function for checking the return value of legculib calls.
void safe_call(legculib_result result, cuda::std::source_location loc = cuda::std::source_location::current())
{
  switch (result.error)
  {
    case LEGCULIB_SUCCESS:
      return;
    case LEGCULIB_ERROR_CUDA_DRIVER:
      std::fprintf(stderr,
                   "%s(%u): legculib CUDA driver error: %s\n",
                   loc.file_name(),
                   static_cast<unsigned>(loc.line()),
                   cudaGetErrorString(result.cuda_error));
      std::exit(1);
    default:
      std::fprintf(stderr,
                   "%s(%u): legculib error: %s\n",
                   loc.file_name(),
                   static_cast<unsigned>(loc.line()),
                   legculib_get_error_string(result.error));
      std::exit(1);
  }
}

int main()
try
{
  // Check that there is a CUDA device.
  if (cuda::devices.size() == 0)
  {
    std::fprintf(stderr, "No CUDA devices were found.\n");
    return 1;
  }

  // We will work with the first device.
  cuda::device_ref device = cuda::devices[0];

  // Initialize the legacy library for the device.
  safe_call(legculib_init(device.get()));

  // Create a stream for the device.
  cuda::stream stream{device};

  // Get the legculib's sum of products kernel compile.
  safe_call(legculib_sum_of_products(nullptr, nullptr, nullptr, nullptr, 0, LEGCULIB_I32));

  // We will use the default pinned and device memory resources for host and device allocations.
  cudax::pinned_memory_resource host_mr{};
  cudax::device_memory_resource device_mr{device};

  constexpr unsigned n = 1024 * 1024;
  auto host_lhs        = cudax::make_async_buffer<float>(stream, host_mr, cudax::no_init);
  auto host_rhs        = cudax::make_async_buffer<float>(stream, host_mr, cudax::no_init);
  auto host_result     = cudax::make_async_buffer<float>(stream, host_mr, cudax::no_init);
  stream.sync();

  // todo: fill lhs and rhs

  auto device_lhs    = cudax::make_async_buffer<float>(stream, device_mr);
  auto device_rhs    = cudax::make_async_buffer<float>(stream, device_mr);
  auto device_result = cudax::make_async_buffer<float>(stream, device_mr);
  stream.sync();

  // Record the start of the computation.
  const auto start = stream.record_timed_event();

  // Launch the sum of products kernel from the legacy library into the stream.
  safe_call(legculib_sum_of_products(
    stream.get(), device_result.data(), device_lhs.data(), device_rhs.data(), n, LEGCULIB_F32));

  // Record the end of the computation.
  const auto end = stream.record_timed_event();

  // Get the elapsed time.
  const auto elapsed_time = end - start;

  // Print the results.
  std::printf("Result:       %d\n", *host_result.begin());
  std::printf("Elapsed time: %lld", elapsed_time.count());

  // Deinitialize the legacy library.
  safe_call(legculib_finalize());
}
catch (const cuda::cuda_error& e)
{
  std::fprintf(stderr, "CUDA error: %s\n", e.what());
  return 1;
}
catch (const std::exception& e)
{
  std::fprintf(stderr, "An unknown error: %s\n", e.what());
  return 1;
}
catch (...)
{
  std::fprintf(stderr, "An unknown error occurred\n");
  return 1;
}
