#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/std/ranges>
#include <cuda/std/source_location>
#include <cuda/stream>

#include <cuda/experimental/algorithm.cuh>
#include <cuda/experimental/container.cuh>
#include <cuda/experimental/memory_resource.cuh>

#include <algorithm>
#include <cstdio>
#include <new>
#include <random>

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

//! @brief Function that fills the range with random values.
template <class It>
void fill_with_random(It start, It end)
{
  static std::random_device device{};
  std::mt19937 engine{device()};
  std::uniform_real_distribution dist{0.f, 1.f};
  std::generate(start, end, [&]() {
    return dist(engine);
  });
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
  safe_call(legculib_sum_of_products(nullptr, nullptr, nullptr, nullptr, 0, LEGCULIB_F32));

  // The size of the buffers.
  constexpr unsigned n = 512 * 1024;

  // Allocate uninitialized host buffers.
  auto host_lhs = cudax::make_async_buffer<float>(stream, cudax::pinned_default_memory_pool(), n, cudax::no_init);
  auto host_rhs = cudax::make_async_buffer<float>(stream, cudax::pinned_default_memory_pool(), n, cudax::no_init);

  // Wait for the allocation before touching the buffers.
  stream.sync();

  // Fill the host buffers with data.
  fill_with_random(host_lhs.begin(), host_lhs.end());
  fill_with_random(host_rhs.begin(), host_rhs.end());

  // We will use legculib's device memory pool, so we need to obtain the handle first.
  cudaMemPool_t legculib_device_mempool{};
  safe_call(legculib_get_device_mempool(&legculib_device_mempool));

  // Create the device memory pool reference to the legculib's device memory pool.
  cudax::device_memory_pool_ref device_mempool{legculib_device_mempool};

  // Allocate and initialize device buffers with data from host.
  auto device_lhs = cudax::make_async_buffer<float>(stream, device_mempool, host_lhs);
  auto device_rhs = cudax::make_async_buffer<float>(stream, device_mempool, host_rhs);

  // Allocate buffer for the result on host and device separately.
  auto host_result   = cudax::make_async_buffer<float>(stream, cudax::pinned_default_memory_pool(), 1, cudax::no_init);
  auto device_result = cudax::make_async_buffer<float>(stream, device_mempool, 1, cudax::no_init);

  // Record the start of the computation.
  const auto start = stream.record_timed_event();

  // Launch the sum of products kernel from the legacy library into the stream.
  safe_call(legculib_sum_of_products(
    stream.get(), device_result.data(), device_lhs.data(), device_rhs.data(), n, LEGCULIB_F32));

  // Record the end of the computation.
  const auto end = stream.record_timed_event();

  // Copy back the result from device to host.
  cudax::copy_bytes(stream, device_result, host_result);

  // Wait for all of the enqueued work to finish.
  stream.sync();

  // Get the elapsed time.
  const auto elapsed_time = end - start;

  // Print the results.
  std::printf("Result:            %f\n", *host_result.begin());
  std::printf("Elapsed time [ms]: %f\n", static_cast<float>(elapsed_time.count()) / 1'000'000);

  // We need to free all of the memory allocated from the legculib's device memory pool before we close the legculib's
  // session. The destruction is asynchronous and will be launched to the buffer.stream() stream.
  device_result.destroy();
  device_lhs.destroy();
  device_rhs.destroy();

  // Wait for the buffer.destroy() operations to complete before closing the legculib's session.
  stream.sync();

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
