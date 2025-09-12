//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Test ctx_resource management with different context types

#include <cuda/experimental/__stf/internal/context.cuh>

#include <atomic>
#include <memory>

using namespace cuda::experimental::stf;

namespace
{

// Counters for tracking resource lifecycle
std::atomic<int> stream_resource_construct_count{0};
std::atomic<int> stream_resource_release_count{0};
std::atomic<int> callback_resource_construct_count{0};
std::atomic<int> callback_resource_release_count{0};

// Test resource that requires a stream for release
class test_stream_resource : public ctx_resource
{
public:
  test_stream_resource()
  {
    stream_resource_construct_count.fetch_add(1);
  }

  ~test_stream_resource() override = default;

  void release(cudaStream_t stream) override
  {
    // Simulate async resource release that needs a stream
    cudaEvent_t event;
    cuda_safe_call(cudaEventCreate(&event));
    cuda_safe_call(cudaEventRecord(event, stream));
    cuda_safe_call(cudaEventSynchronize(event)); // Wait for completion
    cuda_safe_call(cudaEventDestroy(event));

    stream_resource_release_count.fetch_add(1);
  }

  bool can_release_in_callback() const override
  {
    return false; // This resource needs a stream
  }
};

// Test resource that can be released in a host callback
class test_callback_resource : public ctx_resource
{
public:
  test_callback_resource()
  {
    callback_resource_construct_count.fetch_add(1);
  }

  ~test_callback_resource() override = default;

  void release(cudaStream_t /*stream*/) override
  {
    // Should not be called for callback resources
    assert(false && "release() should not be called for callback resources");
  }

  bool can_release_in_callback() const override
  {
    return true; // This resource can be released in a callback
  }

  void release_in_callback() override
  {
    // Simulate host-side resource cleanup
    callback_resource_release_count.fetch_add(1);
  }
};

void reset_counters()
{
  stream_resource_construct_count.store(0);
  stream_resource_release_count.store(0);
  callback_resource_construct_count.store(0);
  callback_resource_release_count.store(0);
}

void check_all_resources_released()
{
  // Verify all constructed resources were properly released
  EXPECT(stream_resource_construct_count.load() == stream_resource_release_count.load());
  EXPECT(callback_resource_construct_count.load() == callback_resource_release_count.load());
}

template <typename CtxType>
void test_context_resources()
{
  reset_counters();

  CtxType ctx;

  // Add a simple host launch to ensure context has some work
  ctx.host_launch()->*[]() {
    // Trivial workload - just increment a counter
    static std::atomic<int> work_counter{0};
    work_counter.fetch_add(1);
  };

  // Add various types of resources
  const int num_stream_resources   = 3;
  const int num_callback_resources = 2;

  // Add stream-dependent resources
  for (int i = 0; i < num_stream_resources; ++i)
  {
    auto resource = ::std::make_shared<test_stream_resource>();
    ctx.add_resource(resource);
  }

  // Add callback resources
  for (int i = 0; i < num_callback_resources; ++i)
  {
    auto resource = ::std::make_shared<test_callback_resource>();
    ctx.add_resource(resource);
  }

  // Verify resources were constructed
  EXPECT(stream_resource_construct_count.load() == num_stream_resources);
  EXPECT(callback_resource_construct_count.load() == num_callback_resources);
  EXPECT(stream_resource_release_count.load() == 0); // Not released yet
  EXPECT(callback_resource_release_count.load() == 0); // Not released yet

  // Finalize the context - this should release resources automatically
  ctx.finalize();

  // Verify all resources were released
  EXPECT(stream_resource_release_count.load() == num_stream_resources);
  EXPECT(callback_resource_release_count.load() == num_callback_resources);

  check_all_resources_released();
}

void test_graph_ctx_manual_resource_release()
{
  reset_counters();

  graph_ctx ctx;

  // Add a simple host launch with work counter
  std::atomic<int> work_counter{0};
  ctx.host_launch()->*[&work_counter]() {
    work_counter.fetch_add(1);
  };

  // Add resources
  const int num_resources = 2;
  for (int i = 0; i < num_resources; ++i)
  {
    ctx.add_resource(std::make_shared<test_stream_resource>());
    ctx.add_resource(std::make_shared<test_callback_resource>());
  }

  EXPECT(stream_resource_construct_count.load() == num_resources);
  EXPECT(callback_resource_construct_count.load() == num_resources);

  // Resources should not be released yet
  EXPECT(stream_resource_release_count.load() == 0);
  EXPECT(callback_resource_release_count.load() == 0);

  // Generate the graph using finalize_as_graph
  ::std::shared_ptr<cudaGraph_t> graph = ctx.finalize_as_graph();

  // Create stream and instantiate graph for multiple launches
  cudaStream_t test_stream;
  cuda_safe_call(cudaStreamCreate(&test_stream));

  cudaGraphExec_t graphExec;
  cuda_safe_call(cudaGraphInstantiate(&graphExec, *graph, nullptr, nullptr, 0));

  // Launch the graph multiple times
  const int num_launches = 3;
  for (int i = 0; i < num_launches; i++)
  {
    cuda_safe_call(cudaGraphLaunch(graphExec, test_stream));
  }

  // Manually release resources after graph executions
  ctx.release_resources(test_stream);
  cuda_safe_call(cudaStreamSynchronize(test_stream));

  // Verify the work was executed
  EXPECT(work_counter.load() == num_launches);

  // Now resources should be released
  EXPECT(stream_resource_release_count.load() == num_resources);
  EXPECT(callback_resource_release_count.load() == num_resources);

  // Clean up
  cuda_safe_call(cudaGraphExecDestroy(graphExec));
  cuda_safe_call(cudaStreamDestroy(test_stream));

  check_all_resources_released();
}

} // anonymous namespace

int main()
{
  // Test with different context types
  test_context_resources<context>();
  test_context_resources<stream_ctx>();
  test_context_resources<graph_ctx>();

  // Test manual resource release (graph_ctx only for the sake of simplicity)
  test_graph_ctx_manual_resource_release();
}
