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
//! \brief Simple test demonstrating ctx_resource management with generic context

#include <cuda/experimental/__stf/internal/context.cuh>

#include <atomic>
#include <memory>

using namespace cuda::experimental::stf;

// Simple test resource that tracks its lifecycle
class simple_test_resource : public ctx_resource
{
  static ::std::atomic<int> alive_count;

public:
  simple_test_resource()
  {
    alive_count.fetch_add(1);
  }

  ~simple_test_resource() override
  {
    alive_count.fetch_sub(1);
  }

  void release(cudaStream_t /*stream*/) override
  {
    // No special release action needed for this test
  }

  bool can_release_in_callback() const override
  {
    return true; // Can be released in a host callback
  }

  void release_in_callback() override
  {
    // Host-side cleanup - nothing to do for this simple test
  }

  static int get_alive_count()
  {
    return alive_count.load();
  }
};

::std::atomic<int> simple_test_resource::alive_count{0};

int main()
{
  // Test with generic context (defaults to stream_ctx)
  {
    context ctx; // Default initialization as stream_ctx

    EXPECT(simple_test_resource::get_alive_count() == 0);

    // Add a simple host launch with some work
    ctx.host_launch()->*[]() {
      // Trivial workload
    };

    // Add some resources to the context
    for (int i = 0; i < 5; ++i)
    {
      auto resource = ::std::make_shared<simple_test_resource>();
      ctx.add_resource(resource);
    }

    // Verify resources are alive
    EXPECT(simple_test_resource::get_alive_count() == 5);

    // Finalize the context - this should release all resources
    ctx.finalize();
  } // Context goes out of scope

  // All resources should have been cleaned up
  EXPECT(simple_test_resource::get_alive_count() == 0);

  // Test with graph context through generic interface
  {
    context ctx = graph_ctx(); // Explicitly use graph backend

    EXPECT(simple_test_resource::get_alive_count() == 0);

    ctx.host_launch()->*[]() {
      // Trivial workload
    };

    // Add resources
    for (int i = 0; i < 3; ++i)
    {
      ctx.add_resource(::std::make_shared<simple_test_resource>());
    }

    EXPECT(simple_test_resource::get_alive_count() == 3);

    ctx.finalize();
  }

  EXPECT(simple_test_resource::get_alive_count() == 0);

  return 0;
}
