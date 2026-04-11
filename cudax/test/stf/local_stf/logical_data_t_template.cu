//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//! \brief Test that ctx_t::logical_data_t works correctly when ctx_t is a template parameter
//!
//! This test ensures that the logical_data_t type alias defined in context classes
//! (stream_ctx, graph_ctx, stackable_ctx, etc.) works correctly in generic code where
//! the context type is a template parameter. This is essential for writing context-agnostic
//! library functions and algorithms.

#include <cuda/std/type_traits>

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Helper kernel for testing
template <typename T>
__global__ void scale_kernel(size_t n, T factor, T* data)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (size_t i = tid; i < n; i += nthreads)
  {
    data[i] *= factor;
  }
}

// Test 1: Compile-time type trait tests
// Verify that ctx_t::logical_data_t<T> is a valid type for different context types
void test_compile_time_logical_data_t_exists()
{
  // Test that logical_data_t is defined for stream_ctx
  using stream_logical_data_int    = stream_ctx::logical_data_t<slice<int>>;
  using stream_logical_data_double = stream_ctx::logical_data_t<slice<double>>;
  static_assert(::std::is_same_v<stream_logical_data_int, logical_data<slice<int>>>,
                "stream_ctx::logical_data_t<slice<int>> should be logical_data<slice<int>>");
  static_assert(::std::is_same_v<stream_logical_data_double, logical_data<slice<double>>>,
                "stream_ctx::logical_data_t<slice<double>> should be logical_data<slice<double>>");

  // Test that logical_data_t is defined for graph_ctx
  using graph_logical_data_int    = graph_ctx::logical_data_t<slice<int>>;
  using graph_logical_data_double = graph_ctx::logical_data_t<slice<double>>;
  static_assert(::std::is_same_v<graph_logical_data_int, logical_data<slice<int>>>,
                "graph_ctx::logical_data_t<slice<int>> should be logical_data<slice<int>>");
  static_assert(::std::is_same_v<graph_logical_data_double, logical_data<slice<double>>>,
                "graph_ctx::logical_data_t<slice<double>> should be logical_data<slice<double>>");

  // Test that logical_data_t is defined for stackable_ctx
  using stackable_logical_data_int    = stackable_ctx::logical_data_t<slice<int>>;
  using stackable_logical_data_double = stackable_ctx::logical_data_t<slice<double>>;
  static_assert(::std::is_same_v<stackable_logical_data_int, stackable_logical_data<slice<int>>>,
                "stackable_ctx::logical_data_t<slice<int>> should be stackable_logical_data<slice<int>>");
  static_assert(::std::is_same_v<stackable_logical_data_double, stackable_logical_data<slice<double>>>,
                "stackable_ctx::logical_data_t<slice<double>> should be stackable_logical_data<slice<double>>");
}

// Test 2: Generic function that uses ctx_t::logical_data_t as a template parameter
// This demonstrates the main use case: writing context-agnostic code
template <typename Ctx>
typename Ctx::template logical_data_t<slice<int>> create_and_initialize_data(Ctx& ctx, size_t n, int init_value)
{
  // Create logical data using the context-specific logical_data_t type
  auto data = ctx.logical_data(shape_of<slice<int>>(n));
  data.set_symbol("initialized_data");

  // Initialize the data
  ctx.parallel_for(data.shape(), data.write())->*[init_value] __device__(size_t i, auto d) {
    d(i) = init_value + static_cast<int>(i);
  };

  return data;
}

// Test 3: Generic function that takes and returns ctx_t::logical_data_t
// Note: Using auto&& for the data parameter to avoid template deduction issues with dependent types
template <typename Ctx, typename LogicalData, typename Factor>
void scale_data(Ctx& ctx, LogicalData&& data, Factor factor)
{
  ctx.task(data.rw())->*[factor](cudaStream_t s, auto d) {
    scale_kernel<<<16, 128, 0, s>>>(d.size(), factor, d.data_handle());
  };
}

// Test 4: Generic class that stores ctx_t::logical_data_t
template <typename Ctx>
class GenericDataHolder
{
public:
  using int_data_t    = typename Ctx::template logical_data_t<slice<int>>;
  using double_data_t = typename Ctx::template logical_data_t<slice<double>>;

  GenericDataHolder(Ctx& ctx, size_t n)
      : int_data_(ctx.logical_data(shape_of<slice<int>>(n)))
      , double_data_(ctx.logical_data(shape_of<slice<double>>(n)))
  {
    int_data_.set_symbol("holder_int_data");
    double_data_.set_symbol("holder_double_data");
  }

  int_data_t& get_int_data()
  {
    return int_data_;
  }

  double_data_t& get_double_data()
  {
    return double_data_;
  }

private:
  int_data_t int_data_;
  double_data_t double_data_;
};

// Test 5: Generic function using auto with ctx_t::logical_data_t
template <typename Ctx>
void test_auto_deduction(Ctx& ctx)
{
  // Create data using the generic function
  auto data = create_and_initialize_data(ctx, 100, 42);

  // Verify the type is correct
  using expected_type = typename Ctx::template logical_data_t<slice<int>>;
  static_assert(::std::is_same_v<decltype(data), expected_type>,
                "Auto-deduced type should match ctx_t::logical_data_t<slice<int>>");

  // Use the data
  ctx.host_launch(data.read())->*[](auto d) {
    for (size_t i = 0; i < d.size(); i++)
    {
      EXPECT(d(i) == 42 + static_cast<int>(i), "Expected ", 42 + i, " but got ", d(i), " at index ", i);
    }
  };
}

// Main test function template
template <typename Ctx>
void run_tests()
{
  Ctx ctx;

  // Test 1: Basic creation and usage with template logical_data_t
  {
    auto data = create_and_initialize_data(ctx, 256, 10);

    ctx.host_launch(data.read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(d(i) == 10 + static_cast<int>(i));
      }
    };
  }

  // Test 2: Using scale_data generic function
  {
    auto data = create_and_initialize_data(ctx, 128, 5);
    scale_data(ctx, data, 3);

    ctx.host_launch(data.read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(d(i) == (5 + static_cast<int>(i)) * 3);
      }
    };
  }

  // Test 3: Using GenericDataHolder class
  {
    GenericDataHolder<Ctx> holder(ctx, 64);

    // Initialize int data
    ctx.parallel_for(holder.get_int_data().shape(), holder.get_int_data().write())->*[] __device__(size_t i, auto d) {
      d(i) = static_cast<int>(i * 2);
    };

    // Initialize double data
    ctx.parallel_for(holder.get_double_data().shape(), holder.get_double_data().write())
        ->*[] __device__(size_t i, auto d) {
              d(i) = static_cast<double>(i) * 1.5;
            };

    // Verify int data
    ctx.host_launch(holder.get_int_data().read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(d(i) == static_cast<int>(i * 2));
      }
    };

    // Verify double data
    ctx.host_launch(holder.get_double_data().read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(fabs(d(i) - static_cast<double>(i) * 1.5) < 1e-9);
      }
    };
  }

  // Test 4: Auto deduction
  test_auto_deduction(ctx);

  // Test 5: Multiple operations with logical_data_t
  {
    typename Ctx::template logical_data_t<slice<int>> data1 = ctx.logical_data(shape_of<slice<int>>(32));
    typename Ctx::template logical_data_t<slice<int>> data2 = ctx.logical_data(shape_of<slice<int>>(32));

    data1.set_symbol("data1");
    data2.set_symbol("data2");

    // Initialize both
    ctx.parallel_for(data1.shape(), data1.write())->*[] __device__(size_t i, auto d) {
      d(i) = static_cast<int>(i);
    };

    ctx.parallel_for(data2.shape(), data2.write())->*[] __device__(size_t i, auto d) {
      d(i) = static_cast<int>(i * 10);
    };

    // Combine them
    ctx.parallel_for(data1.shape(), data1.rw(), data2.read())->*[] __device__(size_t i, auto d1, auto d2) {
      d1(i) += d2(i);
    };

    // Verify
    ctx.host_launch(data1.read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(d(i) == static_cast<int>(i + i * 10));
      }
    };
  }

  ctx.finalize();
}

// Additional test for stackable_ctx specific features
void run_stackable_tests()
{
  stackable_ctx ctx;

  // Test nested contexts with logical_data_t
  {
    auto data = create_and_initialize_data(ctx, 128, 20);

    // Enter nested context
    {
      stackable_ctx::graph_scope_guard scope{ctx};

      // Scale in nested context
      scale_data(ctx, data, 2);

      // Create local data in nested context
      typename stackable_ctx::logical_data_t<slice<int>> local_data = ctx.logical_data(shape_of<slice<int>>(64));
      local_data.set_symbol("nested_local");

      ctx.parallel_for(local_data.shape(), local_data.write())->*[] __device__(size_t i, auto d) {
        d(i) = static_cast<int>(i * 5);
      };

      ctx.host_launch(local_data.read())->*[](auto d) {
        for (size_t i = 0; i < d.size(); i++)
        {
          EXPECT(d(i) == static_cast<int>(i * 5));
        }
      };
    }

    // Verify data after nested context
    ctx.host_launch(data.read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(d(i) == (20 + static_cast<int>(i)) * 2);
      }
    };
  }

  // Test with GenericDataHolder in stackable context
  {
    GenericDataHolder<stackable_ctx> holder(ctx, 48);

    ctx.parallel_for(holder.get_int_data().shape(), holder.get_int_data().write())->*[] __device__(size_t i, auto d) {
      d(i) = static_cast<int>(i + 100);
    };

    ctx.host_launch(holder.get_int_data().read())->*[](auto d) {
      for (size_t i = 0; i < d.size(); i++)
      {
        EXPECT(d(i) == static_cast<int>(i + 100));
      }
    };
  }

  ctx.finalize();
}

int main()
{
  // Run compile-time tests
  test_compile_time_logical_data_t_exists();

  // Run runtime tests with different context types
  run_tests<stream_ctx>();
  run_tests<graph_ctx>();
  run_tests<stackable_ctx>();

  // Run stackable-specific tests
  run_stackable_tests();

  return 0;
}
