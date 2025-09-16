//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Test copy and move semantics for task dependency types
 *
 * This test verifies that task_dep_untyped, task_dep<T>, and stackable_task_dep<T>
 * are properly copyable and movable, which is essential for the STF framework to work
 * correctly with deferred operations and stackable contexts.
 */

#include <cuda/std/type_traits>

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Simple kernel for testing - defined outside to avoid device lambda nesting issues
__global__ void double_values_kernel(int* out, const int* in, int n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    out[idx] = in[idx] * 2;
  }
}

// Helper template to test copy/move operations on any type
template <typename T>
void test_copy_move_semantics(const T& original)
{
  // Test copy constructor
  {
    T copy_constructed(original);
    (void) copy_constructed; // Suppress unused variable warning
  }

  // Test move constructor
  {
    T temp(original);
    T move_constructed(::std::move(temp));
    (void) move_constructed; // Suppress unused variable warning
  }

  // Test copy assignment
  {
    T temp(original); // Create a temporary to assign to
    T copy_assigned(original); // Initialize with copy constructor first
    copy_assigned = temp; // Then test copy assignment
    (void) copy_assigned; // Suppress unused variable warning
  }

  // Test move assignment
  {
    T temp(original);
    T move_assigned(original); // Initialize with copy constructor first
    move_assigned = ::std::move(temp); // Then test move assignment
    (void) move_assigned; // Suppress unused variable warning
  }
}

// Compile-time type trait tests
void test_compile_time_type_traits()
{
  // Test task_dep_untyped
  static_assert(::std::is_copy_constructible_v<task_dep_untyped>, "task_dep_untyped must be copy constructible");
  static_assert(::std::is_copy_assignable_v<task_dep_untyped>, "task_dep_untyped must be copy assignable");
  static_assert(::std::is_move_constructible_v<task_dep_untyped>, "task_dep_untyped must be move constructible");
  static_assert(::std::is_move_assignable_v<task_dep_untyped>, "task_dep_untyped must be move assignable");

  // Test task_dep<T> for common types
  using int_task_dep = task_dep<slice<int>>;
  static_assert(::std::is_copy_constructible_v<int_task_dep>, "task_dep<slice<int>> must be copy constructible");
  static_assert(::std::is_copy_assignable_v<int_task_dep>, "task_dep<slice<int>> must be copy assignable");
  static_assert(::std::is_move_constructible_v<int_task_dep>, "task_dep<slice<int>> must be move constructible");
  static_assert(::std::is_move_assignable_v<int_task_dep>, "task_dep<slice<int>> must be move assignable");

  using double_task_dep = task_dep<slice<double>>;
  static_assert(::std::is_copy_constructible_v<double_task_dep>, "task_dep<slice<double>> must be copy constructible");
  static_assert(::std::is_copy_assignable_v<double_task_dep>, "task_dep<slice<double>> must be copy assignable");
  static_assert(::std::is_move_constructible_v<double_task_dep>, "task_dep<slice<double>> must be move constructible");
  static_assert(::std::is_move_assignable_v<double_task_dep>, "task_dep<slice<double>> must be move assignable");

  // Test stackable_task_dep<T> with proper template parameters
  using int_stackable_task_dep = stackable_task_dep<slice<int>, ::std::monostate, false>;
  static_assert(::std::is_copy_constructible_v<int_stackable_task_dep>,
                "stackable_task_dep<slice<int>> must be copy constructible");
  static_assert(::std::is_copy_assignable_v<int_stackable_task_dep>,
                "stackable_task_dep<slice<int>> must be copy assignable");
  static_assert(::std::is_move_constructible_v<int_stackable_task_dep>,
                "stackable_task_dep<slice<int>> must be move constructible");
  static_assert(::std::is_move_assignable_v<int_stackable_task_dep>,
                "stackable_task_dep<slice<int>> must be move assignable");

  using double_stackable_task_dep = stackable_task_dep<slice<double>, ::std::monostate, false>;
  static_assert(::std::is_copy_constructible_v<double_stackable_task_dep>,
                "stackable_task_dep<slice<double>> must be copy constructible");
  static_assert(::std::is_copy_assignable_v<double_stackable_task_dep>,
                "stackable_task_dep<slice<double>> must be copy assignable");
  static_assert(::std::is_move_constructible_v<double_stackable_task_dep>,
                "stackable_task_dep<slice<double>> must be move constructible");
  static_assert(::std::is_move_assignable_v<double_stackable_task_dep>,
                "stackable_task_dep<slice<double>> must be move assignable");

  // Test stackable_logical_data<T>
  using int_stackable_logical_data = stackable_logical_data<slice<int>>;
  static_assert(::std::is_move_constructible_v<int_stackable_logical_data>,
                "stackable_logical_data<slice<int>> must be move constructible");
  static_assert(::std::is_move_assignable_v<int_stackable_logical_data>,
                "stackable_logical_data<slice<int>> must be move assignable");
}

int main()
{
  // Run compile-time tests
  test_compile_time_type_traits();

  // Create stackable context for runtime tests
  stackable_ctx ctx;

  const size_t N = 64;

  // Create test data
  auto ldata_int    = ctx.logical_data(shape_of<slice<int>>(N));
  auto ldata_double = ctx.logical_data(shape_of<slice<double>>(N));

  // Test runtime copy/move semantics for stackable_task_dep
  {
    auto stackable_read_dep  = ldata_int.read();
    auto stackable_write_dep = ldata_int.write();
    auto stackable_rw_dep    = ldata_int.rw();

    // Test copy/move operations on stackable task dependencies
    test_copy_move_semantics(stackable_read_dep);
    test_copy_move_semantics(stackable_write_dep);
    test_copy_move_semantics(stackable_rw_dep);

    // Test with different data types
    auto stackable_double_dep = ldata_double.read();
    test_copy_move_semantics(stackable_double_dep);
  }

  // Test that stackable logical data itself is movable
  {
    auto ldata_copy  = ldata_int; // Copy
    auto ldata_moved = mv(ldata_int); // Move using STF's mv() utility

    // Verify we can still use the moved-to object
    auto dep_from_moved = ldata_moved.read();
    test_copy_move_semantics(dep_from_moved);
  }

  // Test that we can store task dependencies in containers (requires copy/move)
  {
    auto ldata_container_test = ctx.logical_data(shape_of<slice<float>>(32));

    ::std::vector<decltype(ldata_container_test.read())> read_deps;
    read_deps.push_back(ldata_container_test.read());
    read_deps.push_back(ldata_container_test.read());

    ::std::vector<decltype(ldata_container_test.write())> write_deps;
    write_deps.emplace_back(ldata_container_test.write());

    // Verify container operations work (copy, move, etc.)
    auto read_deps_copy   = read_deps; // Copy container
    auto write_deps_moved = ::std::move(write_deps); // Move container

    (void) read_deps_copy; // Suppress unused variable warnings
    (void) write_deps_moved;
  }

  // Test functional usage with actual tasks to ensure copy/move semantics work in practice
  {
    auto ltest = ctx.logical_data(shape_of<slice<int>>(16));

    // Initialize data
    ctx.parallel_for(ltest.shape(), ltest.write())->*[] __device__(size_t i, auto data) {
      data(i) = static_cast<int>(i);
    };

    ctx.push(); // Enter nested context

    // Create dependencies that will be copied/moved through deferred operations
    auto read_dep  = ltest.read();
    auto write_dep = ltest.write();

    // Use add_deps which relies on copy/move semantics
    auto task = ctx.task();
    task.add_deps(read_dep); // This should copy/move the dependency
    task.add_deps(write_dep); // This should copy/move the dependency

    task->*[&task](cudaStream_t stream) {
      // Get dependencies by index - this is how add_deps works
      auto input  = task.template get<slice<int>>(0); // First dependency (read)
      auto output = task.template get<slice<int>>(1); // Second dependency (write)

      // Simple kernel that doubles the values
      int N = input.size();

      int block_size = 256;
      int grid_size  = (N + block_size - 1) / block_size;

      // Use the global kernel function to avoid device lambda nesting
      double_values_kernel<<<grid_size, block_size, 0, stream>>>(output.data_handle(), input.data_handle(), N);
    };

    ctx.pop(); // Exit nested context

    // Verify result
    ctx.host_launch(ltest.read())->*[](auto data) {
      for (size_t i = 0; i < data.size(); i++)
      {
        EXPECT(data(i) == static_cast<int>(i * 2), "Expected ", i * 2, " but got ", data(i), " at index ", i);
      }
    };
  }

  ctx.finalize();

  return 0;
}
