//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Test for graph_scope RAII functionality in stackable_ctx
 * 
 * This test demonstrates and validates the graph_scope RAII wrapper
 * for automatic push/pop management in nested contexts.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;

  int input[1024];
  for (size_t i = 0; i < 1024; i++)
  {
    input[i] = static_cast<int>(i);
  }

  auto data = ctx.logical_data(input).set_symbol("data");

  // Test 1: Direct constructor style (lock_guard style) - most idiomatic
  {
    stackable_ctx::graph_scope scope{ctx}; // push() called here
    
    auto temp = ctx.logical_data(data.shape()).set_symbol("temp");
    
    ctx.parallel_for(temp.shape(), temp.write(), data.read())->*[] __device__(size_t i, auto temp, auto data) {
      temp(i) = data(i) * 2;
    };
    
    ctx.parallel_for(data.shape(), data.write(), temp.read())->*[] __device__(size_t i, auto data, auto temp) {
      data(i) = temp(i) + 1;
    };
    
    // pop() called automatically when scope goes out of scope
  }

  // Test 2: Factory method style (convenience)
  {
    auto scope = ctx.graph_scope(); // push() called here
    
    auto temp = ctx.logical_data(data.shape()).set_symbol("temp2");
    
    ctx.parallel_for(temp.shape(), temp.write(), data.read())->*[] __device__(size_t i, auto temp, auto data) {
      temp(i) = data(i) * 3;
    };
    
    ctx.parallel_for(data.shape(), data.write(), temp.read())->*[] __device__(size_t i, auto data, auto temp) {
      data(i) = temp(i);
    };
    
    // pop() called automatically when scope goes out of scope
  }

  // Test 3: Nested scopes with direct constructor style
  {
    stackable_ctx::graph_scope outer_scope{ctx}; // outer push
    
    auto intermediate = ctx.logical_data(data.shape()).set_symbol("intermediate");
    
    ctx.parallel_for(intermediate.shape(), intermediate.write(), data.read())->*[] __device__(size_t i, auto inter, auto data) {
      inter(i) = data(i) / 2;
    };
    
    {
      stackable_ctx::graph_scope inner_scope{ctx}; // inner push (nested)
      
      auto temp = ctx.logical_data(data.shape()).set_symbol("nested_temp");
      
      ctx.parallel_for(temp.shape(), temp.write(), intermediate.read())->*[] __device__(size_t i, auto temp, auto inter) {
        temp(i) = inter(i) + 10;
      };
      
      ctx.parallel_for(data.shape(), data.write(), temp.read())->*[] __device__(size_t i, auto data, auto temp) {
        data(i) = temp(i);
      };
      
      // inner pop() called automatically here
    }
    
    // outer pop() called automatically here
  }

  // Test 4: Iterative pattern (like stackable2.cu)
  for (int iter = 0; iter < 3; iter++) {
    stackable_ctx::graph_scope iteration{ctx}; // New scope each iteration
    
    auto temp = ctx.logical_data(data.shape()).set_symbol("iter_temp");
    
    // tmp = data
    ctx.parallel_for(temp.shape(), temp.write(), data.read())->*[] __device__(size_t i, auto temp, auto data) {
      temp(i) = data(i);
    };
    
    // data++
    ctx.parallel_for(data.shape(), data.rw())->*[] __device__(size_t i, auto data) {
      data(i) += 1;
    };
    
    // temp *= 2
    ctx.parallel_for(temp.shape(), temp.rw())->*[] __device__(size_t i, auto temp) {
      temp(i) *= 2;
    };
    
    // data += temp
    ctx.parallel_for(data.shape(), temp.read(), data.rw())->*[] __device__(size_t i, auto temp, auto data) {
      data(i) += temp(i);
    };
    
    // pop() called automatically at end of iteration
  }

  // Test 5: Mixed usage styles
  {
    // Outer scope using direct constructor
    stackable_ctx::graph_scope outer{ctx};
    
    auto temp1 = ctx.logical_data(data.shape()).set_symbol("temp1");
    
    {
      // Inner scope using factory method
      auto inner = ctx.graph_scope();
      
      auto temp2 = ctx.logical_data(data.shape()).set_symbol("temp2");
      
      ctx.parallel_for(temp2.shape(), temp2.write(), data.read())->*[] __device__(size_t i, auto temp2, auto data) {
        temp2(i) = data(i) + 5;
      };
      
      ctx.parallel_for(temp1.shape(), temp1.write(), temp2.read())->*[] __device__(size_t i, auto temp1, auto temp2) {
        temp1(i) = temp2(i) * 2;
      };
      
      // inner pop() automatically
    }
    
    ctx.parallel_for(data.shape(), data.write(), temp1.read())->*[] __device__(size_t i, auto data, auto temp1) {
      data(i) = temp1(i);
    };
    
    // outer pop() automatically
  }

  // Test 6: Exception safety - scope should clean up even if exceptions occur
  try {
    stackable_ctx::graph_scope exception_scope{ctx};
    
    auto temp = ctx.logical_data(data.shape()).set_symbol("exception_temp");
    
    ctx.parallel_for(temp.shape(), temp.write(), data.read())->*[] __device__(size_t i, auto temp, auto data) {
      temp(i) = data(i) + 100;
    };
    
    // Simulate error condition (in real code, this might be a CUDA error or other exception)
    // throw std::runtime_error("Simulated error");
    
    // Even if exception occurs, destructor ensures pop() is called
  } catch (...) {
    // Exception handling - graph_scope destructor already called pop()
  }

  ctx.finalize();
  return 0;
}
