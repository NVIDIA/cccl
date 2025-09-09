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
 * @brief Demonstration of graph_scope RAII usage styles
 * 
 * This example shows different ways to use stackable_ctx::graph_scope
 * for automatic push/pop management in nested contexts.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stackable_ctx ctx;
  
  int data[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto lA = ctx.logical_data(data);
  
  // Style 1: Direct constructor (like std::lock_guard)
  // This is the most idiomatic C++ style
  {
    stackable_ctx::graph_scope scope{ctx}; // Direct constructor - push() called
    
    auto temp = ctx.logical_data(lA.shape());
    ctx.parallel_for(temp.shape(), temp.write(), lA.read())->*[] __device__(size_t i, auto temp, auto a) {
      temp(i) = a(i) * 2;
    };
    
    ctx.parallel_for(lA.shape(), lA.write(), temp.read())->*[] __device__(size_t i, auto a, auto temp) {
      a(i) = temp(i);
    };
    
    // pop() called automatically when scope goes out of scope
  }
  
  // Style 2: Factory method (convenience)
  // Useful when you prefer auto type deduction
  {
    auto scope = ctx.graph_scope(); // Factory method - push() called
    
    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 1;
    };
    
    // pop() called automatically
  }
  
  // Style 3: Direct constructor with explicit type alias
  // Useful for readability in complex scenarios
  {
    using scope_t = stackable_ctx::graph_scope;
    scope_t scope{ctx}; // Explicit type - push() called
    
    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) *= 3;
    };
    
    // pop() called automatically
  }
  
  // Style 4: Iterative pattern (like in stackable2.cu)
  // Demonstrates repeated nested contexts
  for (int iter = 0; iter < 3; iter++) {
    stackable_ctx::graph_scope iteration{ctx}; // New scope each iteration
    
    auto temp = ctx.logical_data(lA.shape());
    
    // tmp = a
    ctx.parallel_for(temp.shape(), temp.write(), lA.read())->*[] __device__(size_t i, auto temp, auto a) {
      temp(i) = a(i);
    };
    
    // a++
    ctx.parallel_for(lA.shape(), lA.rw())->*[] __device__(size_t i, auto a) {
      a(i) += 1;
    };
    
    // tmp *= 2
    ctx.parallel_for(temp.shape(), temp.rw())->*[] __device__(size_t i, auto temp) {
      temp(i) *= 2;
    };
    
    // a += tmp
    ctx.parallel_for(lA.shape(), temp.read(), lA.rw())->*[] __device__(size_t i, auto temp, auto a) {
      a(i) += temp(i);
    };
    
    // pop() called automatically at end of iteration
  }
  
  ctx.finalize();
  return 0;
}
