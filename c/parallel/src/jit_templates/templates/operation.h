//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include <cuda/std/cstddef>
#  include <cuda/std/type_traits>
#  include <cuda/std/utility>

#  include <cccl/c/types.h>
#endif

#include "../mappings/operation.h"
#include "../mappings/type_info.h"

// Helper to define the C function pointer type: void (*)(void*..., void*)
template <typename Seq>
struct MakeVoidPtrFuncPtr;
template <cuda::std::size_t... Is>
struct MakeVoidPtrFuncPtr<cuda::std::index_sequence<Is...>>
{
  // Helper type that is always void*
  template <cuda::std::size_t>
  using void_ptr_t = const void*;
  // Define the function pointer type
  using type = void (*)(void_ptr_t<Is>..., void*); // Last void* is for result
};

// Helper to define the C function pointer type for stateful ops: void (*)(void* state, void*..., void*)
template <typename Seq>
struct MakeVoidPtrStatefulFuncPtr;
template <cuda::std::size_t... Is>
struct MakeVoidPtrStatefulFuncPtr<cuda::std::index_sequence<Is...>>
{
  template <cuda::std::size_t>
  using void_ptr_t = const void*;
  using type       = void (*)(void*, void_ptr_t<Is>..., void*); // First void* is state, last is result
};

template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
struct stateless_user_operation
{
  // Note: The user provided C f  unction (Operation.operation) must match the signature:
  // void (void* arg1, ..., void* argN, void* result_ptr)
  __device__ typename decltype(RetT)::Type operator()(typename decltype(ArgTs)::Type... args) const
  {
    using Indices        = cuda::std::make_index_sequence<sizeof...(ArgTs)>;
    using TargetCFuncPtr = typename MakeVoidPtrFuncPtr<Indices>::type;

    // Cast the stored operation pointer (assumed to be void* or compatible)
    auto c_func_ptr = reinterpret_cast<TargetCFuncPtr>(Operation.operation);

    // Prepare storage for the result
    typename decltype(RetT)::Type result;
    // Get a void pointer to the result storage
    void* result_ptr = static_cast<void*>(&result);

    // Call the C function, casting argument addresses to void*
    c_func_ptr((static_cast<void*>(const_cast<cuda::std::remove_reference_t<decltype(args)>*>(&args)))..., result_ptr);

    return result;
  }
};

template <cuda::std::size_t Size, cuda::std::size_t Alignment>
struct alignas(Alignment) user_operation_state
{
  char data[Size];
};

template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
struct stateful_user_operation
{
  user_operation_state<Operation.size, Operation.alignment> state;
  __device__ typename decltype(RetT)::Type operator()(typename decltype(ArgTs)::Type... args)
  {
    // Note: The user provided C function (Operation.operation) must match the signature:
    // void (void* state, void* arg1, ..., void* argN, void* result_ptr)
    using Indices        = cuda::std::make_index_sequence<sizeof...(ArgTs)>;
    using TargetCFuncPtr = typename MakeVoidPtrStatefulFuncPtr<Indices>::type;

    // Cast the stored operation pointer (assumed to be void* or compatible)
    auto c_func_ptr = reinterpret_cast<TargetCFuncPtr>(Operation.operation);

    // Prepare storage for the result
    typename decltype(RetT)::Type result;
    void* result_ptr = static_cast<void*>(&result);

    // Call the C function, passing state address, casting argument addresses to void*, and result pointer
    c_func_ptr(
      &state, (static_cast<void*>(const_cast<cuda::std::remove_reference_t<decltype(args)>*>(&args)))..., result_ptr);

    return result;
  }
};

struct user_operation_traits
{
  static const constexpr auto name = "user_operation_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping RetT, cccl_type_info_mapping... ArgTs>
  using type = cuda::std::conditional_t<Operation.is_stateless,
                                        stateless_user_operation<Tag, Operation, RetT, ArgTs...>,
                                        stateful_user_operation<Tag, Operation, RetT, ArgTs...>>;
};

struct binary_user_operation_traits
{
  static const constexpr auto name = "binary_user_operation_traits::type";
  template <typename Tag, cccl_op_t_mapping Operation, cccl_type_info_mapping ValueT>
  using type = user_operation_traits::type<Tag, Operation, ValueT, ValueT, ValueT>;
};
