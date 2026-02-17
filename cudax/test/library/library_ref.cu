//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/std/cstddef>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__utility/ensure_current_device.cuh>
#include <cuda/experimental/kernel.cuh>
#include <cuda/experimental/library.cuh>

#include <testing.cuh>

#include "library_cubin.h"

C2H_CCCLRT_TEST("Library reference", "[library_ref]")
{
  const cuda::device_ref device{0};
  const auto cc = device.attribute(cuda::device_attributes::compute_capability);

  // unified function requires at least sm90
  const auto with_unified_function = (cc >= cuda::compute_capability{90});
  const auto lib_src               = make_library_cubin(cc);

  CUlibrary lib1 = ::cuda::__driver::__libraryLoadData(lib_src.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  CUlibrary lib2 = ::cuda::__driver::__libraryLoadData(lib_src.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);

  // Types
  {
    STATIC_REQUIRE(cuda::std::is_same_v<typename cudax::library_ref::value_type, CUlibrary>);
  }

  // Default constructor
  {
    STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cudax::library_ref>);
  }

  // Constructor from library handle
  {
    STATIC_REQUIRE(cuda::std::is_constructible_v<cudax::library_ref, CUlibrary>);
    STATIC_REQUIRE(cuda::std::is_convertible_v<CUlibrary, cudax::library_ref>);

    cudax::library_ref lib_ref{lib1};
    CUDAX_REQUIRE(lib1 == lib_ref.get());
  }

  // Copy constructor
  {
    STATIC_REQUIRE(cuda::std::is_trivially_copy_constructible_v<cudax::library_ref>);

    cudax::library_ref lib_ref1{lib1};
    CUDAX_REQUIRE(lib1 == lib_ref1.get());

    cudax::library_ref lib_ref2{lib1};
    CUDAX_REQUIRE(lib1 == lib_ref2.get());
    CUDAX_REQUIRE(lib_ref1.get() == lib_ref2.get());
  }

  // Has kernel
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().has_kernel(kernel_name)), bool>);

    cudax::library_ref lib_ref{lib1};
    CUDAX_REQUIRE(lib_ref.has_kernel(kernel_name));
    CUDAX_REQUIRE(!lib_ref.has_kernel("non_existent_kernel"));
  }

  // Get kernel
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().kernel<void(int*, int)>(kernel_name)),
                           cudax::kernel_ref<void(int*, int)>>);

    cudax::library_ref lib_ref{lib1};
    auto kernel = lib_ref.kernel<void(int*, int)>(kernel_name);

    CUkernel kernel_handle;
    CUDAX_REQUIRE(::cuda::__driver::__libraryGetKernelNoThrow(kernel_handle, lib1, kernel_name) == cudaSuccess);
    CUDAX_REQUIRE(kernel.get() == kernel_handle);
  }

  // Has global symbol
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().has_global(global_symbol_name, device)),
                           bool>);

    cudax::library_ref lib_ref{lib1};
    CUDAX_REQUIRE(lib_ref.has_global(global_symbol_name, device));
    CUDAX_REQUIRE(lib_ref.has_global(const_symbol_name, device));
    CUDAX_REQUIRE(!lib_ref.has_global("non_existent_global", device));
  }

  // Get global symbol
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().global(global_symbol_name, device)),
                           cudax::library_symbol_info>);

    cudax::library_ref lib_ref{lib1};

    // Test global_symbol_name
    {
      auto global_sym = lib_ref.global(global_symbol_name, device);

      cuda::__ensure_current_context context_guard{device};

      CUdeviceptr global_symbol_ptr;
      cuda::std::size_t global_symbol_size;
      CUDAX_REQUIRE(
        ::cuda::__driver::__libraryGetGlobalNoThrow(global_symbol_ptr, global_symbol_size, lib1, global_symbol_name)
        == cudaSuccess);

      CUDAX_REQUIRE(reinterpret_cast<CUdeviceptr>(global_sym.ptr) == global_symbol_ptr);
      CUDAX_REQUIRE(global_sym.size == global_symbol_size);
      CUDAX_REQUIRE(global_sym.size == sizeof(int));
    }

    // Test const_symbol_name
    {
      auto const_sym = lib_ref.global(const_symbol_name, device);

      cuda::__ensure_current_context context_guard{device};

      CUdeviceptr const_symbol_ptr;
      cuda::std::size_t const_symbol_size;
      CUDAX_REQUIRE(
        ::cuda::__driver::__libraryGetGlobalNoThrow(const_symbol_ptr, const_symbol_size, lib1, const_symbol_name)
        == cudaSuccess);

      CUDAX_REQUIRE(reinterpret_cast<CUdeviceptr>(const_sym.ptr) == const_symbol_ptr);
      CUDAX_REQUIRE(const_sym.size == const_symbol_size);
      CUDAX_REQUIRE(const_sym.size == sizeof(int));
    }
  }

  // Has managed symbol
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().has_managed(managed_symbol_name)), bool>);

    cudax::library_ref lib_ref{lib1};
    CUDAX_REQUIRE(lib_ref.has_managed(managed_symbol_name));
    CUDAX_REQUIRE(!lib_ref.has_managed("non_existent_managed"));
  }

  // Get managed symbol
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().managed(managed_symbol_name)),
                                        cudax::library_symbol_info>);

    cudax::library_ref lib_ref{lib1};
    auto managed_sym = lib_ref.managed(managed_symbol_name);

    CUdeviceptr managed_symbol_ptr;
    cuda::std::size_t managed_symbol_size;
    CUDAX_REQUIRE(
      ::cuda::__driver::__libraryGetManagedNoThrow(managed_symbol_ptr, managed_symbol_size, lib1, managed_symbol_name)
      == cudaSuccess);

    CUDAX_REQUIRE(reinterpret_cast<CUdeviceptr>(managed_sym.ptr) == managed_symbol_ptr);
    CUDAX_REQUIRE(managed_sym.size == managed_symbol_size);
    CUDAX_REQUIRE(managed_sym.size == sizeof(int));
  }

  // Has unified function
  if (with_unified_function)
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().has_unified_function(unified_function_name)),
                           bool>);

    cudax::library_ref lib_ref{lib1};
    CUDAX_REQUIRE(lib_ref.has_unified_function(unified_function_name));
    CUDAX_REQUIRE(!lib_ref.has_unified_function("non_existent_unified"));
  }

  // Get unified function
  if (with_unified_function)
  {
    STATIC_REQUIRE(cuda::std::is_same_v<
                   decltype(cuda::std::declval<cudax::library_ref>().unified_function<int()>(unified_function_name)),
                   int (*)()>);

    cudax::library_ref lib_ref{lib1};
    auto unified_fn = lib_ref.unified_function<int()>(unified_function_name);

    void* unified_fn_addr;
    CUDAX_REQUIRE(
      ::cuda::__driver::__libraryGetUnifiedFunctionNoThrow(unified_fn_addr, lib1, managed_symbol_name) == cudaSuccess);

    CUDAX_REQUIRE(reinterpret_cast<void*>(unified_fn) == unified_fn_addr);
  }

  // Get handle
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library_ref>().get()), CUlibrary>);

    cudax::library_ref lib_ref{lib1};
    CUDAX_REQUIRE(lib1 == lib_ref.get());
  }

  // Equality/Inequality comparison
  {
    cudax::library_ref lib_ref1{lib1};
    cudax::library_ref lib_ref2{lib2};

    CUDAX_REQUIRE(lib_ref1 == lib_ref1);
    CUDAX_REQUIRE(lib_ref1 != lib_ref2);
  }

  CUDAX_REQUIRE(::cuda::__driver::__libraryUnloadNoThrow(lib1) == cudaSuccess);
  CUDAX_REQUIRE(::cuda::__driver::__libraryUnloadNoThrow(lib2) == cudaSuccess);
}
