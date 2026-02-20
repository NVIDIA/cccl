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

C2H_CCCLRT_TEST("Library", "[library]")
{
  const cuda::device_ref device{0};
  const auto cc = device.attribute(cuda::device_attributes::compute_capability);

  // unified function requires at least sm90
  const auto with_unified_function = (cc >= cuda::compute_capability{90});
  const auto lib_src               = make_library_cubin(cc);

  CUlibrary lib1_native =
    ::cuda::__driver::__libraryLoadData(lib_src.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);
  CUlibrary lib2_native =
    ::cuda::__driver::__libraryLoadData(lib_src.c_str(), nullptr, nullptr, 0, nullptr, nullptr, 0);

  // Types
  {
    STATIC_REQUIRE(cuda::std::is_same_v<typename cudax::library::value_type, CUlibrary>);
  }

  // Construction from native handle
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cudax::library::from_native_handle(CUlibrary{})), cudax::library>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib1_native == lib.get());

    (void) lib.release(); // prevent library unload in destructor
  }

  // Default constructor
  {
    STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cudax::library>);
  }

  // Constructor into moved-from state
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_constructible_v<cudax::library, cudax::no_init_t>);
    cudax::library lib{cudax::no_init};
    CUDAX_REQUIRE(lib.get() == CUlibrary{});

    // lib is in a moved-from state
  }

  // Copy constructor
  {
    STATIC_REQUIRE(!cuda::std::is_copy_constructible_v<cudax::library>);
  }

  // Move constructor
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_move_constructible_v<cudax::library>);

    cudax::library lib1 = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib1.get() == lib1_native);

    cudax::library lib2{cuda::std::move(lib1)};
    CUDAX_REQUIRE(lib1.get() == CUlibrary{});
    CUDAX_REQUIRE(lib2.get() == lib1_native);

    // lib1 is in a moved-from state
    (void) lib2.release(); // prevent library unload in destructor
  }

  // Copy assignment
  {
    STATIC_REQUIRE(!cuda::std::is_copy_assignable_v<cudax::library>);
  }

  // Move assignment
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_move_assignable_v<cudax::library>);

    cudax::library lib1 = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib1.get() == lib1_native);

    cudax::library lib2{cudax::no_init};
    CUDAX_REQUIRE(lib2.get() == CUlibrary{});

    lib2 = cuda::std::move(lib1);
    CUDAX_REQUIRE(lib1.get() == CUlibrary{});
    CUDAX_REQUIRE(lib2.get() == lib1_native);

    // lib1 is in a moved-from state
    (void) lib2.release(); // prevent library unload in destructor
  }

  // Has kernel
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().has_kernel(kernel_name)), bool>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib.has_kernel(kernel_name));
    CUDAX_REQUIRE(!lib.has_kernel("non_existent_kernel"));

    (void) lib.release(); // prevent library unload in destructor
  }

  // Get kernel
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().kernel<void(int*, int)>(kernel_name)),
                           cudax::kernel_ref<void(int*, int)>>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    auto kernel        = lib.kernel<void(int*, int)>(kernel_name);

    CUkernel kernel_handle;
    CUDAX_REQUIRE(::cuda::__driver::__libraryGetKernelNoThrow(kernel_handle, lib1_native, kernel_name) == cudaSuccess);
    CUDAX_REQUIRE(kernel.get() == kernel_handle);

    (void) lib.release(); // prevent library unload in destructor
  }

  // Has global symbol
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().has_global(global_symbol_name, device)), bool>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib.has_global(global_symbol_name, device));
    CUDAX_REQUIRE(lib.has_global(const_symbol_name, device));
    CUDAX_REQUIRE(!lib.has_global("non_existent_global", device));

    (void) lib.release(); // prevent library unload in destructor
  }

  // Get global symbol
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().global(global_symbol_name, device)),
                           cudax::library_symbol_info>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);

    // Test global_symbol_name
    {
      auto global_sym = lib.global(global_symbol_name, device);

      cuda::__ensure_current_context context_guard{device};

      CUdeviceptr global_symbol_ptr;
      cuda::std::size_t global_symbol_size;
      CUDAX_REQUIRE(::cuda::__driver::__libraryGetGlobalNoThrow(
                      global_symbol_ptr, global_symbol_size, lib1_native, global_symbol_name)
                    == cudaSuccess);

      CUDAX_REQUIRE(reinterpret_cast<CUdeviceptr>(global_sym.ptr) == global_symbol_ptr);
      CUDAX_REQUIRE(global_sym.size == global_symbol_size);
      CUDAX_REQUIRE(global_sym.size == sizeof(int));
    }

    // Test const_symbol_name
    {
      auto const_sym = lib.global(const_symbol_name, device);

      cuda::__ensure_current_context context_guard{device};

      CUdeviceptr const_symbol_ptr;
      cuda::std::size_t const_symbol_size;
      CUDAX_REQUIRE(
        ::cuda::__driver::__libraryGetGlobalNoThrow(const_symbol_ptr, const_symbol_size, lib1_native, const_symbol_name)
        == cudaSuccess);

      CUDAX_REQUIRE(reinterpret_cast<CUdeviceptr>(const_sym.ptr) == const_symbol_ptr);
      CUDAX_REQUIRE(const_sym.size == const_symbol_size);
      CUDAX_REQUIRE(const_sym.size == sizeof(int));
    }

    (void) lib.release(); // prevent library unload in destructor
  }

  // Has managed symbol
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().has_managed(managed_symbol_name)), bool>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib.has_managed(managed_symbol_name));
    CUDAX_REQUIRE(!lib.has_managed("non_existent_managed"));

    (void) lib.release(); // prevent library unload in destructor
  }

  // Get managed symbol
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().managed(managed_symbol_name)),
                                        cudax::library_symbol_info>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    auto managed_sym   = lib.managed(managed_symbol_name);

    CUdeviceptr managed_symbol_ptr;
    cuda::std::size_t managed_symbol_size;
    CUDAX_REQUIRE(::cuda::__driver::__libraryGetManagedNoThrow(
                    managed_symbol_ptr, managed_symbol_size, lib1_native, managed_symbol_name)
                  == cudaSuccess);

    CUDAX_REQUIRE(reinterpret_cast<CUdeviceptr>(managed_sym.ptr) == managed_symbol_ptr);
    CUDAX_REQUIRE(managed_sym.size == managed_symbol_size);
    CUDAX_REQUIRE(managed_sym.size == sizeof(int));

    (void) lib.release(); // prevent library unload in destructor
  }

  // Has unified function
  if (with_unified_function)
  {
    STATIC_REQUIRE(
      cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().has_unified_function(unified_function_name)),
                           bool>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib.has_unified_function(unified_function_name));
    CUDAX_REQUIRE(!lib.has_unified_function("non_existent_unified"));

    (void) lib.release(); // prevent library unload in destructor
  }

  // Get unified function
  if (with_unified_function)
  {
    STATIC_REQUIRE(cuda::std::is_same_v<
                   decltype(cuda::std::declval<cudax::library_ref>().unified_function<int()>(unified_function_name)),
                   int (*)()>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    auto unified_fn    = lib.unified_function<int()>(unified_function_name);

    void* unified_fn_addr;
    CUDAX_REQUIRE(
      ::cuda::__driver::__libraryGetUnifiedFunctionNoThrow(unified_fn_addr, lib1_native, managed_symbol_name)
      == cudaSuccess);

    CUDAX_REQUIRE(reinterpret_cast<void*>(unified_fn) == unified_fn_addr);

    (void) lib.release(); // prevent library unload in destructor
  }

  // Get handle
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().get()), CUlibrary>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib.get() == lib1_native);

    (void) lib.release(); // prevent library unload in destructor
  }

  // Release handle
  {
    STATIC_REQUIRE(cuda::std::is_same_v<decltype(cuda::std::declval<cudax::library>().release()), CUlibrary>);

    cudax::library lib = cudax::library::from_native_handle(lib1_native);
    CUDAX_REQUIRE(lib.get() == lib1_native);

    CUlibrary released_handle = lib.release();
    CUDAX_REQUIRE(released_handle == lib1_native);
    CUDAX_REQUIRE(lib.get() == CUlibrary{});

    // lib is in a moved-from state
  }

  // Equality/Inequality comparison
  {
    cudax::library lib1 = cudax::library::from_native_handle(lib1_native);
    cudax::library lib2 = cudax::library::from_native_handle(lib2_native);

    CUDAX_REQUIRE(lib1 == lib1);
    CUDAX_REQUIRE(lib1 != lib2);

    (void) lib1.release(); // prevent library unload in destructor
    (void) lib2.release(); // prevent library unload in destructor
  }

  // Destructor
  {
    cudax::library lib1 = cudax::library::from_native_handle(lib1_native);
    cudax::library lib2 = cudax::library::from_native_handle(lib2_native);

    // lib1 and lib2 will be destroyed here, which will unload the libraries
  }
}
