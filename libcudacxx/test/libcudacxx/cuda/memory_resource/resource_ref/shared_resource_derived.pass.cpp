//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// Regression test for https://github.com/NVIDIA/cccl/issues/8037
//
// Constructing a resource_ref from a type that:
// (1) Publicly inherits from cuda::mr::shared_resource<Impl>, AND
// (2) Has a constructor accepting resource_ref
// triggered a recursive constraint satisfaction error on GCC 14.3.0 with C++20:
//
//   error: satisfaction of atomic constraint '...' depends on itself
//
// Root cause: resource_ref previously used __iasync_resource which includes
// __icopyable.  Checking __icopyable<T> evaluates copyable<T>,
// which evaluates is_constructible<T, T&>.  When T's constructors include one
// accepting resource_ref, the compiler considers whether T& converts to
// resource_ref, re-entering the original __satisfies<T,...> check — a cycle.
//
// Fix: resource_ref and synchronous_resource_ref now use __iasync_resource_ref
// / __iresource_ref respectively, which omit __icopyable.  Reference wrappers
// don't own the resource and therefore don't need the underlying type to be
// copyable.

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda/std/cstddef>

struct my_resource_impl
{
  void* allocate(cuda::stream_ref, ::cuda::std::size_t, ::cuda::std::size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, ::cuda::std::size_t, ::cuda::std::size_t) noexcept {}
  void* allocate_sync(::cuda::std::size_t, ::cuda::std::size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, ::cuda::std::size_t, ::cuda::std::size_t) noexcept {}
  bool operator==(my_resource_impl const&) const
  {
    return true;
  }
  bool operator!=(my_resource_impl const&) const
  {
    return false;
  }
  friend void get_property(my_resource_impl const&, cuda::mr::device_accessible) noexcept {}
};

using resource_ref = cuda::mr::resource_ref<cuda::mr::device_accessible>;

// A type that inherits from shared_resource AND has a constructor taking
// resource_ref.  This pattern is common in RMM pool/arena resources.
struct derived_resource : public cuda::mr::shared_resource<my_resource_impl>
{
  using shared_base = cuda::mr::shared_resource<my_resource_impl>;

  explicit derived_resource(resource_ref /*upstream*/)
      : shared_base(cuda::std::in_place_type<my_resource_impl>)
  {}

  friend void get_property(derived_resource const&, cuda::mr::device_accessible) noexcept {}
};

void test()
{
  auto base_sr = cuda::mr::make_shared_resource<my_resource_impl>();
  resource_ref ref{base_sr};
  derived_resource dr{ref};

  // Previously caused: error: satisfaction of atomic constraint depends on
  // itself.  Must compile cleanly after the fix.
  ref = dr;

  // Workaround (casting to shared_resource base) should still compile
  ref = static_cast<derived_resource::shared_base&>(dr);
}

int main(int, char**)
{
  return 0;
}
