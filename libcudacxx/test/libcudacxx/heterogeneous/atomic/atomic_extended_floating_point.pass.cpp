//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
// UNSUPPORTED: force-tile
// The heterogeneous harness uses host threads and the atomic backends use inline PTX.

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/atomic>
#include <cuda/std/__floating_point/cast.h>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../helpers.h"
#include "cuda_space_selector.h"
#include "test_macros.h"

template <class T>
TEST_HOST_DEVICE_FUNC T value(float v)
{
  return cuda::std::__fp_cast<T>(v);
}

template <class T>
TEST_HOST_DEVICE_FUNC float to_float(T v)
{
  return cuda::std::__fp_cast<float>(v);
}

template <class Atomic>
struct owned_object
{
  using value_type = typename cuda::std::remove_cv_t<Atomic>::value_type;
  Atomic atom{};
  TEST_HOST_DEVICE_FUNC Atomic& get()
  {
    return atom;
  }
};

template <class T, class Ref>
struct referenced_object
{
  using value_type = cuda::std::remove_cv_t<T>;
  alignas(4) T data{};
  cuda::std::uint16_t neighbor = 0x5aa5;

  TEST_HOST_DEVICE_FUNC Ref get()
  {
    return Ref{data};
  }
};

struct arithmetic_tester
{
  template <class Object>
  TEST_HOST_DEVICE_FUNC static void initialize(Object& object)
  {
    using T             = typename Object::value_type;
    decltype(auto) atom = object.get();
    static_assert(noexcept(atom.fetch_add(cuda::std::declval<T>())));
    static_assert(noexcept(atom.fetch_sub(cuda::std::declval<T>())));
    static_assert(cuda::std::is_same_v<decltype(atom += value<T>(1.0f)), T>);
    static_assert(cuda::std::is_same_v<decltype(atom -= value<T>(1.0f)), T>);

    for (auto order :
         {cuda::std::memory_order_relaxed,
          cuda::std::memory_order_acquire,
          cuda::std::memory_order_release,
          cuda::std::memory_order_acq_rel,
          cuda::std::memory_order_seq_cst})
    {
      atom.store(value<T>(-1.5f));
      assert(to_float(atom.fetch_add(value<T>(0.25f), order)) == -1.5f);
      assert(to_float(atom.load()) == -1.25f);
      assert(to_float(atom.fetch_sub(value<T>(2.5f), order)) == -1.25f);
      assert(to_float(atom.load()) == -3.75f);
    }
    assert(to_float(atom += value<T>(1.5f)) == -2.25f);
    assert(to_float(atom -= value<T>(0.5f)) == -2.75f);
    assert(to_float(atom.load()) == -2.75f);

    // Round each update to the stored format, including the compound assignment's result.
    auto halfway = cuda::std::is_same_v<T, __half> ? 0.00048828125f : 0.00390625f;
    atom.store(value<T>(1.0f));
    assert(to_float(atom.fetch_add(value<T>(halfway))) == 1.0f);
    assert(to_float(atom.load()) == 1.0f);
    assert(to_float(atom += value<T>(halfway)) == 1.0f);
    assert(to_float(atom.fetch_sub(value<T>(halfway * 0.5f))) == 1.0f);
    assert(to_float(atom.load()) == 1.0f);
    assert(to_float(atom -= value<T>(halfway * 0.5f)) == 1.0f);

    auto negative_zero = cuda::std::bit_cast<T>(cuda::std::uint16_t{0x8000});
    atom.store(negative_zero);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.fetch_add(value<T>(0.0f))) == 0x8000);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.load()) == 0);
    atom.store(negative_zero);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.fetch_sub(value<T>(0.0f))) == 0x8000);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.load()) == 0x8000);

    auto subnormal = cuda::std::bit_cast<T>(cuda::std::uint16_t{1});
    atom.store(subnormal);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.fetch_add(subnormal)) == 1);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.load()) == 2);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.fetch_sub(subnormal)) == 2);
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.load()) == 1);

    // CAS retries compare representations, so a NaN cannot make the loop spin forever.
    auto nan_bits = cuda::std::uint16_t{cuda::std::is_same_v<T, __half> ? 0x7e15 : 0x7fc5};
    atom.store(cuda::std::bit_cast<T>(nan_bits));
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.fetch_add(value<T>(1.0f))) == nan_bits);
    assert(cuda::std::isnan(to_float(atom.load())));
    assert(cuda::std::isnan(to_float(atom.fetch_sub(value<T>(1.0f)))));
    assert(cuda::std::isnan(to_float(atom.load())));
    auto infinity_bits = cuda::std::uint16_t{cuda::std::is_same_v<T, __half> ? 0x7c00 : 0x7f80};
    atom.store(cuda::std::bit_cast<T>(infinity_bits));
    assert(cuda::std::isinf(to_float(atom.fetch_add(value<T>(1.0f)))));
    assert(cuda::std::isinf(to_float(atom.fetch_sub(value<T>(1.0f)))));
    assert(cuda::std::bit_cast<cuda::std::uint16_t>(atom.load()) == infinity_bits);
    atom.store(value<T>(7.0f));
  }

  template <class Object>
  TEST_HOST_DEVICE_FUNC static void validate(Object& object)
  {
    assert(to_float(object.get().load()) == 7.0f);
  }

  template <class T, class Ref>
  TEST_HOST_DEVICE_FUNC static void validate(referenced_object<T, Ref>& object)
  {
    assert(to_float(object.get().load()) == 7.0f);
    assert(object.neighbor == 0x5aa5);
  }
};

struct free_function_tester
{
  template <class Object>
  TEST_HOST_DEVICE_FUNC static void initialize(Object& object)
  {
    using T    = typename Object::value_type;
    auto& atom = object.get();
    atom.store(value<T>(1.5f));
    assert(to_float(cuda::std::atomic_fetch_add(&atom, value<T>(0.5f))) == 1.5f);
    assert(to_float(cuda::std::atomic_fetch_sub(&atom, value<T>(0.25f))) == 2.0f);
    for (auto order :
         {cuda::std::memory_order_relaxed,
          cuda::std::memory_order_acquire,
          cuda::std::memory_order_release,
          cuda::std::memory_order_acq_rel,
          cuda::std::memory_order_seq_cst})
    {
      assert(to_float(cuda::std::atomic_fetch_add_explicit(&atom, value<T>(0.5f), order)) == 1.75f);
      assert(to_float(cuda::std::atomic_fetch_sub_explicit(&atom, value<T>(0.5f), order)) == 2.25f);
    }
  }

  template <class Object>
  TEST_HOST_DEVICE_FUNC static void validate(Object& object)
  {
    assert(to_float(object.get().load()) == 1.75f);
  }
};

template <class T, cuda::thread_scope Scope>
void test_scope()
{
  using tests = tester_list<arithmetic_tester>;
  validate_pinned<owned_object<cuda::atomic<T, Scope>>, tests>();
  validate_pinned<owned_object<volatile cuda::atomic<T, Scope>>, tests>();
  validate_pinned<referenced_object<T, cuda::atomic_ref<T, Scope>>, tests>();
  validate_pinned<referenced_object<volatile T, cuda::atomic_ref<volatile T, Scope>>, tests>();
}

template <class Object>
TEST_DEVICE_FUNC void test_memory_spaces()
{
  {
    local_memory_selector<Object, constructor_initializer> selector;
    Object& object = *selector.construct();
    arithmetic_tester::initialize(object);
    arithmetic_tester::validate(object);
  }
  {
    shared_memory_selector<Object, constructor_initializer> selector;
    Object& object = *selector.construct();
    arithmetic_tester::initialize(object);
    arithmetic_tester::validate(object);
  }
}

template <class T>
__global__ void memory_spaces_kernel()
{
  test_memory_spaces<owned_object<cuda::atomic<T, cuda::thread_scope_block>>>();
  test_memory_spaces<owned_object<volatile cuda::atomic<T, cuda::thread_scope_block>>>();
  test_memory_spaces<referenced_object<T, cuda::atomic_ref<T, cuda::thread_scope_block>>>();
  test_memory_spaces<referenced_object<volatile T, cuda::atomic_ref<volatile T, cuda::thread_scope_block>>>();
  test_memory_spaces<owned_object<cuda::std::atomic<T>>>();
  test_memory_spaces<owned_object<volatile cuda::std::atomic<T>>>();
  test_memory_spaces<referenced_object<T, cuda::std::atomic_ref<T>>>();
  test_memory_spaces<referenced_object<volatile T, cuda::std::atomic_ref<volatile T>>>();
}

template <class Atomic, class = void>
struct has_fetch_add : cuda::std::false_type
{};

template <class Atomic>
struct has_fetch_add<Atomic,
                     cuda::std::void_t<decltype(cuda::std::declval<Atomic&>().fetch_add(
                       cuda::std::declval<typename Atomic::value_type>()))>> : cuda::std::true_type
{};

template <class T>
__global__ void contended_kernel()
{
  using Atomic = cuda::atomic<T, cuda::thread_scope_block>;
  __shared__ alignas(4) unsigned char owned_storage[sizeof(Atomic)];
  __shared__ alignas(4) unsigned char referenced_storage[2 * sizeof(T)];
  auto* owned    = reinterpret_cast<Atomic*>(owned_storage);
  auto* elements = reinterpret_cast<T*>(referenced_storage);
  if (threadIdx.x == 0)
  {
    new (owned) Atomic(value<T>(0.0f));
    new (elements) T(value<T>(0.0f));
    new (elements + 1) T(value<T>(0.0f));
  }
  __syncthreads();
  cuda::atomic_ref<T, cuda::thread_scope_block> ref(elements[threadIdx.x % 2]);
  for (int i = 0; i < 16; ++i)
  {
    ref.fetch_add(value<T>(1.0f), cuda::std::memory_order_relaxed);
    if (threadIdx.x < 16)
    {
      owned->fetch_add(value<T>(1.0f), cuda::std::memory_order_relaxed);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
  {
    assert(to_float(owned->load()) == 256.0f);
    assert(to_float(elements[0]) == 256.0f);
    assert(to_float(elements[1]) == 256.0f);
  }
}

template <class T>
void test_type()
{
  static_assert(has_fetch_add<cuda::atomic<T>>::value);
  static_assert(has_fetch_add<cuda::atomic_ref<T>>::value);
  static_assert(!has_fetch_add<cuda::atomic_ref<const T>>::value);
  static_assert(!has_fetch_add<cuda::std::atomic_ref<const T>>::value);
  static_assert(sizeof(cuda::atomic<T>) == sizeof(cuda::atomic<cuda::std::uint16_t>));
  static_assert(alignof(cuda::atomic<T>) == alignof(cuda::atomic<cuda::std::uint16_t>));
  static_assert(sizeof(cuda::std::atomic<T>) == sizeof(cuda::std::atomic<cuda::std::uint16_t>));
  test_scope<T, cuda::thread_scope_block>();
  test_scope<T, cuda::thread_scope_cluster>();
  test_scope<T, cuda::thread_scope_device>();
  test_scope<T, cuda::thread_scope_system>();
  using tests = tester_list<arithmetic_tester>;
  validate_pinned<owned_object<cuda::std::atomic<T>>, tests>();
  validate_pinned<owned_object<volatile cuda::std::atomic<T>>, tests>();
  validate_pinned<referenced_object<T, cuda::std::atomic_ref<T>>, tests>();
  validate_pinned<referenced_object<volatile T, cuda::std::atomic_ref<volatile T>>, tests>();
  using free_tests = tester_list<free_function_tester>;
  validate_pinned<owned_object<cuda::std::atomic<T>>, free_tests>();
  validate_pinned<owned_object<volatile cuda::std::atomic<T>>, free_tests>();
  memory_spaces_kernel<T><<<1, 1>>>();
  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
  contended_kernel<T><<<1, 32>>>();
  HETEROGENEOUS_SAFE_CALL(cudaGetLastError());
  HETEROGENEOUS_SAFE_CALL(cudaDeviceSynchronize());
}

void kernel_invoker()
{
#if _CCCL_HAS_NVFP8_E4M3()
  static_assert(!has_fetch_add<cuda::atomic<__nv_fp8_e4m3>>::value);
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP16()
  test_type<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))
  return 0;
}
