//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#undef NDEBUG

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/memory.cuh>

#include <utility.cuh>

__global__ void test_static_shared_storage()
{
  constexpr cuda::std::size_t size  = 32;
  constexpr cuda::std::size_t align = 16;

  using SharedStorage = cudax::static_shared_storage<size, align>;

  // 0. Test static public members.
  static_assert(SharedStorage::size == size);
  static_assert(SharedStorage::alignment == align);

  // 1. Test that the type is nothrow default constructible.
  static_assert(cuda::std::is_nothrow_default_constructible_v<SharedStorage>);

  // 2. Test that the type is not copyable.
  static_assert(!cuda::std::is_copy_constructible_v<SharedStorage>);

  // 3. Test that the type is not movable.
  static_assert(!cuda::std::is_move_constructible_v<SharedStorage>);

  // 4. Test that the type is trivially destructible.
  static_assert(cuda::std::is_trivially_destructible_v<SharedStorage>);

  // 5. Test that the type is not copy assignable.
  static_assert(!cuda::std::is_copy_assignable_v<SharedStorage>);

  // 6. Test that the type is not move assignable.
  static_assert(!cuda::std::is_move_assignable_v<SharedStorage>);

  // 7. Test that if the type is constructed multiple times, the actual address of the objects is not the same.
  {
    SharedStorage a;
    SharedStorage b;
    assert(a.get() != b.get());
  }

  // 8. Test get method.
  static_assert(
    cuda::std::is_same_v<cudax::shared_memory_ptr<void>, decltype(cuda::std::declval<const SharedStorage>().get())>);
  static_assert(noexcept(cuda::std::declval<const SharedStorage>().get()));
  {
    SharedStorage a;
    assert(&a
           == cudax::shared_memory_ptr{::__cvta_shared_to_generic(cuda::std::to_underlying((&a).__get_smem_addr()))});
  }

  // 9. Test operator&.
  static_assert(
    cuda::std::is_same_v<cudax::shared_memory_ptr<void>, decltype(&cuda::std::declval<const SharedStorage>())>);
  static_assert(noexcept(&cuda::std::declval<const SharedStorage>()));
  {
    SharedStorage a;
    assert(a.get() == &a);
  }

  // 10. Test that the object really is in shared memory.
  {
    SharedStorage a;
    assert(__isShared((&a).get()));
  }
}

C2H_TEST("Static shared storage", "")
{
  test_static_shared_storage<<<1, 1>>>();
  CUDAX_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

enum class State
{
  uninitialized,
  constructed,
  destroyed,
};

__device__ State state{State::uninitialized};

__device__ void reset_state()
{
  __syncthreads();
  if (threadIdx.x == 0)
  {
    state = State::uninitialized;
  }
  __syncthreads();
}

__device__ void check_state(State state)
{
  __syncthreads();
  assert(state == state);
}

struct TestType
{
  __device__ TestType(int value) noexcept
      : value_{value}
  {
    state = State::constructed;
  }

  __device__ ~TestType()
  {
    state = State::destroyed;
  }

  int value_;
};

__global__ void test_static_shared()
{
  using SST = cudax::static_shared<TestType>;

  // 0. Test static public members.
  static_assert(SST::default_thread_index == uint3{0, 0, 0});
  static_assert(SST::size == sizeof(TestType));
  static_assert(SST::alignment == alignof(TestType));

  // 1. Test that the type is not default constructible.
  static_assert(!cuda::std::is_default_constructible_v<SST>);

  // 2. Test that the type is nothrow constructible from cuda::no_init_t;
  static_assert(cuda::std::is_nothrow_constructible_v<SST, cuda::no_init_t>);
  reset_state();
  {
    SST a{cuda::no_init};
    check_state(State::uninitialized);
  }
  check_state(State::destroyed);

  // 3. Test that the type is nothrow constructible from int.
  static_assert(cuda::std::is_nothrow_constructible_v<SST, int>);
  reset_state();
  {
    SST a{10};
    check_state(State::constructed);

    assert(a.get().value_ == 10);
  }
  check_state(State::destroyed);

  // 4. Test that the type is nothrow constructible from short.
  static_assert(cuda::std::is_nothrow_constructible_v<SST, short>);

  // 5. Test that the type is not copyable.
  static_assert(!cuda::std::is_copy_constructible_v<SST>);

  // 6. Test that the type is not movable.
  static_assert(!cuda::std::is_move_constructible_v<SST>);

  // 7. Test that the type is not copy assignable.
  static_assert(!cuda::std::is_copy_assignable_v<SST>);

  // 8. Test that the type is not move assignable.
  static_assert(!cuda::std::is_move_assignable_v<SST>);

  // 9. Test that if the type is constructed multiple times, the actual address of the objects is not the same.
  {
    SST a{cuda::no_init};
    SST b{cuda::no_init};
    assert(&a.get() != &b.get());
  }

  // 10. Test the construct method.
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<SST>().construct(int{}))>);
  reset_state();
  {
    SST a{cuda::no_init};
    check_state(State::uninitialized);

    a.construct(10);
    check_state(State::constructed);

    assert(a.get().value_ == 10);
  }
  check_state(State::destroyed);

  // 11. Test the construct_by method.
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<SST>().construct_by(uint3{}, int{}))>);
  reset_state();
  {
    SST a{cuda::no_init};
    check_state(State::uninitialized);

    // Construct the object by thread 1 instead of thread 0.
    a.construct_by(uint3{1, 0, 0}, 10);
    check_state(State::constructed);

    assert(a.get().value_ == 10);
  }
  check_state(State::destroyed);

  // 12. Test the destroy method.
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<SST>().destroy())>);
  reset_state();
  {
    SST a{cuda::no_init};
    check_state(State::uninitialized);

    a.destroy();
    check_state(State::destroyed);
  }
  check_state(State::destroyed);

  reset_state();
  {
    SST a{10};
    check_state(State::uninitialized);

    a.destroy();
    check_state(State::destroyed);
  }
  check_state(State::destroyed);

  // 13. Test the destroy_by method.
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<SST>().destroy())>);
  {
    const uint3 tid{1, 0, 0};

    reset_state();
    {
      SST a{cuda::no_init};
      check_state(State::uninitialized);

      a.destroy_by(tid);
      check_state(State::destroyed);
    }
    check_state(State::destroyed);

    reset_state();
    {
      SST a{10};
      check_state(State::uninitialized);

      a.destroy_by(tid);
      check_state(State::destroyed);
    }
    check_state(State::destroyed);
  }

  // 14. Test get method.
  static_assert(cuda::std::is_same_v<TestType&, decltype(cuda::std::declval<const SST>().get())>);
  static_assert(noexcept(cuda::std::declval<const SST>().get()));
  {
    SST a{128};

    __syncthreads();
    assert(a.get().value_ == 128);
  }

  // 15. Test operator T&.
  static_assert(cuda::std::is_nothrow_convertible_v<const SST, TestType&>);
  {
    SST a{128};

    __syncthreads();
    TestType& b = a;
    assert(b.value_ == 128);
  }

  // 16. Test operator&.
  static_assert(cuda::std::is_same_v<cudax::shared_memory_ptr<TestType>, decltype(&cuda::std::declval<const SST&>())>);
  static_assert(noexcept(&cuda::std::declval<const SST&>()));
  {
    SST a{128};
    assert((&a).get() != nullptr);
  }

  // 17. Test that the object really is in shared memory.
  {
    SST a{128};
    assert(cuda::device::is_object_from(a.get(), cuda::device::address_space::shared));
  }
}

C2H_TEST("Static shared", "")
{
  test_static_shared<<<1, 2>>>();
  CUDAX_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}

__global__ void test_shared_memory_ptr()
{
  using T   = int;
  using SMP = cudax::shared_memory_ptr<T>;

  // 0. Test public type aliases.
  static_assert(cuda::std::is_same_v<T, SMP::element_type>);
  static_assert(cuda::std::is_same_v<T*, SMP::pointer>);

  // 1. Test that the type is not default constructible.
  static_assert(!cuda::std::is_default_constructible_v<SMP>);

  // 2. Test that the type is not constructible from nullptr_t.
  static_assert(!cuda::std::is_constructible_v<SMP, cuda::std::nullptr_t>);

  // 3. Test that the type is nothrow constructible from cuda::no_init_t.
  static_assert(cuda::std::is_nothrow_constructible_v<SMP, cuda::no_init_t>);

  // 4. Test that the type is explicitly nothrow constructible from T*.
  static_assert(cuda::std::is_nothrow_constructible_v<SMP, T*>);
  static_assert(!cuda::std::is_convertible_v<T*, SMP>);
  {
    __shared__ T value;
    SMP p{&value};
    assert(p.get() == &value);
  }

  // 7. Test that the type is nothrow constructible from another instance of different type.
  // static_assert(cuda::std::is_nothrow_constructible_v<cudax::static_shared<void>, SMP>);
  // static_assert(cuda::std::is_convertible_v<SMP, cudax::static_shared<void>>);
  // {
  //   __shared__ T value;
  //   SMP a{&value};
  //   cudax::static_shared<void> p{a};
  //   assert(a.get() == p.get());
  // }

  // 8. Test that the type is trivially copyable.
  static_assert(cuda::std::is_trivially_copyable_v<SMP>);

  // 9. Test that the type is trivially movable.
  static_assert(cuda::std::is_trivially_move_constructible_v<SMP>);

  // 10. Test that the type is trivially copy assignable.
  static_assert(cuda::std::is_trivially_copy_assignable_v<SMP>);

  // 11. Test that the type is trivially move assignable.
  static_assert(cuda::std::is_trivially_move_assignable_v<SMP>);

  // 12. Test reset method.
  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::declval<SMP>().reset(cuda::std::declval<T*>()))>);
  static_assert(noexcept(cuda::std::declval<SMP>().reset(cuda::std::declval<T*>())));
  {
    __shared__ T value;
    SMP a{cuda::no_init};
    a.reset(&value);
    assert(a.get() == &value);
  }

  // 13. Test swap method.
  static_assert(cuda::std::is_same_v<void, decltype(cuda::std::declval<SMP>().swap(cuda::std::declval<SMP&>()))>);
  static_assert(noexcept(cuda::std::declval<SMP>().swap(cuda::std::declval<SMP&>())));
  {
    __shared__ T value1;
    __shared__ T value2;
    SMP a{&value1};
    SMP b{&value2};
    a.swap(b);
    assert(a.get() == &value2);
    assert(b.get() == &value1);
  }

  // 14. Test get method.
  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::declval<const SMP>().get())>);
  static_assert(noexcept(cuda::std::declval<const SMP>().get()));
  {
    __shared__ T value;
    const SMP a{&value};
    assert(a.get() == &value);
  }

  // 14. Test operator bool.
  static_assert(!cuda::std::is_convertible_v<const SMP, bool>);
  static_assert(noexcept(cuda::std::declval<const SMP>().operator bool()));
  {
    const SMP a{cuda::no_init};
    assert(static_cast<bool>(a));
  }

  // 15. Test operator->.
  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::declval<const SMP>().operator->())>);
  static_assert(noexcept(cuda::std::declval<const SMP>().operator->()));
  {
    __shared__ T value;
    const SMP a{&value};
    assert(a.operator->() == &value);
  }

  // 16. Test operator*.
  static_assert(cuda::std::is_same_v<T&, decltype(cuda::std::declval<const SMP>().operator*())>);
  static_assert(noexcept(cuda::std::declval<const SMP>().operator*()));
  {
    __shared__ T value;
    const SMP a{&value};
    assert(&a.operator*() == &value);
  }

  // 17. Test operator T*.
  static_assert(cuda::std::is_same_v<T*, decltype(cuda::std::declval<const SMP>().operator T*())>);
  static_assert(!cuda::std::is_convertible_v<const SMP, T*>);
  static_assert(noexcept(cuda::std::declval<const SMP>().operator T*()));
  {
    __shared__ T value;
    const SMP a{&value};
    assert(a.operator T*() == &value);
  }
}

C2H_TEST("Shared memory pointer", "")
{
  test_shared_memory_ptr<<<1, 1>>>();
  CUDAX_REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
}
