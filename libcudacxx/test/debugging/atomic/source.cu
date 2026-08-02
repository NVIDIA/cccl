// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/atomic>

template <class T>
[[gnu::noinline]] void keep_for_debugger(const T& value)
{
  asm volatile("" : : "g"(&value) : "memory");
}

struct alignas(16) payload
{
  int first;
  int second;
  int third;
};

struct small_payload
{
  bool first;
  bool second;
};

static_assert(sizeof(payload) > 8);
static_assert(alignof(payload) >= cuda::std::atomic_ref<payload>::required_alignment);
static_assert(sizeof(small_payload) < 4);

using atomic_alias = cuda::std::atomic<long long>;

[[gnu::noinline]] void inspect_integer(const cuda::std::atomic<int>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_boolean(const cuda::std::atomic<bool>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_small_object(const cuda::std::atomic<small_payload>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_floating_point(const cuda::std::atomic<double>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_pointer(const cuda::std::atomic<int*>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_locked(const cuda::std::atomic<payload>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_alias(const atomic_alias& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_nested(const cuda::std::array<cuda::std::atomic<int>, 2>& values)
{
  keep_for_debugger(values);
}

[[gnu::noinline]] void inspect_reference(const cuda::std::atomic_ref<int>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_locked_reference(const cuda::std::atomic_ref<payload>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_before_update(const cuda::std::atomic<int>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_after_update(const cuda::std::atomic<int>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_reference_before_update(const cuda::std::atomic_ref<int>& value)
{
  keep_for_debugger(value);
}

[[gnu::noinline]] void inspect_reference_after_update(const cuda::std::atomic_ref<int>& value)
{
  keep_for_debugger(value);
}

int main()
{
  const cuda::std::atomic<int> integer{-7};
  const cuda::std::atomic<bool> boolean{true};
  const cuda::std::atomic<small_payload> small_object{small_payload{true, false}};
  const cuda::std::atomic<double> floating_point{-3.5};
  int pointee = 42;
  const cuda::std::atomic<int*> pointer{&pointee};
  const cuda::std::atomic<payload> locked{payload{13, -5, 88}};
  const atomic_alias alias{-31};
  cuda::std::array<cuda::std::atomic<int>, 2> nested{};
  nested[0].store(17);
  nested[1].store(-64);

  int referenced = 52;
  const cuda::std::atomic_ref<int> reference{referenced};
  payload referenced_payload{6, -91, 3};
  const cuda::std::atomic_ref<payload> locked_reference{referenced_payload};

  cuda::std::atomic<int> updated{85};
  int updated_target = -12;
  cuda::std::atomic_ref<int> updated_reference{updated_target};

  inspect_integer(integer);
  inspect_boolean(boolean);
  inspect_small_object(small_object);
  inspect_floating_point(floating_point);
  inspect_pointer(pointer);
  inspect_locked(locked);
  inspect_alias(alias);
  inspect_nested(nested);
  inspect_reference(reference);
  inspect_locked_reference(locked_reference);
  inspect_before_update(updated);
  updated.store(99);
  inspect_after_update(updated);
  inspect_reference_before_update(updated_reference);
  updated_target = 73;
  inspect_reference_after_update(updated_reference);
}
