//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory_resource/shared_block_ptr.h>
#include <cuda/std/type_traits>

#include <testing.cuh>

struct trivial_payload
{
  int __value;
  explicit trivial_payload(int __v)
      : __value(__v)
  {}
};

struct counting_payload
{
  static int __construct_count;
  static int __destruct_count;

  static void __reset()
  {
    __construct_count = 0;
    __destruct_count  = 0;
  }

  int __value;

  explicit counting_payload(int __v)
      : __value(__v)
  {
    ++__construct_count;
  }

  ~counting_payload()
  {
    ++__destruct_count;
  }

  counting_payload(const counting_payload&)            = delete;
  counting_payload& operator=(const counting_payload&) = delete;
  counting_payload(counting_payload&&)                 = delete;
  counting_payload& operator=(counting_payload&&)      = delete;
};

int counting_payload::__construct_count = 0;
int counting_payload::__destruct_count  = 0;

// --- static assertions ---

static_assert(!cuda::std::is_trivially_destructible<cuda::mr::__shared_block_ptr<trivial_payload>>::value);
static_assert(cuda::std::is_copy_constructible<cuda::mr::__shared_block_ptr<trivial_payload>>::value);
static_assert(cuda::std::is_move_constructible<cuda::mr::__shared_block_ptr<trivial_payload>>::value);
static_assert(cuda::std::is_copy_assignable<cuda::mr::__shared_block_ptr<trivial_payload>>::value);
static_assert(cuda::std::is_move_assignable<cuda::mr::__shared_block_ptr<trivial_payload>>::value);

// --- construction ---

C2H_CCCLRT_TEST("__shared_block_ptr default construction", "[memory_resource]")
{
  cuda::mr::__shared_block_ptr<trivial_payload> ptr;
  CHECK(!ptr);
}

C2H_CCCLRT_TEST("__shared_block_ptr payload construction", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> ptr(42);
    CHECK(static_cast<bool>(ptr));
    CHECK(ptr.__payload().__value == 42);
    CHECK(counting_payload::__construct_count == 1);
    CHECK(counting_payload::__destruct_count == 0);
  }
  CHECK(counting_payload::__destruct_count == 1);
}

// --- copy semantics ---

C2H_CCCLRT_TEST("__shared_block_ptr copy construction", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> ptr(10);
    {
      cuda::mr::__shared_block_ptr<counting_payload> copy(ptr); // NOLINT(performance-unnecessary-copy-initialization)
      CHECK(static_cast<bool>(copy));
      CHECK(copy.__payload().__value == 10);
      CHECK(&copy.__payload() == &ptr.__payload());
      CHECK(counting_payload::__destruct_count == 0);
    }
    // copy destroyed — payload still alive
    CHECK(counting_payload::__destruct_count == 0);
    CHECK(ptr.__payload().__value == 10);
  }
  CHECK(counting_payload::__destruct_count == 1);
}

C2H_CCCLRT_TEST("__shared_block_ptr copy assignment", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> a(1);
    cuda::mr::__shared_block_ptr<counting_payload> b(2);
    CHECK(counting_payload::__construct_count == 2);

    b = a;
    // b's old payload destroyed, b now shares with a
    CHECK(counting_payload::__destruct_count == 1);
    CHECK(b.__payload().__value == 1);
    CHECK(&b.__payload() == &a.__payload());
  }
  CHECK(counting_payload::__destruct_count == 2);
}

// --- move semantics ---

C2H_CCCLRT_TEST("__shared_block_ptr move construction", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> ptr(99);
    cuda::mr::__shared_block_ptr<counting_payload> moved(cuda::std::move(ptr));
    CHECK(static_cast<bool>(moved));
    CHECK(!ptr);
    CHECK(moved.__payload().__value == 99);
    CHECK(counting_payload::__destruct_count == 0);
  }
  CHECK(counting_payload::__destruct_count == 1);
}

C2H_CCCLRT_TEST("__shared_block_ptr move assignment", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> a(1);
    cuda::mr::__shared_block_ptr<counting_payload> b(2);

    b = cuda::std::move(a);
    // b's old payload destroyed, b took a's block
    CHECK(counting_payload::__destruct_count == 1);
    CHECK(static_cast<bool>(b));
    CHECK(!a);
    CHECK(b.__payload().__value == 1);
  }
  CHECK(counting_payload::__destruct_count == 2);
}

// --- equality ---

C2H_CCCLRT_TEST("__shared_block_ptr equality", "[memory_resource]")
{
  cuda::mr::__shared_block_ptr<trivial_payload> a(1);
  cuda::mr::__shared_block_ptr<trivial_payload> b(2);
  cuda::mr::__shared_block_ptr<trivial_payload> a_copy(a); // NOLINT(performance-unnecessary-copy-initialization)
  cuda::mr::__shared_block_ptr<trivial_payload> null;

  CHECK(a == a);
  CHECK(a == a_copy);
  CHECK(a != b);
  CHECK(a != null);
  CHECK(null == null);
}

// --- swap ---

C2H_CCCLRT_TEST("__shared_block_ptr swap", "[memory_resource]")
{
  cuda::mr::__shared_block_ptr<trivial_payload> a(1);
  cuda::mr::__shared_block_ptr<trivial_payload> b(2);

  auto* a_payload = &a.__payload();
  auto* b_payload = &b.__payload();

  a.swap(b);
  CHECK(&a.__payload() == b_payload);
  CHECK(&b.__payload() == a_payload);
}

// --- refcount behavior ---

C2H_CCCLRT_TEST("__shared_block_ptr refcount multiple copies", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> p1(42);
    {
      cuda::mr::__shared_block_ptr<counting_payload> p2(p1); // NOLINT(performance-unnecessary-copy-initialization)
      {
        cuda::mr::__shared_block_ptr<counting_payload> p3(p2); // NOLINT(performance-unnecessary-copy-initialization)
        CHECK(counting_payload::__destruct_count == 0);
      }
      CHECK(counting_payload::__destruct_count == 0);
    }
    CHECK(counting_payload::__destruct_count == 0);
  }
  CHECK(counting_payload::__destruct_count == 1);
  CHECK(counting_payload::__construct_count == 1);
}

// --- null operations ---

C2H_CCCLRT_TEST("__shared_block_ptr null copy and move", "[memory_resource]")
{
  cuda::mr::__shared_block_ptr<trivial_payload> null;
  cuda::mr::__shared_block_ptr<trivial_payload> null_copy(null); // NOLINT(performance-unnecessary-copy-initialization)
  cuda::mr::__shared_block_ptr<trivial_payload> null_moved(cuda::std::move(null));

  CHECK(!null_copy);
  CHECK(!null_moved);
}

// --- self assignment ---

C2H_CCCLRT_TEST("__shared_block_ptr self assignment", "[memory_resource]")
{
  counting_payload::__reset();
  {
    cuda::mr::__shared_block_ptr<counting_payload> ptr(7);
    auto& ref = ptr;
    ptr       = ref;
    CHECK(static_cast<bool>(ptr));
    CHECK(ptr.__payload().__value == 7);
    CHECK(counting_payload::__destruct_count == 0);
  }
  CHECK(counting_payload::__destruct_count == 1);
}
