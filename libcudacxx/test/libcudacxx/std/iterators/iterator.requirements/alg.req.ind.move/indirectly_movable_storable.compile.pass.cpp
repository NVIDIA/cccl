//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class In, class Out>
// concept indirectly_movable_storable;

#include <cuda/std/iterator>

#include "MoveOnly.h"
#include "test_macros.h"

template <class T>
struct PointerTo
{
  using value_type = T;
  __host__ __device__ T& operator*() const;
};

// Copying the underlying object between pointers (or dereferenceable classes) works. This is a non-exhaustive check
// because this functionality comes from `indirectly_movable`.
static_assert(cuda::std::indirectly_movable_storable<int*, int*>);
static_assert(cuda::std::indirectly_movable_storable<const int*, int*>);
static_assert(!cuda::std::indirectly_movable_storable<int*, const int*>);
static_assert(!cuda::std::indirectly_movable_storable<const int*, const int*>);
static_assert(cuda::std::indirectly_movable_storable<int*, int[2]>);
static_assert(!cuda::std::indirectly_movable_storable<int[2], int*>);
#ifndef TEST_COMPILER_MSVC_2017 // MSVC2017 has issues determining common_reference
static_assert(cuda::std::indirectly_movable_storable<MoveOnly*, MoveOnly*>);
static_assert(cuda::std::indirectly_movable_storable<PointerTo<MoveOnly>, PointerTo<MoveOnly>>);
#endif // TEST_COMPILER_MSVC_2017

// The dereference operator returns a different type from `value_type` and the reference type cannot be assigned from a
// `ValueType`.
struct NoAssignment
{
  struct ValueType;

  struct ReferenceType
  {
    ReferenceType& operator=(ValueType) = delete;
  };

  // `ValueType` is convertible to `ReferenceType` but not assignable to it. This is implemented by explicitly deleting
  // `operator=(ValueType)` in `ReferenceType`.
  struct ValueType
  {
    __host__ __device__ operator ReferenceType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

// The case when `indirectly_writable<iter_rvalue_reference>` but not `indirectly_writable<iter_value>` (you can
// do `ReferenceType r = ValueType();` but not `r = ValueType();`).
static_assert(cuda::std::indirectly_writable<NoAssignment, cuda::std::iter_rvalue_reference_t<NoAssignment>>);
static_assert(!cuda::std::indirectly_writable<NoAssignment, cuda::std::iter_value_t<NoAssignment>>);
static_assert(!cuda::std::indirectly_movable_storable<NoAssignment, NoAssignment>);

struct DeletedMoveCtor
{
  DeletedMoveCtor(DeletedMoveCtor&&)            = delete;
  DeletedMoveCtor& operator=(DeletedMoveCtor&&) = default;
};

struct DeletedMoveAssignment
{
  DeletedMoveAssignment(DeletedMoveAssignment&&)            = default;
  DeletedMoveAssignment& operator=(DeletedMoveAssignment&&) = delete;
};

static_assert(!cuda::std::indirectly_movable_storable<DeletedMoveCtor*, DeletedMoveCtor*>);
static_assert(!cuda::std::indirectly_movable_storable<DeletedMoveAssignment*, DeletedMoveAssignment*>);

struct InconsistentIterator
{
  struct ValueType;

  struct ReferenceType
  {
    __host__ __device__ ReferenceType& operator=(ValueType const&);
  };

  struct ValueType
  {
    ValueType() = default;
    __host__ __device__ ValueType(const ReferenceType&);
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

// `ValueType` can be constructed with a `ReferenceType` and assigned to a `ReferenceType`, so it does model
// `indirectly_movable_storable`.
static_assert(cuda::std::indirectly_movable_storable<InconsistentIterator, InconsistentIterator>);

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not constructible from ReferenceType.
struct NotConstructibleFromRefIn
{
  struct CommonType
  {};

  struct ReferenceType
  {
    __host__ __device__ operator CommonType&() const;
  };

  struct ValueType
  {
    ValueType(ReferenceType) = delete;
    __host__ __device__ operator CommonType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::
  basic_common_reference<NotConstructibleFromRefIn::ValueType, NotConstructibleFromRefIn::ReferenceType, X, Y>
{
  using type = NotConstructibleFromRefIn::CommonType&;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::
  basic_common_reference<NotConstructibleFromRefIn::ReferenceType, NotConstructibleFromRefIn::ValueType, X, Y>
{
  using type = NotConstructibleFromRefIn::CommonType&;
};

static_assert(
  cuda::std::common_reference_with<NotConstructibleFromRefIn::ValueType&, NotConstructibleFromRefIn::ReferenceType&>);

struct AssignableFromAnything
{
  template <class T>
  __host__ __device__ AssignableFromAnything& operator=(T&&);
};

// A type that can't be constructed from its own reference isn't `indirectly_movable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(cuda::std::indirectly_movable_storable<int*, AssignableFromAnything*>);
static_assert(!cuda::std::indirectly_movable_storable<NotConstructibleFromRefIn, AssignableFromAnything*>);

// ReferenceType is a (proxy) reference for ValueType, but ValueType is not assignable from ReferenceType.
struct NotAssignableFromRefIn
{
  struct CommonType
  {};

  struct ReferenceType
  {
    __host__ __device__ operator CommonType&() const;
  };

  struct ValueType
  {
    __host__ __device__ ValueType(ReferenceType);
    ValueType& operator=(ReferenceType) = delete;
    __host__ __device__ operator CommonType&() const;
  };

  using value_type = ValueType;
  __host__ __device__ ReferenceType& operator*() const;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::basic_common_reference<NotAssignableFromRefIn::ValueType, NotAssignableFromRefIn::ReferenceType, X, Y>
{
  using type = NotAssignableFromRefIn::CommonType&;
};

template <template <class> class X, template <class> class Y>
struct cuda::std::basic_common_reference<NotAssignableFromRefIn::ReferenceType, NotAssignableFromRefIn::ValueType, X, Y>
{
  using type = NotAssignableFromRefIn::CommonType&;
};

static_assert(
  cuda::std::common_reference_with<NotAssignableFromRefIn::ValueType&, NotAssignableFromRefIn::ReferenceType&>);

// A type that can't be assigned from its own reference isn't `indirectly_movable_storable`, even when assigning it
// to a type that can be assigned from anything.
static_assert(!cuda::std::indirectly_movable_storable<NotAssignableFromRefIn, AssignableFromAnything*>);

int main(int, char**)
{
  return 0;
}
