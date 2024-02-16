//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// template<class In>
// concept indirectly_readable;

#include <cuda/std/iterator>

#include <cuda/std/concepts>

#include "test_macros.h"
#include "read_write.h"

template <class In>
TEST_HOST_DEVICE
constexpr bool check_indirectly_readable() {
  constexpr bool result = cuda::std::indirectly_readable<In>;
  static_assert(cuda::std::indirectly_readable<In const> == result);
  static_assert(cuda::std::indirectly_readable<In volatile> == result);
  static_assert(cuda::std::indirectly_readable<In const volatile> == result);
  static_assert(cuda::std::indirectly_readable<In const&> == result);
  static_assert(cuda::std::indirectly_readable<In volatile&> == result);
  static_assert(cuda::std::indirectly_readable<In const volatile&> == result);
  static_assert(cuda::std::indirectly_readable<In const&&> == result);
  static_assert(cuda::std::indirectly_readable<In volatile&&> == result);
  static_assert(cuda::std::indirectly_readable<In const volatile&&> == result);
  return result;
}

static_assert(!check_indirectly_readable<void*>());
static_assert(!check_indirectly_readable<void const*>());
static_assert(!check_indirectly_readable<void volatile*>());
static_assert(!check_indirectly_readable<void const volatile*>());

static_assert(check_indirectly_readable<int*>());
static_assert(check_indirectly_readable<int const*>());
static_assert(check_indirectly_readable<int volatile*>());
static_assert(check_indirectly_readable<int const volatile*>());

static_assert(check_indirectly_readable<value_type_indirection>());
static_assert(check_indirectly_readable<element_type_indirection>());
static_assert(check_indirectly_readable<proxy_indirection>());
static_assert(check_indirectly_readable<read_only_indirection>());

struct indirection_mismatch {
  using value_type = int;
  TEST_HOST_DEVICE float& operator*() const;
};
static_assert(!cuda::std::same_as<cuda::std::iter_value_t<indirection_mismatch>, cuda::std::iter_reference_t<indirection_mismatch> > &&
              check_indirectly_readable<indirection_mismatch>());
static_assert(!check_indirectly_readable<missing_dereference>());

// `iter_rvalue_reference_t` can't be missing unless the dereference operator is also missing.

struct iter_move_mismatch {
  using value_type = int;
  TEST_HOST_DEVICE value_type& operator*() const;

  TEST_HOST_DEVICE friend float& iter_move(iter_move_mismatch&);
};
static_assert(!check_indirectly_readable<iter_move_mismatch>());

struct indirection_and_iter_move_mismatch {
  using value_type = int;
  TEST_HOST_DEVICE float& operator*() const;

  TEST_HOST_DEVICE friend unsigned long long& iter_move(indirection_and_iter_move_mismatch&);
};
static_assert(!check_indirectly_readable<indirection_and_iter_move_mismatch>());

struct missing_iter_value_t {
  TEST_HOST_DEVICE int operator*() const;
};
static_assert(!check_indirectly_readable<missing_iter_value_t>());

struct unrelated_lvalue_ref_and_rvalue_ref {};

struct iter_ref1 {};
namespace cuda {
namespace std {
template <>
struct common_reference<iter_ref1&, iter_ref1&&> {};

template <>
struct common_reference<iter_ref1&&, iter_ref1&> {};
} // namespace std
} // namespace cuda
static_assert(!cuda::std::common_reference_with<iter_ref1&, iter_ref1&&>);

struct bad_iter_reference_t {
  using value_type = int;
  TEST_HOST_DEVICE iter_ref1& operator*() const;
};
static_assert(!check_indirectly_readable<bad_iter_reference_t>());

struct iter_ref2 {};
struct iter_rvalue_ref {};

struct unrelated_iter_ref_rvalue_and_iter_rvalue_ref_rvalue {
  using value_type = iter_ref2;
  TEST_HOST_DEVICE iter_ref2& operator*() const;
  TEST_HOST_DEVICE friend iter_rvalue_ref&& iter_move(unrelated_iter_ref_rvalue_and_iter_rvalue_ref_rvalue);
};
static_assert(!check_indirectly_readable<unrelated_iter_ref_rvalue_and_iter_rvalue_ref_rvalue>());

struct iter_ref3 {
  TEST_HOST_DEVICE operator iter_rvalue_ref() const;
};
namespace cuda {
namespace std {
template <template <class> class XQual, template <class> class YQual>
struct basic_common_reference<iter_ref3, iter_rvalue_ref, XQual, YQual> {
  using type = iter_rvalue_ref;
};
template <template <class> class XQual, template <class> class YQual>
struct basic_common_reference<iter_rvalue_ref, iter_ref3, XQual, YQual> {
  using type = iter_rvalue_ref;
};
} // namespace std
} // namespace cuda
static_assert(cuda::std::common_reference_with<iter_ref3&&, iter_rvalue_ref&&>);

struct different_reference_types_with_common_reference {
  using value_type = iter_ref3;
  TEST_HOST_DEVICE iter_ref3& operator*() const;
  TEST_HOST_DEVICE friend iter_rvalue_ref&& iter_move(different_reference_types_with_common_reference);
};
static_assert(check_indirectly_readable<different_reference_types_with_common_reference>());

struct iter_ref4 {
  TEST_HOST_DEVICE operator iter_rvalue_ref() const;
};
namespace cuda {
namespace std {
template <template <class> class XQual, template <class> class YQual>
struct basic_common_reference<iter_ref4, iter_rvalue_ref, XQual, YQual> {
  using type = iter_rvalue_ref;
};
template <template <class> class XQual, template <class> class YQual>
struct basic_common_reference<iter_rvalue_ref, iter_ref4, XQual, YQual> {
  using type = iter_rvalue_ref;
};

template <>
struct common_reference<iter_ref4 const&, iter_rvalue_ref&&> {};
template <>
struct common_reference<iter_rvalue_ref&&, iter_ref4 const&> {};
} // namespace std
} // namespace cuda
static_assert(cuda::std::common_reference_with<iter_ref4&&, iter_rvalue_ref&&>);
static_assert(!cuda::std::common_reference_with<iter_ref4 const&, iter_rvalue_ref&&>);

struct different_reference_types_without_common_reference_to_const {
  using value_type = iter_ref4;
  TEST_HOST_DEVICE iter_ref4& operator*() const;
  TEST_HOST_DEVICE friend iter_rvalue_ref&& iter_move(different_reference_types_without_common_reference_to_const);
};
static_assert(!check_indirectly_readable<different_reference_types_without_common_reference_to_const>());

struct non_const_deref {
  TEST_HOST_DEVICE int& operator*();
};
static_assert(!check_indirectly_readable<non_const_deref>());

struct not_referenceable {
  using value_type = void;
  TEST_HOST_DEVICE void operator*() const;
};
static_assert(!cuda::std::indirectly_readable<not_referenceable>);

struct legacy_output_iterator {
  using value_type = void;
  TEST_HOST_DEVICE legacy_output_iterator& operator*();
};

static_assert(!cuda::std::indirectly_readable<legacy_output_iterator>);

struct S {};
static_assert(!cuda::std::indirectly_readable<S>);
static_assert(!cuda::std::indirectly_readable<int S::*>);
static_assert(!cuda::std::indirectly_readable<int (S::*)()>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() noexcept>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() &>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() & noexcept>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() &&>);
static_assert(!cuda::std::indirectly_readable < int (S::*)() && noexcept >);
static_assert(!cuda::std::indirectly_readable<int (S::*)() const>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() const noexcept>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() const&>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() const & noexcept>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() const&&>);
static_assert(!cuda::std::indirectly_readable < int (S::*)() const&& noexcept >);
static_assert(!cuda::std::indirectly_readable<int (S::*)() volatile>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() volatile noexcept>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() volatile&>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() volatile & noexcept>);
static_assert(!cuda::std::indirectly_readable<int (S::*)() volatile&&>);
static_assert(!cuda::std::indirectly_readable < int (S::*)() volatile&& noexcept >);

int main(int, char**)
{
  return 0;
}
