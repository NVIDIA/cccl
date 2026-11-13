// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/type_traits>

#include <cstdint>

#include "catch2_test_numeric_helpers.cuh"
#include "cub_test_macros.h"

template <class WrappedT, class UnwrappedT>
struct unwrap_test_types
{
  using wrapped_t   = WrappedT;
  using unwrapped_t = UnwrappedT;
};

using unwrap_types =
  c2h::type_list<unwrap_test_types<std::int32_t, std::int32_t>
#if TEST_HALF_T()
                 ,
                 unwrap_test_types<half_t, __half>
#endif // TEST_HALF_T()
#if TEST_BF_T()
                 ,
                 unwrap_test_types<bfloat16_t, __nv_bfloat16>
#endif // TEST_BF_T()
                 >;

CUB_TEST("unwrap_it preserves pointer addresses and constness", "[test-helpers][numeric]", CUB_SMALL, unwrap_types)
{
  using types       = c2h::get<0, TestType>;
  using wrapped_t   = typename types::wrapped_t;
  using unwrapped_t = typename types::unwrapped_t;

  wrapped_t value{1};
  wrapped_t* ptr             = &value;
  const wrapped_t* const_ptr = &value;

  STATIC_REQUIRE(cuda::std::is_same_v<decltype(unwrap_it(ptr)), unwrapped_t*>);
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(unwrap_it(const_ptr)), const unwrapped_t*>);

  CHECK(static_cast<void*>(unwrap_it(ptr)) == static_cast<void*>(ptr));
  CHECK(static_cast<const void*>(unwrap_it(const_ptr)) == static_cast<const void*>(const_ptr));
  CHECK(unwrap_it(static_cast<wrapped_t*>(nullptr)) == nullptr);
  CHECK(unwrap_it(static_cast<const wrapped_t*>(nullptr)) == nullptr);
}

CUB_TEST("unwrap_it preserves constant iterator values and offset types",
         "[test-helpers][numeric]",
         CUB_SMALL,
         unwrap_types,
         c2h::type_list<std::int32_t, std::int64_t>)
{
  using types       = c2h::get<0, TestType>;
  using wrapped_t   = typename types::wrapped_t;
  using unwrapped_t = typename types::unwrapped_t;
  using offset_t    = c2h::get<1, TestType>;

  const wrapped_t value{42};
  const cuda::constant_iterator<wrapped_t, offset_t> it{value};
  auto unwrapped = unwrap_it(it);

  STATIC_REQUIRE(cuda::std::is_same_v<decltype(unwrapped), cuda::constant_iterator<unwrapped_t, offset_t>>);
  CHECK(wrapped_t{*unwrapped} == value);

  const offset_t offset{7};
  CHECK(wrapped_t{unwrapped[offset]} == value);
}
