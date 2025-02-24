//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_COMMON_HELPERS_TYPE_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_COMMON_HELPERS_TYPE_H

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class MDS, class H, cuda::std::enable_if_t<cuda::std::equality_comparable<H>, int> = 0>
__host__ __device__ constexpr void test_equality_handle(const MDS& m, const H& handle)
{
  assert(m.data_handle() == handle);
}
template <class MDS, class H, cuda::std::enable_if_t<!cuda::std::equality_comparable<H>, int> = 0>
__host__ __device__ constexpr void test_equality_handle(const MDS&, const H&)
{}

template <class MDS, class M, cuda::std::enable_if_t<cuda::std::equality_comparable<M>, int> = 0>
__host__ __device__ constexpr void test_equality_mapping(const MDS& m, const M& map)
{
  assert(m.mapping() == map);
}
template <class MDS, class M, cuda::std::enable_if_t<!cuda::std::equality_comparable<M>, int> = 0>
__host__ __device__ constexpr void test_equality_mapping(const MDS&, const M&)
{}

template <class MDS, class A, cuda::std::enable_if_t<cuda::std::equality_comparable<A>, int> = 0>
__host__ __device__ constexpr void test_equality_accessor(const MDS& m, const A& acc)
{
  assert(m.accessor() == acc);
}
template <class MDS, class A, cuda::std::enable_if_t<!cuda::std::equality_comparable<A>, int> = 0>
__host__ __device__ constexpr void test_equality_accessor(const MDS&, const A&)
{}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            cuda::std::equality_comparable_with<typename ToMDS::data_handle_type, typename FromMDS::data_handle_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_handle(const ToMDS& to_mds, const FromMDS& from_mds)
{
  assert(to_mds.data_handle() == from_mds.data_handle());
}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            !cuda::std::equality_comparable_with<typename ToMDS::data_handle_type, typename FromMDS::data_handle_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_handle(const ToMDS&, const FromMDS&)
{}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            cuda::std::equality_comparable_with<typename ToMDS::mapping_type, typename FromMDS::mapping_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_mapping(const ToMDS& to_mds, const FromMDS& from_mds)
{
  assert(to_mds.mapping() == from_mds.mapping());
}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            !cuda::std::equality_comparable_with<typename ToMDS::mapping_type, typename FromMDS::mapping_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_mapping(const ToMDS&, const FromMDS&)
{}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            cuda::std::equality_comparable_with<typename ToMDS::accessor_type, typename FromMDS::accessor_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_accessor(const ToMDS& to_mds, const FromMDS& from_mds)
{
  assert(to_mds.accessor() == from_mds.accessor());
}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            !cuda::std::equality_comparable_with<typename ToMDS::accessor_type, typename FromMDS::accessor_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_accessor(const ToMDS&, const FromMDS&)
{}

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_COMMON_HELPERS_TYPE_H
