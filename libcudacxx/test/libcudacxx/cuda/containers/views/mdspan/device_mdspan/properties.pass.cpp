//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right,
//            class AccessorPolicy = default_accessor<ElementType>>
//   class mdspan {
//   public:
//     static constexpr rank_type rank() noexcept { return extents_type::rank(); }
//     static constexpr rank_type rank_dynamic() noexcept { return extents_type::rank_dynamic(); }
//     static constexpr size_t static_extent(rank_type r) noexcept
//       { return extents_type::static_extent(r); }
//     constexpr index_type extent(rank_type r) const noexcept { return extents().extent(r); }
//
//     constexpr size_type size() const noexcept;
//     [[nodiscard]] constexpr bool empty() const noexcept;
//
//
//     constexpr const extents_type& extents() const noexcept { return map_.extents(); }
//     constexpr const data_handle_type& data_handle() const noexcept { return ptr_; }
//     constexpr const mapping_type& mapping() const noexcept { return map_; }
//     constexpr const accessor_type& accessor() const noexcept { return acc_; }
//     static constexpr bool is_always_unique()
//       { return mapping_type::is_always_unique(); }
//     static constexpr bool is_always_exhaustive()
//       { return mapping_type::is_always_exhaustive(); }
//     static constexpr bool is_always_strided()
//       { return mapping_type::is_always_strided(); }
//
//     constexpr bool is_unique() const
//       { return map_.is_unique(); }
//     constexpr bool is_exhaustive() const
//       { return map_.is_exhaustive(); }
//     constexpr bool is_strided() const
//       { return map_.is_strided(); }
//     constexpr index_type stride(rank_type r) const
//       { return map_.stride(r); }
//   };
//
// Each specialization MDS of mdspan models copyable and
//    - is_nothrow_move_constructible_v<MDS> is true,
//    - is_nothrow_move_assignable_v<MDS> is true, and
//    - is_nothrow_swappable_v<MDS> is true.
// A specialization of mdspan is a trivially copyable type if its accessor_type, mapping_type, and data_handle_type are
// trivially copyable types.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

template <class MDS, cuda::std::enable_if_t<(MDS::rank() > 0), int> = 0>
__host__ __device__ constexpr void test_mdspan_size(const MDS& m)
{
  typename MDS::size_type size = 1;
  for (typename MDS::rank_type r = 0; r < MDS::rank(); r++)
  {
    static_assert(cuda::std::is_same_v<decltype(MDS::static_extent(r)), size_t>);
    static_assert(noexcept(MDS::static_extent(r)));
    assert(MDS::static_extent(r) == MDS::extents_type::static_extent(r));
    static_assert(cuda::std::is_same_v<decltype(m.extent(r)), typename MDS::index_type>);
    static_assert(noexcept(m.extent(r)));
    assert(m.extent(r) == m.extents().extent(r));
    size *= m.extent(r);
  }
  assert(m.size() == size);
}

template <class MDS, cuda::std::enable_if_t<(MDS::rank() == 0), int> = 0>
__host__ __device__ constexpr void test_mdspan_size(const MDS& m)
{
  assert(m.size() == 1);
}

template <class MDS, class M, cuda::std::enable_if_t<(MDS::rank() > 0), int> = 0>
__host__ __device__ constexpr void test_mdspan_stride(const MDS& m, const M& map)
{
  if (m.is_strided())
  {
    for (typename MDS::rank_type r = 0; r < MDS::rank(); r++)
    {
      static_assert(cuda::std::is_same_v<decltype(m.stride(r)), typename MDS::index_type>);
      assert(!noexcept(m.stride(r)));
      assert(m.stride(r) == map.stride(r));
    }
  }
}

template <class MDS, class M, cuda::std::enable_if_t<(MDS::rank() == 0), int> = 0>
__host__ __device__ constexpr void test_mdspan_stride(const MDS&, const M&)
{}

template <class H, class M, class A>
__host__ __device__ constexpr void test_mdspan_types(const H& handle, const M& map, const A& acc)
{
  using MDS = cuda::device_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  MDS m(handle, map, acc);

  // =====================================
  // Traits for every mdspan
  // =====================================
  static_assert(cuda::std::copyable<MDS>, "");
  static_assert(cuda::std::is_nothrow_move_constructible<MDS>::value, "");
  static_assert(cuda::std::is_nothrow_move_assignable<MDS>::value, "");
  static_assert(cuda::std::is_nothrow_swappable<MDS>::value, "");

  // =====================================
  // Invariants coming from data handle
  // =====================================
  // data_handle()
  static_assert(cuda::std::is_same_v<decltype(m.data_handle()), const H&>);
  static_assert(noexcept(m.data_handle()));
  test_equality_handle(m, handle);

  // =====================================
  // Invariants coming from extents
  // =====================================

  // extents()
  static_assert(cuda::std::is_same_v<decltype(m.extents()), const typename MDS::extents_type&>);
  static_assert(noexcept(m.extents()));
  assert(m.extents() == map.extents());

  // rank()
  static_assert(cuda::std::is_same_v<decltype(m.rank()), typename MDS::rank_type>);
  static_assert(noexcept(m.rank()));
  assert(MDS::rank() == MDS::extents_type::rank());

  // rank_dynamic()
  static_assert(cuda::std::is_same_v<decltype(m.rank_dynamic()), typename MDS::rank_type>);
  static_assert(noexcept(m.rank_dynamic()));
  assert(MDS::rank_dynamic() == MDS::extents_type::rank_dynamic());

  // extent(r), static_extent(r), size()
  test_mdspan_size(m);
  static_assert(cuda::std::is_same_v<decltype(m.size()), typename MDS::size_type>);
  static_assert(noexcept(m.size()));

  // empty()
  static_assert(cuda::std::is_same_v<decltype(m.empty()), bool>);
  static_assert(noexcept(m.empty()));
  assert(m.empty() == (m.size() == 0));

  // =====================================
  // Invariants coming from mapping
  // =====================================

  // mapping()
  static_assert(cuda::std::is_same_v<decltype(m.mapping()), const M&>);
  static_assert(noexcept(m.mapping()));

  // is_[always_]unique/exhaustive/strided()
  static_assert(cuda::std::is_same_v<decltype(MDS::is_always_unique()), bool>);
  static_assert(cuda::std::is_same_v<decltype(MDS::is_always_exhaustive()), bool>);
  static_assert(cuda::std::is_same_v<decltype(MDS::is_always_strided()), bool>);
  static_assert(cuda::std::is_same_v<decltype(m.is_unique()), bool>);
  static_assert(cuda::std::is_same_v<decltype(m.is_exhaustive()), bool>);
  static_assert(cuda::std::is_same_v<decltype(m.is_strided()), bool>);
  assert(noexcept(MDS::is_always_unique() == noexcept(M::is_always_unique())));
  assert(noexcept(MDS::is_always_exhaustive()) == noexcept(M::is_always_exhaustive()));
  assert(noexcept(MDS::is_always_strided()) == noexcept(M::is_always_strided()));
  assert(noexcept(m.is_unique()) == noexcept(m.is_always_unique()));
  assert(noexcept(m.is_exhaustive()) == noexcept(m.is_exhaustive()));
  assert(noexcept(m.is_strided()) == noexcept(m.is_strided()));
  assert(MDS::is_always_unique() == M::is_always_unique());
  assert(MDS::is_always_exhaustive() == M::is_always_exhaustive());
  assert(MDS::is_always_strided() == M::is_always_strided());
  assert(m.is_unique() == map.is_unique());
  assert(m.is_exhaustive() == map.is_exhaustive());
  assert(m.is_strided() == map.is_strided());

  // stride(r)
  test_mdspan_stride(m, map);

  // =====================================
  // Invariants coming from accessor
  // =====================================

  // accessor()
  static_assert(cuda::std::is_same_v<decltype(m.accessor()), const cuda::device_accessor<A>&>);
  static_assert(noexcept(m.accessor()));
}

template <class H, class L, class A>
__host__ __device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_types(handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <class H, class A>
__host__ __device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents(handle, cuda::std::layout_left(), acc);
  mixin_extents(handle, cuda::std::layout_right(), acc);
  mixin_extents(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout(elements.data(), cuda::std::default_accessor<T>());
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible<T>::value, int> = 0>
__host__ __device__ TEST_CONSTEXPR_CXX20 void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout(elements.get_ptr(), cuda::std::default_accessor<T>());
}

__host__ __device__ constexpr bool test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
  return true;
}

int main(int, char**)
{
  test();
  // static_assert(test(), "");

  test_evil();
#if TEST_STD_VER >= 2020
  static_assert(test_evil(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
