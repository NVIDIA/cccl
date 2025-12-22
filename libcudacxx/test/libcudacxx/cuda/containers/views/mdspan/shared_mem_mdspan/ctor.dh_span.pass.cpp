//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <mdspan>

// template<class OtherIndexType, size_t N>
//   constexpr explicit(N != rank_dynamic())
//     mdspan(data_handle_type p, span<OtherIndexType, N> exts);
//
// Constraints:
//   - is_convertible_v<const OtherIndexType&, index_type> is true,
//   - (is_nothrow_constructible<index_type, const OtherIndexType&> && ...) is true,
//   - N == rank() || N == rank_dynamic() is true,
//   - is_constructible_v<mapping_type, extents_type> is true, and
//   - is_default_constructible_v<accessor_type> is true.
//
// Preconditions: [0, map_.required_span_size()) is an accessible range of p and acc_
//                for the values of map_ and acc_ after the invocation of this constructor.
//
// Effects:
//   - Direct-non-list-initializes ptr_ with cuda::std::move(p),
//   - direct-non-list-initializes map_ with extents_type(exts), and
//   - value-initializes acc_.
#define _CCCL_DISABLE_MDSPAN_ACCESSOR_DETECT_INVALIDITY

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "../ConvertibleToIntegral.h"
#include "../CustomTestAccessors.h"
#include "../CustomTestLayouts.h"
#include "../MinimalElementType.h"
#include "test_macros.h"

#if _CCCL_CUDA_COMPILER(NVRTC)
#  define _CCCL_NVRTC_CONCEPT_DECORATOR() __device__
#else
#  define _CCCL_NVRTC_CONCEPT_DECORATOR()
#endif // _CCCL_CUDA_COMPILER(NVRTC)

template <class Extents, size_t... Idxs>
__device__ constexpr auto array_from_extents(const Extents& exts, cuda::std::index_sequence<Idxs...>)
{
  return cuda::std::array<typename Extents::index_type, Extents::rank()>{exts.extent(Idxs)...};
}

template <class MDS>
__device__ void check_implicit_construction(MDS);

template <class MDS, class Exts>
_CCCL_NVRTC_CONCEPT_DECORATOR()
constexpr bool check_implicit_construction_impl(...)
{
  return false;
}
template <class MDS, class Exts>
_CCCL_NVRTC_CONCEPT_DECORATOR()
constexpr auto check_implicit_construction_impl(int)
  -> decltype(check_implicit_construction<MDS>({cuda::std::declval<typename MDS::data_handle_type>(),
                                                cuda::std::declval<const Exts&>()}),
              true)
{
  return true;
}

template <class MDS, class Exts>
_CCCL_CONCEPT check_mdspan_ctor_implicit = check_implicit_construction_impl<MDS, Exts>(0);

template <class H, class M, class A, size_t N>
__device__ constexpr void
test_mdspan_ctor_span(const H& handle, const M& map, const A&, cuda::std::span<const typename M::index_type, N> exts)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    move_counted_handle<typename MDS::element_type>::move_counter() = 0;
  }
  MDS m(handle, exts);
  test_move_counter<MDS, H>();

  static_assert(!noexcept(MDS(handle, exts)));
  static_assert(check_mdspan_ctor_implicit<MDS, decltype(exts)> == (N == MDS::rank_dynamic()));

  assert(m.extents() == map.extents());
  test_equality_handle(m, handle);
  test_equality_mapping(m, map);
  test_equality_accessor(m, A{});
}

template <class MDS, class Extents, cuda::std::enable_if_t<(MDS::rank() > 0), int> = 0>
__device__ constexpr cuda::std::array<typename MDS::index_type, MDS::rank_dynamic()>
get_exts_dynamic(const cuda::std::array<typename Extents::index_type, Extents::rank()>& exts)
{
  cuda::std::array<typename MDS::index_type, MDS::rank_dynamic()> exts_dynamic{};
  size_t r_dyn = 0;
  for (size_t r = 0; r < MDS::rank(); r++)
  {
    if (MDS::static_extent(r) == cuda::std::dynamic_extent)
    {
      exts_dynamic[r_dyn++] = exts[r];
    }
  }
  return exts_dynamic;
}
template <class MDS, class Extents, cuda::std::enable_if_t<(MDS::rank() == 0), int> = 0>
__device__ constexpr cuda::std::array<typename MDS::index_type, MDS::rank_dynamic()>
get_exts_dynamic(const cuda::std::array<typename Extents::index_type, Extents::rank()>&)
{
  return cuda::std::array<typename MDS::index_type, MDS::rank_dynamic()>{};
}

template <bool mec, bool ac, class H, class M, class A, cuda::std::enable_if_t<mec && ac, int> = 0>
__device__ constexpr void test_mdspan_ctor(const H& handle, const M& map, const A& acc)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  static_assert(mec == cuda::std::is_constructible_v<M, typename M::extents_type>);
  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  // test from all extents
  const auto exts = array_from_extents(map.extents(), cuda::std::make_index_sequence<MDS::rank()>());
  test_mdspan_ctor_span(
    handle, map, acc, cuda::std::span<const typename M::extents_type::index_type, M::extents_type::rank()>(exts));

  // test from dynamic extents
  const auto exts_dynamic = get_exts_dynamic<MDS, typename M::extents_type>(exts);
  test_mdspan_ctor_span(
    handle, map, acc, cuda::std::span<const typename MDS::index_type, MDS::rank_dynamic()>(exts_dynamic));
}
template <bool mec, bool ac, class H, class M, class A, cuda::std::enable_if_t<!(mec && ac), int> = 0>
__device__ constexpr void test_mdspan_ctor(const H& handle, const M& map, const A& acc)
{
  using MDS =
    cuda::shared_memory_mdspan<typename A::element_type, typename M::extents_type, typename M::layout_type, A>;
  static_assert(mec == cuda::std::is_constructible_v<M, typename M::extents_type>);
  static_assert(ac == cuda::std::is_default_constructible_v<A>);
  static_assert(!cuda::std::is_constructible_v<MDS, const H&, cuda::std::span<typename MDS::index_type, MDS::rank()>>);
}

template <bool mec, bool ac, class H, class L, class A>
__device__ constexpr void mixin_extents(const H& handle, const L& layout, const A& acc)
{
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<int>()), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D>(7)), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<unsigned, 7>()), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<size_t, D, 4, D>(2, 3)), acc);
  test_mdspan_ctor<mec, ac>(handle, construct_mapping(layout, cuda::std::extents<char, D, 7, D>(0, 3)), acc);
  test_mdspan_ctor<mec, ac>(
    handle, construct_mapping(layout, cuda::std::extents<int64_t, D, 7, D, 4, D, D>(1, 2, 3, 2)), acc);
}

template <bool ac, class H, class A>
__device__ constexpr void mixin_layout(const H& handle, const A& acc)
{
  mixin_extents<true, ac>(handle, cuda::std::layout_left(), acc);
  mixin_extents<true, ac>(handle, cuda::std::layout_right(), acc);

  // Sanity check that this layouts mapping is constructible from extents (via its move constructor)
  static_assert(
    cuda::std::is_constructible_v<typename layout_wrapping_integral<8>::template mapping<cuda::std::extents<int>>,
                                  cuda::std::extents<int>>);
  static_assert(
    !cuda::std::is_constructible_v<typename layout_wrapping_integral<8>::template mapping<cuda::std::extents<int>>,
                                   const cuda::std::extents<int>&>);
  mixin_extents<true, ac>(handle, layout_wrapping_integral<8>(), acc);
  // Sanity check that this layouts mapping is not constructible from extents
  static_assert(
    !cuda::std::is_constructible_v<typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>,
                                   cuda::std::extents<int>>);
  static_assert(
    !cuda::std::is_constructible_v<typename layout_wrapping_integral<4>::template mapping<cuda::std::extents<int>>,
                                   const cuda::std::extents<int>&>);
  mixin_extents<false, ac>(handle, layout_wrapping_integral<4>(), acc);
}

template <class T, cuda::std::enable_if_t<cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ constexpr void mixin_accessor()
{
  cuda::std::array<T, 1024> elements{42};
  mixin_layout<true>(elements.data(), cuda::std::default_accessor<T>());
}

template <class T, cuda::std::enable_if_t<!cuda::std::is_default_constructible_v<T>, int> = 0>
__device__ void mixin_accessor()
{
  ElementPool<T, 1024> elements;
  mixin_layout<true>(elements.get_ptr(), cuda::std::default_accessor<T>());
}

__device__ void test()
{
  mixin_accessor<int>();
  mixin_accessor<const int>();
  mixin_accessor<double>();
  mixin_accessor<const double>();

  // test non-constructibility from wrong span type
  [[maybe_unused]] constexpr size_t D = cuda::std::dynamic_extent;
  using mds_t                         = cuda::shared_memory_mdspan<float, cuda::std::extents<unsigned, 3, D, D>>;
  // sanity check
  static_assert(cuda::std::is_constructible_v<mds_t, float*, cuda::std::span<int, 3>>);
  static_assert(cuda::std::is_constructible_v<mds_t, float*, cuda::std::span<int, 2>>);
  // wrong size
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, cuda::std::span<int, 1>>);
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, cuda::std::span<int, 4>>);
  // not convertible to index_type
  static_assert(cuda::std::is_convertible_v<const IntType&, int>);
  static_assert(!cuda::std::is_convertible_v<const IntType&, unsigned>);
  static_assert(!cuda::std::is_constructible_v<mds_t, float*, cuda::std::span<IntType, 2>>);

  // index_type is not nothrow constructible
  using mds_uchar_t = cuda::shared_memory_mdspan<float, cuda::std::extents<unsigned char, 3, D, D>>;
  static_assert(cuda::std::is_convertible_v<IntType, unsigned char>);
  static_assert(cuda::std::is_convertible_v<const IntType&, unsigned char>);
  static_assert(!cuda::std::is_nothrow_constructible_v<unsigned char, const IntType&>);
  static_assert(!cuda::std::is_constructible_v<mds_uchar_t, float*, cuda::std::span<IntType, 2>>);

  // convertible from non-const to index_type but not  from const
  using mds_int_t = cuda::shared_memory_mdspan<float, cuda::std::extents<int, 3, D, D>>;
  static_assert(cuda::std::is_convertible_v<IntTypeNC, int>);
  static_assert(!cuda::std::is_convertible_v<const IntTypeNC&, int>);
  static_assert(cuda::std::is_nothrow_constructible_v<int, IntTypeNC>);
  static_assert(!cuda::std::is_constructible_v<mds_int_t, float*, cuda::std::span<IntTypeNC, 2>>);

  // can't test a combo where cuda::std::is_nothrow_constructible_v<int, const IntTypeNC&> is true,
  // but cuda::std::is_convertible_v<const IntType&, int> is false

  // test non-constructibility from wrong handle_type
  static_assert(!cuda::std::is_constructible_v<mds_t, const float*, cuda::std::span<int, 2>>);
}

__device__ void test_evil()
{
  mixin_accessor<MinimalElementType>();
  mixin_accessor<const MinimalElementType>();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test(); test_evil();))
  return 0;
}
