// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_CUSTOM_TEST_LAYOUTS_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_CUSTOM_TEST_LAYOUTS_H

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

// Layout that wraps indices to test some idiosyncratic behavior
// - basically it is a layout_left where indices are first wrapped i.e. i%Wrap
// - only accepts integers as indices
// - is_always_strided and is_always_unique are false
// - is_strided and is_unique are true if all extents are smaller than Wrap
// - not default constructible
// - not extents constructible
// - not trivially copyable
// - does not check dynamic to static extent conversion in converting ctor
// - check via side-effects that mdspan::swap calls mappings swap via ADL

__host__ __device__ bool mul_overflow(size_t x, size_t y, size_t* res)
{
  *res = x * y;
  return x && ((*res / x) != y);
}

template <class T>
__host__ __device__ inline const T& Min(const T& __a, const T& __b)
{
  return __b < __a ? __b : __a;
}

struct not_extents_constructible_tag
{};

STATIC_TEST_GLOBAL_VAR int layout_wrapping_integral_swap_counter = 0;
template <size_t Wrap>
class layout_wrapping_integral
{
public:
  template <class Extents>
  class mapping;
};

template <size_t WrapArg>
template <class Extents>
class layout_wrapping_integral<WrapArg>::mapping
{
  static constexpr typename Extents::index_type Wrap = static_cast<typename Extents::index_type>(WrapArg);

public:
  using extents_type = Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_wrapping_integral<Wrap>;

private:
  template <class Extents2 = Extents, cuda::std::enable_if_t<Extents2::rank() == 0, int> = 0>
  __host__ __device__ static constexpr bool required_span_size_is_representable(const extents_type& ext)
  {
    return true;
  }

  template <class Extents2 = Extents, cuda::std::enable_if_t<Extents2::rank() != 0, int> = 0>
  __host__ __device__ static constexpr bool required_span_size_is_representable(const extents_type& ext)
  {
    index_type prod = ext.extent(0);
    for (rank_type r = 1; r < extents_type::rank(); r++)
    {
      bool overflowed = mul_overflow(prod, Min(ext.extent(r), Wrap), &prod);
      if (overflowed)
      {
        return false;
      }
    }
    return true;
  }

public:
  __host__ __device__ constexpr mapping() noexcept = delete;
  __host__ __device__ constexpr mapping(const mapping& other) noexcept
      : extents_(other.extents()){};
  template <size_t Wrap2 = Wrap, cuda::std::enable_if_t<Wrap2 == 8, int> = 0>
  __host__ __device__ constexpr mapping(extents_type&& ext) noexcept
      : extents_(ext)
  {}
  __host__ __device__ constexpr mapping(const extents_type& ext, not_extents_constructible_tag) noexcept
      : extents_(ext)
  {}

  template <class OtherExtents, class Extents2 = Extents, cuda::std::enable_if_t<Extents2::rank() != 0, int> = 0>
  __host__ __device__ static constexpr cuda::std::array<index_type, extents_type::rank_dynamic()>
  get_dyn_extents(const mapping<OtherExtents>& other) noexcept
  {
    cuda::std::array<index_type, extents_type::rank_dynamic()> dyn_extents;
    rank_type count = 0;
    for (rank_type r = 0; r != extents_type::rank(); r++)
    {
      if (extents_type::static_extent(r) == cuda::std::dynamic_extent)
      {
        dyn_extents[count++] = other.extents().extent(r);
      }
    }
    return dyn_extents;
  }
  template <class OtherExtents, class Extents2 = Extents, cuda::std::enable_if_t<Extents2::rank() == 0, int> = 0>
  __host__ __device__ static constexpr cuda::std::array<index_type, extents_type::rank_dynamic()>
  get_dyn_extents(const mapping<OtherExtents>& other) noexcept
  {
    return {};
  }

  template <
    class OtherExtents,
    cuda::std::enable_if_t<cuda::std::is_constructible<extents_type, OtherExtents>::value && (Wrap != 8), int> = 0,
    cuda::std::enable_if_t<cuda::std::is_convertible<OtherExtents, extents_type>::value, int>                  = 0>
  __host__ __device__ constexpr mapping(const mapping<OtherExtents>& other) noexcept
  {
    extents_ = extents_type(get_dyn_extents(other));
  }

  template <
    class OtherExtents,
    cuda::std::enable_if_t<cuda::std::is_constructible<extents_type, OtherExtents>::value && (Wrap != 8), int> = 0,
    cuda::std::enable_if_t<!cuda::std::is_convertible<OtherExtents, extents_type>::value, int>                 = 0>
  __host__ __device__ constexpr explicit mapping(const mapping<OtherExtents>& other) noexcept
  {
    extents_ = extents_type(get_dyn_extents(other));
  }

  template <
    class OtherExtents,
    cuda::std::enable_if_t<cuda::std::is_constructible<extents_type, OtherExtents>::value && (Wrap == 8), int> = 0,
    cuda::std::enable_if_t<cuda::std::is_convertible<OtherExtents, extents_type>::value, int>                  = 0>
  __host__ __device__ constexpr mapping(mapping<OtherExtents>&& other) noexcept
  {
    extents_ = extents_type(get_dyn_extents(other));
  }

  template <
    class OtherExtents,
    cuda::std::enable_if_t<cuda::std::is_constructible<extents_type, OtherExtents>::value && (Wrap == 8), int> = 0,
    cuda::std::enable_if_t<!cuda::std::is_convertible<OtherExtents, extents_type>::value, int>                 = 0>
  __host__ __device__ constexpr explicit mapping(mapping<OtherExtents>&& other) noexcept
  {
    extents_ = extents_type(get_dyn_extents(other));
  }

  __host__ __device__ constexpr mapping& operator=(const mapping& other) noexcept
  {
    extents_ = other.extents_;
    return *this;
  };

  __host__ __device__ constexpr const extents_type& extents() const noexcept
  {
    return extents_;
  }

  __host__ __device__ constexpr index_type required_span_size() const noexcept
  {
    index_type size = 1;
    for (size_t r = 0; r != extents_type::rank(); r++)
    {
      size *= extents_.extent(r) < Wrap ? extents_.extent(r) : Wrap;
    }
    return size;
  }

  struct rank_accumulator
  {
    __host__ __device__ constexpr rank_accumulator(const extents_type& extents) noexcept
        : extents_(extents)
    {}

    template <size_t... Pos, class... Indices>
    __host__ __device__ constexpr index_type operator()(cuda::std::index_sequence<Pos...>, Indices... idx) const noexcept
    {
      cuda::std::array<index_type, extents_type::rank()> idx_a{
        static_cast<index_type>(static_cast<index_type>(idx) % Wrap)...};
      cuda::std::array<size_t, extents_type::rank()> position = {(extents_type::rank() - 1 - Pos)...};

      index_type res = 0;
      for (size_t index : position)
      {
        res = idx_a[index] + (extents_.extent(index) < Wrap ? extents_.extent(index) : Wrap) * res;
      }
      return res;
    }

    const extents_type& extents_{};
  };

  template <
    class... Indices,
    cuda::std::enable_if_t<sizeof...(Indices) == extents_type::rank(), int>                                        = 0,
    cuda::std::enable_if_t<cuda::std::__all<cuda::std::integral<Indices>...>::value, int>                          = 0,
    cuda::std::enable_if_t<cuda::std::__all<cuda::std::is_convertible<Indices, index_type>::value...>::value, int> = 0,
    cuda::std::enable_if_t<cuda::std::__all<cuda::std::is_nothrow_constructible<index_type, Indices>::value...>::value,
                           int>                                                                                    = 0>
  __host__ __device__ constexpr index_type operator()(Indices... idx) const noexcept
  {
    return rank_accumulator{extents_}(cuda::std::make_index_sequence<sizeof...(Indices)>(), idx...);
  }

  __host__ __device__ static constexpr bool is_always_unique() noexcept
  {
    return false;
  }
  __host__ __device__ static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }
  __host__ __device__ static constexpr bool is_always_strided() noexcept
  {
    return false;
  }

  TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

  __host__ __device__ constexpr bool is_unique() const noexcept
  {
    for (rank_type r = 0; r != extents_type::rank(); r++)
    {
      if (extents_.extent(r) > Wrap)
      {
        return false;
      }
    }
    return true;
  }
  __host__ __device__ static constexpr bool is_exhaustive() noexcept
  {
    return true;
  }
  __host__ __device__ constexpr bool is_strided() const noexcept
  {
    for (rank_type r = 0; r != extents_type::rank(); r++)
    {
      if (extents_.extent(r) > Wrap)
      {
        return false;
      }
    }
    return true;
  }

  template <size_t Rank = extents_type::rank(), cuda::std::enable_if_t<(Rank > 0), int> = 0>
  __host__ __device__ constexpr index_type stride(rank_type r) const noexcept
  {
    index_type s = 1;
    for (rank_type i = extents_type::rank() - 1; i > r; i--)
    {
      s *= extents_.extent(i);
    }
    return s;
  }

  template <class OtherExtents, cuda::std::enable_if_t<OtherExtents::rank() == extents_type::rank(), int> = 0>
  __host__ __device__ friend constexpr bool operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
  {
    return lhs.extents() == rhs.extents();
  }

#if TEST_STD_VER <= 2017
  template <class OtherExtents, cuda::std::enable_if_t<OtherExtents::rank() == extents_type::rank(), int> = 0>
  __host__ __device__ friend constexpr bool operator!=(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
  {
    return lhs.extents() != rhs.extents();
  }
#endif // TEST_STD_VER <= 2017

  __host__ __device__ friend constexpr void swap(mapping& x, mapping& y) noexcept
  {
    swap(x.extents_, y.extents_);
    if (!cuda::std::__cccl_default_is_constant_evaluated())
    {
      layout_wrapping_integral_swap_counter++;
    }
  }

  __host__ __device__ static int& swap_counter()
  {
    return layout_wrapping_integral_swap_counter;
  }

private:
  extents_type extents_{};
};

template <
  class MDS,
  cuda::std::enable_if_t<cuda::std::is_same<typename MDS::layout_type, layout_wrapping_integral<4>>::value, int> = 0>
__host__ __device__ constexpr void test_swap_counter()
{
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    assert(MDS::mapping_type::swap_counter() > 0);
  }
}
template <
  class MDS,
  cuda::std::enable_if_t<!cuda::std::is_same<typename MDS::layout_type, layout_wrapping_integral<4>>::value, int> = 0>
__host__ __device__ constexpr void test_swap_counter()
{}

template <class Extents>
__host__ __device__ constexpr auto construct_mapping(cuda::std::layout_left, Extents exts)
{
  return cuda::std::layout_left::mapping<Extents>(exts);
}

template <class Extents>
__host__ __device__ constexpr auto construct_mapping(cuda::std::layout_right, Extents exts)
{
  return cuda::std::layout_right::mapping<Extents>(exts);
}

template <size_t Wraps, class Extents>
__host__ __device__ constexpr auto construct_mapping(layout_wrapping_integral<Wraps>, Extents exts)
{
  return typename layout_wrapping_integral<Wraps>::template mapping<Extents>(exts, not_extents_constructible_tag{});
}

// This layout does not check convertibility of extents for its conversion ctor
// Allows triggering mdspan's ctor static assertion on convertibility of extents
STATIC_TEST_GLOBAL_VAR int always_convertible_layout_swap_counter = 0;
class always_convertible_layout
{
public:
  template <class Extents>
  class mapping;
};

template <class Extents>
class always_convertible_layout::mapping
{
public:
  using extents_type = Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = always_convertible_layout;

private:
  template <class Extents2 = Extents, cuda::std::enable_if_t<(Extents2::rank() == 0), int> = 0>
  __host__ __device__ static constexpr bool required_span_size_is_representable(const extents_type& ext)
  {
    return true;
  }
  template <class Extents2 = Extents, cuda::std::enable_if_t<(Extents2::rank() != 0), int> = 0>
  __host__ __device__ static constexpr bool required_span_size_is_representable(const extents_type& ext)
  {
    index_type prod = ext.extent(0);
    for (rank_type r = 1; r < extents_type::rank(); r++)
    {
      bool overflowed = mul_overflow(prod, ext.extent(r), &prod);
      if (overflowed)
      {
        return false;
      }
    }
    return true;
  }

public:
  __host__ __device__ constexpr mapping() noexcept = delete;
  __host__ __device__ constexpr mapping(const mapping& other) noexcept
      : extents_(other.extents_)
      , offset_(other.offset_)
      , scaling_(other.scaling_)
  {}
  __host__ __device__ constexpr mapping(const extents_type& ext) noexcept
      : extents_(ext)
      , offset_(0)
      , scaling_(1){};
  __host__ __device__ constexpr mapping(const extents_type& ext, index_type offset) noexcept
      : extents_(ext)
      , offset_(offset)
      , scaling_(1){};
  __host__ __device__ constexpr mapping(const extents_type& ext, index_type offset, index_type scaling) noexcept
      : extents_(ext)
      , offset_(offset)
      , scaling_(scaling){};

  template <class OtherExtents, cuda::std::enable_if_t<(extents_type::rank() == OtherExtents::rank()), int> = 0>
  __host__ __device__ constexpr mapping(const mapping<OtherExtents>& other) noexcept
  {
    cuda::std::array<index_type, extents_type::rank_dynamic()> dyn_extents;
    rank_type count = 0;
    for (rank_type r = 0; r != extents_type::rank(); r++)
    {
      if (extents_type::static_extent(r) == cuda::std::dynamic_extent)
      {
        dyn_extents[count++] = other.extents().extent(r);
      }
    }
    extents_ = extents_type(dyn_extents);
    offset_  = other.offset_;
    scaling_ = other.scaling_;
  }
  template <class OtherExtents, cuda::std::enable_if_t<(extents_type::rank() != OtherExtents::rank()), int> = 0>
  __host__ __device__ constexpr mapping(const mapping<OtherExtents>& other) noexcept
  {
    extents_ = extents_type();
    offset_  = other.offset_;
    scaling_ = other.scaling_;
  }

  __host__ __device__ constexpr mapping& operator=(const mapping& other) noexcept
  {
    extents_ = other.extents_;
    offset_  = other.offset_;
    scaling_ = other.scaling_;
    return *this;
  };

  __host__ __device__ constexpr const extents_type& extents() const noexcept
  {
    return extents_;
  }

  __host__ __device__ static constexpr const index_type& Max(const index_type& __a, const index_type& __b) noexcept
  {
    return __a > __b ? __a : __b;
  }

  __host__ __device__ constexpr index_type required_span_size() const noexcept
  {
    index_type size = 1;
    for (size_t r = 0; r != extents_type::rank(); r++)
    {
      size *= extents_.extent(r);
    }
    return Max(size * scaling_ + offset_, offset_);
  }

  struct rank_accumulator
  {
    __host__ __device__ constexpr rank_accumulator(const extents_type& extents) noexcept
        : extents_(extents)
    {}

    template <size_t... Pos, class... Indices>
    __host__ __device__ constexpr index_type operator()(cuda::std::index_sequence<Pos...>, Indices... idx) const noexcept
    {
      cuda::std::array<index_type, extents_type::rank()> idx_a{
        static_cast<index_type>(static_cast<index_type>(idx))...};
      cuda::std::array<size_t, extents_type::rank()> position = {(extents_type::rank() - 1 - Pos)...};

      index_type res = 0;
      for (size_t index : position)
      {
        res = idx_a[index] + extents_.extent(index) * res;
      }
      return res;
    }

    const extents_type& extents_{};
  };

  template <
    class... Indices,
    cuda::std::enable_if_t<sizeof...(Indices) == extents_type::rank(), int>                                        = 0,
    cuda::std::enable_if_t<cuda::std::__all<cuda::std::integral<Indices>...>::value, int>                          = 0,
    cuda::std::enable_if_t<cuda::std::__all<cuda::std::is_convertible<Indices, index_type>::value...>::value, int> = 0,
    cuda::std::enable_if_t<cuda::std::__all<cuda::std::is_nothrow_constructible<index_type, Indices>::value...>::value,
                           int>                                                                                    = 0>
  __host__ __device__ constexpr index_type operator()(Indices... idx) const noexcept
  {
    return offset_
         + scaling_ * rank_accumulator{extents_}(cuda::std::make_index_sequence<sizeof...(Indices)>(), idx...);
  }

  __host__ __device__ static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  __host__ __device__ static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }
  __host__ __device__ static constexpr bool is_always_strided() noexcept
  {
    return true;
  }

  __host__ __device__ static constexpr bool is_unique() noexcept
  {
    return true;
  }
  __host__ __device__ static constexpr bool is_exhaustive() noexcept
  {
    return true;
  }
  __host__ __device__ static constexpr bool is_strided() noexcept
  {
    return true;
  }

  template <size_t Rank = extents_type::rank(), cuda::std::enable_if_t<(Rank > 0), int> = 0>
  __host__ __device__ constexpr index_type stride(rank_type r) const noexcept
  {
    index_type s = 1;
    for (rank_type i = 0; i < r; i++)
    {
      s *= extents_.extent(i);
    }
    return s * scaling_;
  }

  template <class OtherExtents>
  __host__ __device__ friend constexpr auto operator==(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
    -> cuda::std::enable_if_t<OtherExtents::rank() == extents_type::rank(), bool>
  {
    return lhs.extents() == rhs.extents() && lhs.offset_ == rhs.offset && lhs.scaling_ == rhs.scaling_;
  }

#if TEST_STD_VER < 2020
  template <class OtherExtents>
  __host__ __device__ friend constexpr auto operator!=(const mapping& lhs, const mapping<OtherExtents>& rhs) noexcept
    -> cuda::std::enable_if_t<OtherExtents::rank() != extents_type::rank(), bool>
  {
    return !(lhs == rhs);
  }
#endif // TEST_STD_VER < 2020

  __host__ __device__ friend constexpr void swap(mapping& x, mapping& y) noexcept
  {
    swap(x.extents_, y.extents_);
    if (!cuda::std::__cccl_default_is_constant_evaluated())
    {
      always_convertible_layout_swap_counter++;
    }
  }

private:
  template <class>
  friend class mapping;

  extents_type extents_{};
  index_type offset_{};
  index_type scaling_{};
};
#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_CUSTOM_TEST_LAYOUTS_H
