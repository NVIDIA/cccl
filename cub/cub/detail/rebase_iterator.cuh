// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

//! @file
//! Prototype of extracting the raw pointers from a (fancy) iterator and rebasing the iterator onto other storage.
//!
//! An iterator is viewed as a tree: adaptors like transform, zip or permutation iterators are inner nodes, and the
//! iterators they wrap are their children. Contiguous iterators are leaves that hold a raw pointer. All other iterators
//! (counting, constant, strided, unknown iterators, ...) are opaque leaves, which are never looked into.
//!
//! `extract_raw_pointers(it)` walks the tree and returns a tuple of the raw pointers of all contiguous leaves, in
//! depth-first order. A kernel can then load the elements behind these pointers in a fast way, e.g., vectorized into
//! registers or via bulk copies into shared memory.
//!
//! `rebase_iterator(it, ptrs)` rebuilds `it` with the same type, but with each contiguous leaf replaced by the
//! corresponding pointer in `ptrs`, typically pointing to the loaded elements. Functors and opaque leaves are copied
//! from `it` unchanged. If `ptrs[j]` points to a copy of the elements `[extract_raw_pointers(it)[j], ... + n)`, then
//! `rebase_iterator(it, ptrs)[k] == it[k]` for all `k < n`. Opaque leaves are thus correctly offset, as long as `it`
//! was advanced to the position of the loaded elements before extracting and rebasing.
//!
//! Output iterators are handled the same way: writing through the rebased iterator writes to the storage behind `ptrs`,
//! which a kernel can then store to the original pointers.
//!
//! Reverse iterators are kept opaque for now. Rebasing them is more involved, because they access the elements before
//! their base: the pointer to load from is `base() - n` for `n` loaded elements, and the rebased iterator's base must
//! point to the end of the loaded copy.
//!
//! Functors are invoked on references to the copies instead of the original elements. This is only correct if they do
//! not depend on the address of their arguments, see also `cuda::proclaims_copyable_arguments`.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/zip_iterator.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__iterator/permutation_iterator.h>
#include <cuda/__iterator/zip_iterator.h>
#include <cuda/__iterator/zip_transform_iterator.h>
#include <cuda/std/__iterator/counted_iterator.h>
#include <cuda/std/__iterator/move_iterator.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN
namespace detail
{
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace rebase
{
template <typename It, typename = void>
inline constexpr bool has_base = false;

template <typename It>
inline constexpr bool has_base<It, ::cuda::std::void_t<decltype(::cuda::std::declval<const It&>().base())>> = true;

template <typename It>
using base_t = ::cuda::std::remove_cvref_t<decltype(::cuda::std::declval<const It&>().base())>;

// TODO(bgruber): The way to unwrap and rebuild iterators should eventually be a customization point, e.g. a
// cuda::rebase_traits<It> (similar to cuda::std::pointer_traits), which users can specialize for their own iterators.
// It would default to the logic below (raw pointers, a __rebase member function, a constructor from a pointer, or
// rebuilding from base()) and also describe how adaptors decompose into child iterators (base() and __rebase for
// single-child adaptors), replacing the node specializations below.

// Iterators can provide a member function `It __rebase(Base) const`, which returns a copy of the iterator with its base
// replaced, retaining all other state (e.g. bounds or functors). For contiguous iterators, Base is the raw pointer.
// This is needed if It cannot be constructed from a raw pointer. For single-child adaptors, Base is the type of base().
template <typename It, typename Ptr, typename = void>
inline constexpr bool has_rebase_member = false;

template <typename It, typename Ptr>
inline constexpr bool has_rebase_member<
  It,
  Ptr,
  ::cuda::std::void_t<decltype(::cuda::std::declval<const It&>().__rebase(::cuda::std::declval<Ptr>()))>> = true;

// Whether a contiguous iterator of type It can be rebuilt from a raw pointer of type Ptr. In order of preference, this
// is done by its __rebase member function, by constructing It from Ptr, or by constructing It from its (recursively
// rebuilt) base, e.g. thrust::detail::normal_iterator<thrust::device_ptr<T>>. Every level must be contiguous, since
// e.g. a reverse iterator also holds a pointer, but does not traverse the storage in the order of the pointer.
template <typename It, typename Ptr>
_CCCL_HOST_DEVICE constexpr bool can_rebuild_contiguous()
{
  if constexpr (!THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>)
  {
    return false;
  }
  else if constexpr (::cuda::std::is_pointer_v<It>)
  {
    return ::cuda::std::is_convertible_v<Ptr, It>;
  }
  else if constexpr (has_rebase_member<It, Ptr> || ::cuda::std::is_constructible_v<It, Ptr>)
  {
    return true;
  }
  else if constexpr (has_base<It>)
  {
    if constexpr (::cuda::std::is_constructible_v<It, base_t<It>>)
    {
      return can_rebuild_contiguous<base_t<It>, Ptr>();
    }
    else
    {
      return false;
    }
  }
  else
  {
    return false;
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <typename It, typename Ptr>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE It rebuild_contiguous(const It& it, Ptr ptr)
{
  if constexpr (::cuda::std::is_pointer_v<It>)
  {
    return ptr;
  }
  else if constexpr (has_rebase_member<It, Ptr>)
  {
    return it.__rebase(ptr);
  }
  else if constexpr (::cuda::std::is_constructible_v<It, Ptr>)
  {
    return It(ptr);
  }
  else
  {
    return It(rebase::rebuild_contiguous(it.base(), ptr));
  }
}

// Whether the raw pointer of a contiguous iterator of type It can be obtained, either by cuda::std::to_address or from
// its (recursively unwrapped) base, e.g. for thrust::detail::tagged_iterator, which does not support to_address. Every
// level must be contiguous, since e.g. to_address also works on a reverse iterator.
template <typename It>
_CCCL_HOST_DEVICE constexpr bool can_unwrap_contiguous()
{
  if constexpr (!THRUST_NS_QUALIFIER::is_contiguous_iterator_v<It>)
  {
    return false;
  }
  else if constexpr (::cuda::std::is_pointer_v<It> || ::cuda::std::__can_to_address<It>)
  {
    return true;
  }
  else if constexpr (has_base<It>)
  {
    return can_unwrap_contiguous<base_t<It>>();
  }
  else
  {
    return false;
  }
}

template <typename It>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto unwrap_contiguous(const It& it)
{
  if constexpr (::cuda::std::is_pointer_v<It>)
  {
    return it;
  }
  else if constexpr (::cuda::std::__can_to_address<It>)
  {
    return ::cuda::std::to_address(it);
  }
  else
  {
    return rebase::unwrap_contiguous(it.base());
  }
}

template <typename It>
_CCCL_HOST_DEVICE constexpr bool is_rebasable_leaf()
{
  if constexpr (can_unwrap_contiguous<It>())
  {
    return can_rebuild_contiguous<It, decltype(rebase::unwrap_contiguous(::cuda::std::declval<const It&>()))>();
  }
  else
  {
    return false;
  }
}

// Describes how an iterator of type It decomposes into raw pointers. Specialized below for iterator adaptors. The
// primary template handles leaves: contiguous iterators that can be rebuilt from their raw pointer contribute that
// pointer, all other iterators are opaque and contribute nothing.
template <typename It, typename = void>
struct node
{
  static constexpr bool contiguous = is_rebasable_leaf<It>();

  static constexpr int num_raw_pointers = contiguous ? 1 : 0;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto extract(const It& it)
  {
    if constexpr (contiguous)
    {
      return ::cuda::std::tuple{rebase::unwrap_contiguous(it)};
    }
    else
    {
      return ::cuda::std::tuple<>{};
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <int Offset, typename Ptrs>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static It rebase(const It& it, const Ptrs& ptrs)
  {
    if constexpr (contiguous)
    {
      using ptr_t = decltype(rebase::unwrap_contiguous(it));
      return rebase::rebuild_contiguous(it, static_cast<ptr_t>(::cuda::std::get<Offset>(ptrs)));
    }
    else
    {
      return it;
    }
  }
};

// Handles a list of child iterators held in a tuple, e.g. by a zip iterator.
template <typename... Its>
struct children
{
  static constexpr int num_raw_pointers = (0 + ... + node<Its>::num_raw_pointers);

  // Index of the first raw pointer of child I among all raw pointers of the children
  _CCCL_HOST_DEVICE static constexpr int offset_of(::cuda::std::size_t child)
  {
    constexpr int counts[] = {node<Its>::num_raw_pointers..., 0};
    int offset             = 0;
    for (::cuda::std::size_t i = 0; i < child; ++i)
    {
      offset += counts[i];
    }
    return offset;
  }

  template <typename Tuple, ::cuda::std::size_t... Is>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto
  extract(const Tuple& its, ::cuda::std::index_sequence<Is...>)
  {
    return ::cuda::std::tuple_cat(node<Its>::extract(::cuda::std::get<Is>(its))...);
  }

  template <int Offset, typename Tuple, typename Ptrs, ::cuda::std::size_t... Is>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static ::cuda::std::tuple<Its...>
  rebase(const Tuple& its, const Ptrs& ptrs, ::cuda::std::index_sequence<Is...>)
  {
    return ::cuda::std::tuple<Its...>{
      node<Its>::template rebase<Offset + offset_of(Is)>(::cuda::std::get<Is>(its), ptrs)...};
  }
};

// Base for adaptors wrapping a single (sequentially accessed) child iterator of type Child. Derived must provide
// `static const Child& child(const Self&)` and `static Self rebuild(const Self&, Child)`, where rebuild copies all
// other state (functors, counts, ...) from the given iterator.
template <typename Derived, typename Child>
struct unary_adaptor_node
{
  static constexpr int num_raw_pointers = node<Child>::num_raw_pointers;

  template <typename Self>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto extract(const Self& it)
  {
    return node<Child>::extract(Derived::child(it));
  }

  template <int Offset, typename Self, typename Ptrs>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static Self rebase(const Self& it, const Ptrs& ptrs)
  {
    return Derived::rebuild(it, node<Child>::template rebase<Offset>(Derived::child(it), ptrs));
  }
};

// Whether It is an adaptor that exposes its single (sequentially accessed) child iterator through base() and can be
// rebuilt around a new child by a member function `It __rebase(base_t<It>) const`, which copies all other state (e.g.
// functors). This holds for cuda:: and thrust:: transform, transform_output and transform_input_output iterators, and
// for thrust::permutation_iterator, whose base is its index iterator.
template <typename It>
_CCCL_HOST_DEVICE constexpr bool is_rebasable_adaptor()
{
  if constexpr (has_base<It>)
  {
    return !is_rebasable_leaf<It>() && has_rebase_member<It, base_t<It>>;
  }
  else
  {
    return false;
  }
}

template <typename It>
struct node<It, ::cuda::std::enable_if_t<is_rebasable_adaptor<It>()>> : unary_adaptor_node<node<It>, base_t<It>>
{
  _CCCL_HOST_DEVICE static const base_t<It>& child(const It& it)
  {
    return it.base();
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE static It rebuild(const It& it, base_t<It> child)
  {
    return it.__rebase(::cuda::std::move(child));
  }
};

// Only the index iterator is rebased. The permuted iterator is accessed randomly and therefore kept.
template <typename It, typename Index>
struct node<::cuda::permutation_iterator<It, Index>>
    : unary_adaptor_node<node<::cuda::permutation_iterator<It, Index>>, Index>
{
  using self_t = ::cuda::permutation_iterator<It, Index>;

  _CCCL_HOST_DEVICE static const Index& child(const self_t& it)
  {
    return it.__index();
  }

  _CCCL_HOST_DEVICE static self_t rebuild(const self_t& it, Index child)
  {
    return self_t{it.base(), child};
  }
};

template <typename It>
struct node<::cuda::std::move_iterator<It>> : unary_adaptor_node<node<::cuda::std::move_iterator<It>>, It>
{
  using self_t = ::cuda::std::move_iterator<It>;

  _CCCL_HOST_DEVICE static const It& child(const self_t& it)
  {
    return it.base();
  }

  _CCCL_HOST_DEVICE static self_t rebuild(const self_t&, It child)
  {
    return self_t{child};
  }
};

// Also handles counted iterators over contiguous storage, which are contiguous iterators themselves, but cannot be
// constructed from a raw pointer.
template <typename It>
struct node<::cuda::std::counted_iterator<It>> : unary_adaptor_node<node<::cuda::std::counted_iterator<It>>, It>
{
  using self_t = ::cuda::std::counted_iterator<It>;

  _CCCL_HOST_DEVICE static const It& child(const self_t& it)
  {
    return it.base();
  }

  _CCCL_HOST_DEVICE static self_t rebuild(const self_t& it, It child)
  {
    return self_t{child, it.count()};
  }
};

template <typename... Its>
struct node<::cuda::zip_iterator<Its...>>
{
  using self_t     = ::cuda::zip_iterator<Its...>;
  using children_t = children<Its...>;

  static constexpr int num_raw_pointers = children_t::num_raw_pointers;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto extract(const self_t& it)
  {
    return children_t::extract(it.__iterators(), ::cuda::std::index_sequence_for<Its...>{});
  }

  template <int Offset, typename Ptrs>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static self_t rebase(const self_t& it, const Ptrs& ptrs)
  {
    return self_t{
      children_t::template rebase<Offset>(it.__iterators(), ptrs, ::cuda::std::index_sequence_for<Its...>{})};
  }
};

template <typename Fn, typename... Its>
struct node<::cuda::zip_transform_iterator<Fn, Its...>>
{
  using self_t     = ::cuda::zip_transform_iterator<Fn, Its...>;
  using children_t = children<Its...>;

  static constexpr int num_raw_pointers = children_t::num_raw_pointers;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto extract(const self_t& it)
  {
    return children_t::extract(self_t{it}.__base(), ::cuda::std::index_sequence_for<Its...>{});
  }

  template <int Offset, typename Ptrs>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static self_t rebase(const self_t& it, const Ptrs& ptrs)
  {
    return it.__rebase(
      children_t::template rebase<Offset>(self_t{it}.__base(), ptrs, ::cuda::std::index_sequence_for<Its...>{}));
  }
};

template <typename... Its>
struct node<THRUST_NS_QUALIFIER::zip_iterator<::cuda::std::tuple<Its...>>>
{
  using self_t     = THRUST_NS_QUALIFIER::zip_iterator<::cuda::std::tuple<Its...>>;
  using children_t = children<Its...>;

  static constexpr int num_raw_pointers = children_t::num_raw_pointers;

  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static auto extract(const self_t& it)
  {
    return children_t::extract(it.get_iterator_tuple(), ::cuda::std::index_sequence_for<Its...>{});
  }

  template <int Offset, typename Ptrs>
  [[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE static self_t rebase(const self_t& it, const Ptrs& ptrs)
  {
    return self_t{
      children_t::template rebase<Offset>(it.get_iterator_tuple(), ptrs, ::cuda::std::index_sequence_for<Its...>{})};
  }
};
} // namespace rebase

//! Number of raw pointers that @c extract_raw_pointers returns for an iterator of type @c It.
template <typename It>
inline constexpr int raw_pointer_count_v = rebase::node<It>::num_raw_pointers;

//! Returns the raw pointers of all contiguous iterators in the iterator tree of @p it, in depth-first order.
template <typename It>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto extract_raw_pointers(const It& it)
{
  return rebase::node<It>::extract(it);
}

//! The tuple of raw pointers returned by @c extract_raw_pointers for an iterator of type @c It.
template <typename It>
using raw_pointers_t = decltype(extract_raw_pointers(::cuda::std::declval<const It&>()));

//! Returns a copy of @p it with all contiguous iterators in its iterator tree replaced by the corresponding pointers
//! in @p ptrs. The order of @p ptrs matches the one of @c extract_raw_pointers.
template <typename It, typename... Ts>
[[nodiscard]] _CCCL_HOST_DEVICE _CCCL_FORCEINLINE It
rebase_iterator(const It& it, const ::cuda::std::tuple<Ts*...>& ptrs)
{
  static_assert(sizeof...(Ts) == raw_pointer_count_v<It>, "Number of pointers must match the iterator's raw pointers");
  return rebase::node<It>::template rebase<0>(it, ptrs);
}

#endif // !_CCCL_DOXYGEN_INVOKED
} // namespace detail
CUB_NAMESPACE_END
