//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_FLAT_MDSPAN_ITERATOR
#define _CUDA___MDSPAN_FLAT_MDSPAN_ITERATOR

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/cmp.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_lvalue_reference.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/array>
#include <cuda/std/iterator>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _MDSpan>
class __flat_mdspan_view;

/**
 * @brief An iterator over an `mdspan` that presents a flat view and allows
 * random elementwise access. It is particularly handy for passing to Thrust
 * algorithms to perform elementwise operations in parallel.
 */
template <typename _Element, typename _Extent, typename _Layout, typename _Accessor>
class __flat_mdspan_iterator<_CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>>
{
  class __pointer_wrapper
  {
  public:
    _CUDA_VSTD::remove_const_t<_Element> __elem_;

    [[nodiscard]] constexpr _Element* operator->() const noexcept
    {
      return _CUDA_VSTD::addressof(__elem_);
    }
  };

public:
  using mdspan_type = _CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>;
  using index_type  = typename mdspan_type::index_type;
  // The CCCL mdspan impls will fail to compile for 0-D, but in classic C++ fashion the
  // compiler error messages are inscrutable. So better to just guard against that mess with an
  // early static_assert()
  static_assert(mdspan_type::rank() >= 1, "Flat views over 0-D mdspans are not supported");

  using value_type        = _CUDA_VSTD::remove_const_t<_Element>;
  using reference         = typename mdspan_type::reference;
  using difference_type   = _CUDA_VSTD::ptrdiff_t;
  using iterator_category = std::random_access_iterator_tag;
  using pointer           = _CUDA_VSTD::
    conditional_t<_CUDA_VSTD::is_lvalue_reference_v<reference>, _CUDA_VSTD::add_pointer_t<reference>, __pointer_wrapper>;

  class __construct_key
  {
    friend __flat_mdspan_iterator;
    friend __flat_mdspan_view<mdspan_type>;
  };

  __flat_mdspan_iterator() = delete;

  /**
   * @brief Construct a flat mdspan iterator.
   *
   * @param span The span to view.
   * @param idx The linear index of the iterator (`0` for begin, `span.size()` for end).
   */
  constexpr explicit __flat_mdspan_iterator(__construct_key, const mdspan_type& __span, index_type __idx) noexcept
      : __span_{_CUDA_VSTD::addressof(__span)}
      , __idx_{__idx}
  {}

  /**
   * @return A reference to the selected element.
   */
  [[nodiscard]] constexpr reference operator*() const noexcept
  {
    constexpr auto __DIM = mdspan_type::rank();

    if constexpr (__DIM <= 1)
    {
      // The std::array implementation is correct for any dimension, but going through it appears
      // to confuse the optimizers and they generate very suboptimal code...
      //
      // So for 1D access we can skip it (there is no need to transform our offset) and can use
      // our `idx_` directly.
      return (*__span_)(__idx_);
    }
    else
    {
      // This code right here is why flat mdspan iterators are (currently) terribly
      // inefficient. The compilers will able to optimize this loop to its minimum, but they are
      // *not* able to optimize the outer loop to get rid of this calculation. For some reason,
      // this code below is complex enough to defeat the vectorization engines of every major
      // compiler.
      //
      // Concretely, a completely equivalent loop:
      //
      // for (std::size_t i = 0; i < span.extent(0); ++i) {
      //   for (std::size_t j = 0; j < span.extent(1); ++j) {
      //     span(i, j) = ...
      //   }
      // }
      //
      // Will be fully vectorized by optimizers, but the following (which is more or less what
      // our iterator expands to):
      //
      // for (std::size_t i = 0; i < PROD(span.extents()...); ++i) {
      //   std::array<std::size_t, DIM> md_idx = delinearize(i); // this is us
      //
      //   span(md_idx) = ...
      // }
      //
      // Is considered too complicated to unravel. A shame. Hopefully optimizers eventually
      // become smart enough to unravel this.
      _CUDA_VSTD::array<index_type, __DIM> __md_idx;
      {
        auto __index = idx_; // Cannot declare in for-loop, since index_type != rank_type

        for (auto __dim = __DIM; __dim-- > 0;)
        {
          __md_idx[__dim] = __index_ % __span_->extent(__dim);
          __index /= __span_->extent(__dim);
        }
      }
      return (*__span_)(__md_idx);
    }
  }

  /**
   * @return A pointer to the selected element.
   */
  [[nodiscard]] constexpr pointer operator->() const noexcept
  {
    if constexpr (_CUDA_VSTD::is_lvalue_reference_v<reference>)
    {
      return &operator*();
    }
    else
    {
      return __pointer_wrapper{operator*()};
    }
  }

  /**
   * @brief Pre-increment the iterator.
   *
   * @return A reference to this.
   */
  constexpr __flat_mdspan_iterator& operator++() noexcept
  {
    _CCCL_ASSERT(_CUDA_VSTD::cmp_less(__idx_, __span_->size()), "");
    ++__idx_;
    return *this;
  }

  /**
   * @brief Post-increment the iterator.
   *
   * @return A copy of the old iterator.
   */
  constexpr __flat_mdspan_iterator operator++(int) noexcept
  {
    auto __copy = *this;

    ++(*this);
    return __copy;
  }

  /**
   * @brief Pre-decrement the iterator.
   *
   * @return A reference to this.
   */
  constexpr __flat_mdspan_iterator& operator--() noexcept
  {
    _CCCL_ASSERT(__idx_ > 0, "");
    --__idx_;
    return *this;
  }

  /**
   * @brief Post-decrement the iterator.
   *
   * @return A copy of the old iterator.
   */
  constexpr __flat_mdspan_iterator operator--(int) noexcept
  {
    auto __copy = *this;

    --(*this);
    return __copy;
  }

  /**
   * @brief In-place add to the iterator.
   *
   * @param n The amount to increment.
   *
   * @return A reference to this.
   */
  constexpr __flat_mdspan_iterator& operator+=(difference_type __n) noexcept
  {
    if (__n < 0)
    {
      _CCCL_ASSERT(_CUDA_VSTD::cmp_greater_equal(__idx_, -__n));
      __idx_ -= static_cast<index_type>(-__n);
    }
    else
    {
      _CCCL_ASSERT(_CUDA_VSTD::cmp_less_equal(__idx_ + static_cast<index_type>(__n), __span_->size()));
      __idx_ += static_cast<index_type>(__n);
    }
    return *this;
  }

  /**
   * @brief In-place minus to the iterator.
   *
   * @param n The amount to decrement.
   *
   * @return A reference to this.
   */
  constexpr __flat_mdspan_iterator& operator-=(difference_type __n) noexcept
  {
    return operator+=(-__n);
  }

  /**
   * @brief Access the iterator at a linear offset.
   *
   * @param n The linear index.
   *
   * @return A reference to the element at that index.
   */
  [[nodiscard]] constexpr reference operator[](difference_type __n) const noexcept
  {
    return *(*this + __n);
  }

  [[nodiscard]] friend difference_type
  operator-(const __flat_mdspan_iterator& __self, const __flat_mdspan_iterator& __other) noexcept
  {
    using difference_type =
      typename __flat_mdspan_iterator<_CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>>::difference_type;

    _CCCL_ASSERT(__self.__span_ == __other.__span_);
    return static_cast<difference_type>(__self.__idx_) - static_cast<difference_type>(__other.__idx_);
  }

  [[nodiscard]] friend __flat_mdspan_iterator operator-(__flat_mdspan_iterator __self, difference_type __n) noexcept
  {
    __self += __n;
    return __self;
  }

  [[nodiscard]] friend __flat_mdspan_iterator operator+(__flat_mdspan_iterator __self, difference_type __n) noexcept
  {
    __self += __n;
    return __self;
  }

  [[nodiscard]] friend __flat_mdspan_iterator operator+(difference_type __n, __flat_mdspan_iterator __self) noexcept
  {
    __self -= __n;
    return __self;
  }

  [[nodiscard]] friend bool operator==(const __flat_mdspan_iterator& __lhs, const __flat_mdspan_iterator& __rhs) noexcept
  {
    _CCCL_ASSERT(__lhs.__span_ == __rhs.__span_, "");
    return __lhs.__idx_ == __rhs.__idx_;
  }

  [[nodiscard]] friend bool operator!=(const __flat_mdspan_iterator& __lhs, const __flat_mdspan_iterator& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  [[nodiscard]] friend bool operator<(const __flat_mdspan_iterator& __lhs, const __flat_mdspan_iterator& __rhs) noexcept
  {
    _CCCL_ASSERT(__lhs.__span_ == __rhs.__span_, "");
    return __lhs.__idx_ < __rhs.__idx_;
  }

  [[nodiscard]] friend bool operator>(const __flat_mdspan_iterator& __lhs, const __flat_mdspan_iterator& __rhs) noexcept
  {
    return __rhs < __lhs;
  }

  [[nodiscard]] friend bool operator<=(const __flat_mdspan_iterator& __lhs, const __flat_mdspan_iterator& __rhs) noexcept
  {
    return !(__rhs < __lhs);
  }

  [[nodiscard]] friend bool operator>=(const __flat_mdspan_iterator& __lhs, const __flat_mdspan_iterator& __rhs) noexcept
  {
    return !(__lhs < __rhs);
  }

private:
  const mdspan_type* __span_{};
  index_type __idx_{};
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR
