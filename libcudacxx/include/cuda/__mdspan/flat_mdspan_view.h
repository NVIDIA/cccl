/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _CUDA___MDSPAN_FLAT_MDSPAN_VIEW
#define _CUDA___MDSPAN_FLAT_MDSPAN_VIEW

#include <cuda/__mdspan/flat_mdspan_iterator.h>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _MDSpan>
class __flat_mdspan_view;

/**
 * @brief A flattened view of an `mdspan` that allows efficient random
 * elementwise access.
 */
template <typename _Element, typename _Extent, typename _Layout, typename _Accessor>
class __flat_mdspan_view<_CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>>
{
public:
  using mdspan_type    = _CUDA_VSTD::mdspan<_Element, _Extent, _Layout, _Accessor>;
  using iterator       = __flat_mdspan_iterator<mdspan_type>;
  using const_iterator = iterator;

  /**
   * @brief Construct a flat mdspan view.
   *
   * @param span The span to view.
   */
  constexpr explicit __flat_mdspan_view(mdspan_type __span) noexcept
      : __span_{_CUDA_VSTD::move(__span)}
  {}

  /**
   * @return An iterator to the beginning of the range.
   */
  [[nodiscard]] constexpr iterator begin() const noexcept
  {
    return cbegin();
  }

  /**
   * @return An iterator to the beginning of the range.
   */
  [[nodiscard]] constexpr iterator cbegin() const noexcept
  {
    return iterator{{}, __span_, 0};
  }

  /**
   * @return An iterator to the end of the range.
   */
  [[nodiscard]] constexpr iterator end() const noexcept
  {
    return cend();
  }

  /**
   * @return An iterator to the beginning of the range.
   */
  [[nodiscard]] constexpr iterator cend() const noexcept
  {
    return iterator{{}, __span_, static_cast<typename mdspan_type::index_type>(__span_.size())};
  }

private:
  mdspan_type __span_{};
};

template <typename _T>
__flat_mdspan_view(_T) -> __flat_mdspan_view<_T>;

_LIBCUDACXX_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_HOST_DEVICE_ACCESSOR
