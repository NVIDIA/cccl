//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
#  include <cuda/std/cstddef>
#  include <cuda/std/iterator>
#  include <cuda/std/optional>
#  include <cuda/std/type_traits>
#  include <cuda/std/utility>

#  include "../traits.h"
#  include <cccl/c/types.h>
#endif

#include "../mappings/iterator.h"
#include "../mappings/type_info.h"

template <typename Tag, cuda::std::size_t Size, cuda::std::size_t Alignment>
struct alignas(Alignment) output_iterator_state_t
{
  char data[Size];
};

template <typename Tag, cuda::std::size_t Size, cuda::std::size_t Alignment, typename AssignT, auto AssignF>
struct output_iterator_proxy_t
{
  __device__ output_iterator_proxy_t& operator=(AssignT x)
  {
    AssignF(&state, &x);
    return *this;
  }

  output_iterator_state_t<Tag, Size, Alignment> state;
};

template <typename Tag, cccl_iterator_t_mapping Iterator, cccl_type_info_mapping AssignTV>
struct output_iterator_t
{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using difference_type   = cuda::std::size_t;
  using value_type        = void;
  using reference =
    output_iterator_proxy_t<Tag, Iterator.size, Iterator.alignment, typename decltype(AssignTV)::Type, Iterator.assign>;
  using pointer = reference*;

  __device__ reference operator*() const
  {
    return {state};
  }

  __device__ output_iterator_t& operator+=(difference_type diff)
  {
    Iterator.advance(&state, &diff);
    return *this;
  }

  __device__ reference operator[](difference_type diff) const
  {
    output_iterator_t result = *this;
    result += diff;
    return {result.state};
  }

  __device__ output_iterator_t operator+(difference_type diff) const
  {
    output_iterator_t result = *this;
    result += diff;
    return result;
  }

  output_iterator_state_t<Tag, Iterator.size, Iterator.alignment> state;
};

#ifndef _CCCL_C_PARALLEL_JIT_TEMPLATES_PREPROCESS
struct output_iterator_traits
{
  template <typename Tag, cccl_iterator_t_mapping Iterator, cccl_type_info_mapping AssignTV>
  using type = output_iterator_t<Tag, Iterator, AssignTV>;

  static const constexpr auto name = "output_iterator_t";

  template <typename>
  static cuda::std::optional<specialization> special(cccl_iterator_t it, cccl_type_info assign_t)
  {
    if (it.type == cccl_iterator_kind_t::CCCL_POINTER)
    {
      return cuda::std::make_optional(specialization{cccl_type_enum_to_name(assign_t.type, true), ""});
    }

    return cuda::std::nullopt;
  }
};
#endif
