// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

// Define CCCL_IGNORE_DEPRECATED_API before including this header. thrust::constant_iterator and
// thrust::strided_iterator are deprecated. Strided inputs stay in the sweep after they stopped being synthesizing, so
// the comparison still includes that iterator. Shuffle inputs use the element count as the bijection size. Their index
// is the value type, so only types that can hold 2^28 are instantiated.

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/shuffle_iterator.h>
#include <thrust/iterator/strided_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/random>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include <string>

#include <nvbench/nvbench.cuh>

// The stride is not 1, so a strided iterator is not just a counting iterator with a different type. ValueType is the
// value type of the underlying counting iterator. Derived iterators keep that same value type.
inline constexpr int synthesizing_stride = 2;

template <class ValueType>
struct synthesizing_plus_one_t
{
  _CCCL_HOST_DEVICE constexpr ValueType operator()(ValueType value) const noexcept
  {
    return static_cast<ValueType>(value + ValueType{1});
  }
};

// Sum of two counting sequences. The cast keeps the element type when ValueType promotes under arithmetic.
template <class ValueType>
struct synthesizing_add_t
{
  _CCCL_HOST_DEVICE constexpr ValueType operator()(ValueType lhs, ValueType rhs) const noexcept
  {
    return static_cast<ValueType>(lhs + rhs);
  }
};

// Zip inputs produce tuple<ValueType, ValueType>, but a zip iterator's reference can be a proxy with a different type.
// The two arguments are therefore independent template parameters, and the result is always tuple<ValueType,
// ValueType>. The cast keeps the element type when ValueType promotes under arithmetic, as uint8_t + uint8_t does.
template <class ValueType>
struct synthesizing_tuple_plus_t
{
  template <class Lhs, class Rhs>
  _CCCL_HOST_DEVICE ::cuda::std::tuple<ValueType, ValueType> operator()(const Lhs& lhs, const Rhs& rhs) const
  {
    return {static_cast<ValueType>(::cuda::std::get<0>(lhs) + ::cuda::std::get<0>(rhs)),
            static_cast<ValueType>(::cuda::std::get<1>(lhs) + ::cuda::std::get<1>(rhs))};
  }
};

// SubtractLeft invokes the operator as (left, right).
template <class ValueType>
struct synthesizing_tuple_minus_t
{
  template <class Lhs, class Rhs>
  _CCCL_HOST_DEVICE ::cuda::std::tuple<ValueType, ValueType> operator()(const Lhs& lhs, const Rhs& rhs) const
  {
    return {static_cast<ValueType>(::cuda::std::get<0>(lhs) - ::cuda::std::get<0>(rhs)),
            static_cast<ValueType>(::cuda::std::get<1>(lhs) - ::cuda::std::get<1>(rhs))};
  }
};

// no_init is only valid for a trivially constructible value type. Zip outputs are tuples, so those are
// value-initialized. thrust::default_init value-initializes element types that are not trivially constructible.
template <class T>
thrust::device_vector<T> synthesizing_make_output(std::size_t elements)
{
  if constexpr (::cuda::std::is_trivially_constructible_v<T>)
  {
    return thrust::device_vector<T>(elements, thrust::no_init);
  }
  else
  {
    return thrust::device_vector<T>(elements, thrust::default_init);
  }
}

// Even values for scalars. For a zip, the first element, which is the counting iterator.
struct synthesizing_select_even_t
{
  template <class T>
  _CCCL_HOST_DEVICE constexpr bool operator()(const T& value) const noexcept
  {
    if constexpr (::cuda::std::is_integral_v<T>)
    {
      return (value & 1) == 0;
    }
    else
    {
      return ((*this)(::cuda::std::get<0>(value)));
    }
  }
};

template <class ValueType>
struct cuda_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return cuda::counting_iterator<ValueType>{ValueType{0}};
  }

  // Values 0, 1, 2, ... so every other value is even, with the extra item even when the length is odd.
  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return (num_items + 1) / 2;
  }
};

template <class ValueType>
struct thrust_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return thrust::make_counting_iterator(ValueType{0});
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return (num_items + 1) / 2;
  }
};

template <class ValueType>
struct cuda_strided_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return cuda::make_strided_iterator(cuda::counting_iterator<ValueType>{ValueType{0}}, synthesizing_stride);
  }

  // Values 0, 2, 4, ... are all even.
  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items;
  }
};

template <class ValueType>
struct thrust_strided_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return thrust::make_strided_iterator(thrust::make_counting_iterator(ValueType{0}), synthesizing_stride);
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items;
  }
};

template <class ValueType>
struct cuda_transform_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return cuda::transform_iterator{
      cuda::counting_iterator<ValueType>{ValueType{0}}, synthesizing_plus_one_t<ValueType>{}};
  }

  // Values 1, 2, 3, ... so half of them, rounded down, are even.
  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items / 2;
  }
};

template <class ValueType>
struct thrust_transform_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return thrust::make_transform_iterator(
      thrust::make_counting_iterator(ValueType{0}), synthesizing_plus_one_t<ValueType>{});
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items / 2;
  }
};

template <class ValueType>
struct cuda_zip_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = false;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return cuda::make_zip_iterator(
      cuda::counting_iterator<ValueType>{ValueType{0}},
      cuda::transform_iterator{cuda::counting_iterator<ValueType>{ValueType{0}}, synthesizing_plus_one_t<ValueType>{}});
  }

  // Selection looks at the counting iterator, whose values are 0, 1, 2, ...
  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return (num_items + 1) / 2;
  }
};

template <class ValueType>
struct thrust_zip_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = false;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return thrust::make_zip_iterator(
      thrust::make_counting_iterator(ValueType{0}),
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(ValueType{42}), synthesizing_plus_one_t<ValueType>{}));
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return (num_items + 1) / 2;
  }
};

// A constant zero is even, and zero modulo three is zero. The index type is independent of ValueType, so every width
// can address 2^28 items.
template <class ValueType>
struct cuda_constant_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return cuda::make_constant_iterator(ValueType{0});
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items;
  }
};

template <class ValueType>
struct thrust_constant_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return thrust::make_constant_iterator(ValueType{0});
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items;
  }
};

// A shuffle iterator is a permutation of [0, num_items), so it has the same number of even values as a counting
// iterator. The bijection rejects an index at or past num_items.
template <class ValueType>
struct cuda_shuffle_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t num_items)
  {
    return cuda::shuffle_iterator<ValueType>{static_cast<ValueType>(num_items), cuda::std::minstd_rand{0xDEADBEEF}};
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return (num_items + 1) / 2;
  }
};

template <class ValueType>
struct thrust_shuffle_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t num_items)
  {
    return thrust::make_shuffle_iterator(static_cast<ValueType>(num_items), thrust::default_random_engine{0xDEADBEEF});
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return (num_items + 1) / 2;
  }
};

// 0 + 0, 1 + 1, 2 + 2, ... are all even. Modulo three the values cycle through 0, 2, 1.
template <class ValueType>
struct cuda_zip_transform_counting_t
{
  using value_type = ValueType;

  static constexpr bool scalar = true;

  static auto make(::cuda::std::int64_t /*num_items*/)
  {
    return cuda::zip_transform_iterator{
      synthesizing_add_t<ValueType>{},
      cuda::counting_iterator<ValueType>{ValueType{0}},
      cuda::counting_iterator<ValueType>{ValueType{0}}};
  }

  static constexpr ::cuda::std::int64_t num_selected(::cuda::std::int64_t num_items)
  {
    return num_items;
  }
};

// Every integer width. Policies follow the value type of the counting iterator these inputs are built on.
using synthesizing_value_types = nvbench::type_list<
  cuda::std::int8_t,
  cuda::std::int16_t,
  cuda::std::int32_t,
  cuda::std::int64_t,
  cuda::std::uint8_t,
  cuda::std::uint16_t,
  cuda::std::uint32_t,
  cuda::std::uint64_t
#if _CCCL_HAS_INT128()
  ,
  __int128_t,
  __uint128_t
#endif // _CCCL_HAS_INT128()
  >;

// shuffle_iterator stores its position in the value type. 2^28 does not fit in an 8- or 16-bit index.
using synthesizing_shuffle_value_types =
  nvbench::type_list<cuda::std::int32_t,
                     cuda::std::int64_t,
                     cuda::std::uint32_t,
                     cuda::std::uint64_t
#if _CCCL_HAS_INT128()
                     ,
                     __int128_t,
                     __uint128_t
#endif // _CCCL_HAS_INT128()
                     >;

template <template <class> class Input, class ValueTypes>
struct synthesizing_instantiate_inputs;

template <template <class> class Input, class... ValueTypes>
struct synthesizing_instantiate_inputs<Input, nvbench::type_list<ValueTypes...>>
{
  using type = nvbench::type_list<Input<ValueTypes>...>;
};

template <template <class> class Input, class ValueTypes = synthesizing_value_types>
using synthesizing_inputs_for_t = typename synthesizing_instantiate_inputs<Input, ValueTypes>::type;

template <class... Lists>
struct synthesizing_concat_inputs;

template <class List>
struct synthesizing_concat_inputs<List>
{
  using type = List;
};

template <class List1, class List2, class... Rest>
struct synthesizing_concat_inputs<List1, List2, Rest...>
{
  using type = typename synthesizing_concat_inputs<nvbench::tl::concat<List1, List2>, Rest...>::type;
};

using synthesizing_input_types = typename synthesizing_concat_inputs<
  synthesizing_inputs_for_t<cuda_counting_t>,
  synthesizing_inputs_for_t<thrust_counting_t>,
  synthesizing_inputs_for_t<cuda_constant_t>,
  synthesizing_inputs_for_t<thrust_constant_t>,
  synthesizing_inputs_for_t<cuda_shuffle_t, synthesizing_shuffle_value_types>,
  synthesizing_inputs_for_t<thrust_shuffle_t, synthesizing_shuffle_value_types>,
  synthesizing_inputs_for_t<cuda_strided_counting_t>,
  synthesizing_inputs_for_t<thrust_strided_counting_t>,
  synthesizing_inputs_for_t<cuda_transform_counting_t>,
  synthesizing_inputs_for_t<thrust_transform_counting_t>,
  synthesizing_inputs_for_t<cuda_zip_counting_t>,
  synthesizing_inputs_for_t<thrust_zip_counting_t>,
  synthesizing_inputs_for_t<cuda_zip_transform_counting_t>>::type;

#define CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(Tag, InputPrefix, DescriptionPrefix, DescriptionSuffix) \
  template <class ValueType>                                                                           \
  struct nvbench::type_strings<Tag<ValueType>>                                                         \
  {                                                                                                    \
    static std::string input_string()                                                                  \
    {                                                                                                  \
      return std::string{InputPrefix} + nvbench::type_strings<ValueType>::input_string() + ">";        \
    }                                                                                                  \
                                                                                                       \
    static std::string description()                                                                   \
    {                                                                                                  \
      auto value = nvbench::type_strings<ValueType>::description();                                    \
      if (value.empty())                                                                               \
      {                                                                                                \
        value = nvbench::type_strings<ValueType>::input_string();                                      \
      }                                                                                                \
      return std::string{DescriptionPrefix} + value + DescriptionSuffix;                               \
    }                                                                                                  \
  };

CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(cuda_counting_t, "cuda_counting<", "cuda::counting_iterator<", ">")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(thrust_counting_t, "thrust_counting<", "thrust::counting_iterator<", ">")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(cuda_constant_t, "cuda_constant<", "cuda::constant_iterator<", ">")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(thrust_constant_t, "thrust_constant<", "thrust::constant_iterator<", ">")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(cuda_shuffle_t, "cuda_shuffle<", "cuda::shuffle_iterator<", ">")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(thrust_shuffle_t, "thrust_shuffle<", "thrust::shuffle_iterator<", ">")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  cuda_strided_counting_t, "cuda_strided_counting<", "cuda::strided_iterator<counting<", ">>")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  thrust_strided_counting_t, "thrust_strided_counting<", "thrust::strided_iterator<counting<", ">>")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  cuda_transform_counting_t, "cuda_transform_counting<", "cuda::transform_iterator<counting<", ">>")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  thrust_transform_counting_t, "thrust_transform_counting<", "thrust::transform_iterator<counting<", ">>")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  cuda_zip_counting_t, "cuda_zip_counting<", "cuda::zip_iterator<counting, transform<", ">>")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  thrust_zip_counting_t, "thrust_zip_counting<", "thrust::zip_iterator<counting, transform<", ">>")
CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS(
  cuda_zip_transform_counting_t, "cuda_zip_transform<", "cuda::zip_transform_iterator<counting, counting<", ">>")

#undef CCCL_DECLARE_SYNTHESIZING_TYPE_STRINGS
