#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/normal_iterator.h>
#include <thrust/iterator/transform_input_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/memory.h>

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <unittest/unittest.h>

struct ContiguousAdaptorPlusOne
{
  _CCCL_HOST_DEVICE int operator()(int value) const
  {
    return value + 1;
  }
};

TEST_CASE("contiguous iterator adaptor thrust storage", "[iterators]")
{
  static_assert(cuda::std::__has_contiguous_traversal<thrust::device_ptr<int>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_ptr<int>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::pointer<int, thrust::device_system_tag, int&>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::pointer<int, thrust::device_system_tag, int&>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::detail::normal_iterator<thrust::device_ptr<int>>>);
  static_assert(cuda::std::__has_contiguous_traversal<thrust::detail::normal_iterator<int*>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::detail::normal_iterator<int*>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::host_vector<int>::iterator>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_vector<int>::iterator>);

  using Transformed       = thrust::transform_iterator<ContiguousAdaptorPlusOne, thrust::device_ptr<int>>;
  using Nested            = thrust::transform_iterator<ContiguousAdaptorPlusOne, Transformed>;
  using Normal            = thrust::detail::normal_iterator<int*>;
  using TransformedNormal = thrust::transform_iterator<ContiguousAdaptorPlusOne, Normal>;
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Transformed>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Transformed>
                == cuda::__is_contiguous_iterator_or_adaptor_v<thrust::device_ptr<int>>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<Nested>);
  static_assert(cuda::__is_contiguous_iterator_or_adaptor_v<TransformedNormal>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::counting_iterator<int>>>);
  // A reverse iterator unwraps to a pointer, but does not traverse the storage forward
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::std::reverse_iterator<int*>>>);
  static_assert(
    cuda::__is_contiguous_iterator_or_adaptor_v<
      thrust::transform_iterator<ContiguousAdaptorPlusOne, cuda::transform_iterator<ContiguousAdaptorPlusOne, int*>>>);
  static_assert(
    cuda::__is_contiguous_iterator_or_adaptor_v<cuda::transform_iterator<ContiguousAdaptorPlusOne, Transformed>>);

  using TransformedOutput = thrust::transform_output_iterator<ContiguousAdaptorPlusOne, int*>;
  using TransformedInOut =
    thrust::transform_input_output_iterator<ContiguousAdaptorPlusOne, ContiguousAdaptorPlusOne, int*>;
  using NestedOutput = thrust::transform_output_iterator<ContiguousAdaptorPlusOne, Transformed>;
  using NestedInOut =
    thrust::transform_input_output_iterator<ContiguousAdaptorPlusOne, ContiguousAdaptorPlusOne, Normal>;
  // Output adaptors are not contiguous adaptors, even over contiguous storage
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<TransformedOutput>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<const TransformedOutput>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<TransformedInOut>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<const TransformedInOut>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<NestedOutput>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<NestedInOut>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_output_iterator<ContiguousAdaptorPlusOne, thrust::counting_iterator<int>>>);
  static_assert(!cuda::__is_contiguous_iterator_or_adaptor_v<
                thrust::transform_input_output_iterator<ContiguousAdaptorPlusOne,
                                                        ContiguousAdaptorPlusOne,
                                                        thrust::counting_iterator<int>>>);

  // The base of a counting iterator is its counter, so there is no storage to rebase
  using Counting = thrust::counting_iterator<int>;
  static_assert(cuda::std::is_same_v<cuda::__unwrapped_iter_value_t<Counting>, int>);
  static_assert(!cuda::__iterator_can_rebase<Counting>);
  static_assert(!cuda::__iterator_can_rebase<thrust::transform_iterator<ContiguousAdaptorPlusOne, Counting>>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedOutput>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedInOut>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<NestedOutput>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<NestedInOut>())), int*>);

  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<thrust::device_ptr<int>>())),
                         int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(
                                       ::cuda::std::declval<thrust::pointer<int, thrust::device_system_tag, int&>>())),
                                     int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Normal>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(
                           ::cuda::std::declval<thrust::detail::normal_iterator<thrust::device_ptr<int>>>())),
                         int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Transformed>())), int*>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<Nested>())), int*>);
  static_assert(
    cuda::std::is_same_v<decltype(cuda::__unwrap_iterator_adaptor(::cuda::std::declval<TransformedNormal>())), int*>);

  static_assert(cuda::__iterator_can_rebase<Transformed>);
  static_assert(cuda::__iterator_can_rebase<Nested>);
  static_assert(cuda::__iterator_can_rebase<TransformedOutput>);
  static_assert(cuda::__iterator_can_rebase<TransformedInOut>);

  int input[]  = {1, 2, 3, 4};
  int staged[] = {10, 20, 30, 40};

  {
    auto iter      = thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{});
    auto unwrapped = cuda::__unwrap_iterator_adaptor(iter);
    REQUIRE(unwrapped == input);

    auto rebased = iter;
    rebased.__rebase(staged);
    REQUIRE(*iter == 2);
    REQUIRE(*rebased == 11);
    REQUIRE(rebased.base() == staged);
  }

  {
    auto inner = thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{});
    auto outer = thrust::make_transform_iterator(inner, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(outer) == input);
    auto rebased = outer;
    rebased.__rebase(staged);
    REQUIRE(*outer == 3);
    REQUIRE(*rebased == 12);
  }

  {
    Normal iter{input};
    auto rebased = iter;
    rebased.__rebase(staged);
    REQUIRE(iter.base() == input);
    REQUIRE(rebased.base() == staged);
    REQUIRE(*rebased == 10);

    auto fancy = thrust::make_transform_iterator(iter, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == input);
    auto fancy_rebased = fancy;
    fancy_rebased.__rebase(staged);
    REQUIRE(*fancy == 2);
    REQUIRE(*fancy_rebased == 11);
  }

  {
    thrust::pointer<int, thrust::device_system_tag, int&> stored{input};
    auto stored_rebased = stored;
    stored_rebased.__rebase(staged);
    REQUIRE(stored.get() == input);
    REQUIRE(stored_rebased.get() == staged);

    thrust::device_ptr<int> ptr{input};
    auto ptr_rebased = ptr;
    ptr_rebased.__rebase(staged);
    REQUIRE(ptr.get() == input);
    REQUIRE(ptr_rebased.get() == staged);

    auto fancy = thrust::make_transform_iterator(ptr, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == input);
    auto fancy_rebased = fancy;
    fancy_rebased.__rebase(staged);
    REQUIRE(fancy.base().get() == input);
    REQUIRE(fancy_rebased.base().get() == staged);
  }

  {
    int alternative = 7;
    thrust::device_vector<int> values(1);
    auto rebased = values.begin();
    rebased.__rebase(&alternative);
    REQUIRE(rebased.base().get() == &alternative);

    auto fancy = thrust::make_transform_iterator(values.begin(), ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(fancy) == values.begin().base().get());
    auto fancy_rebased = fancy;
    fancy_rebased.__rebase(&alternative);
    REQUIRE(fancy_rebased.base().base().get() == &alternative);
  }

  {
    auto iter = thrust::make_transform_output_iterator(input, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == input);
    auto rebased = iter;
    rebased.__rebase(staged);
    *rebased = 5;
    REQUIRE(input[0] == 1);
    REQUIRE(staged[0] == 6);
    REQUIRE(rebased.base() == staged);

    auto nested = thrust::make_transform_output_iterator(
      thrust::make_transform_iterator(input, ContiguousAdaptorPlusOne{}), ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(nested) == input);
    auto nested_rebased = nested;
    nested_rebased.__rebase(staged);
    REQUIRE(nested_rebased.base().base() == staged);
  }

  {
    staged[0] = 10;
    auto iter =
      thrust::make_transform_input_output_iterator(input, ContiguousAdaptorPlusOne{}, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(iter) == input);
    REQUIRE(*iter == 2);
    auto rebased = iter;
    rebased.__rebase(staged);
    REQUIRE(*rebased == 11);
    *rebased = 7;
    REQUIRE(input[0] == 1);
    REQUIRE(staged[0] == 8);
    REQUIRE(rebased.base() == staged);

    staged[0] = 10;
    Normal stored{input};
    auto nested =
      thrust::make_transform_input_output_iterator(stored, ContiguousAdaptorPlusOne{}, ContiguousAdaptorPlusOne{});
    REQUIRE(cuda::__unwrap_iterator_adaptor(nested) == input);
    auto nested_rebased = nested;
    nested_rebased.__rebase(staged);
    REQUIRE(*nested_rebased == 11);
    REQUIRE(nested_rebased.base().base() == staged);
  }
}
