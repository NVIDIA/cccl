#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H

#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"
#include "test_range.h"

struct MoveOnlyView : cuda::std::ranges::view_base
{
  int* ptr_;

  __host__ __device__ constexpr explicit MoveOnlyView(int* ptr)
      : ptr_(ptr)
  {}
  MoveOnlyView(MoveOnlyView&&)            = default;
  MoveOnlyView& operator=(MoveOnlyView&&) = default;

  __host__ __device__ constexpr int* begin() const
  {
    return ptr_;
  }
  __host__ __device__ constexpr sentinel_wrapper<int*> end() const
  {
    return sentinel_wrapper<int*>{ptr_ + 8};
  }
};
static_assert(cuda::std::ranges::view<MoveOnlyView>);
static_assert(cuda::std::ranges::contiguous_range<MoveOnlyView>);
static_assert(!cuda::std::copyable<MoveOnlyView>);

struct CopyableView : cuda::std::ranges::view_base
{
  int* ptr_;
  __host__ __device__ constexpr explicit CopyableView(int* ptr)
      : ptr_(ptr)
  {}

  __host__ __device__ constexpr int* begin() const
  {
    return ptr_;
  }
  __host__ __device__ constexpr sentinel_wrapper<int*> end() const
  {
    return sentinel_wrapper<int*>{ptr_ + 8};
  }
};
static_assert(cuda::std::ranges::view<CopyableView>);
static_assert(cuda::std::ranges::contiguous_range<CopyableView>);
static_assert(cuda::std::copyable<CopyableView>);

using ForwardIter = forward_iterator<int*>;
struct SizedForwardView : cuda::std::ranges::view_base
{
  int* ptr_;
  __host__ __device__ constexpr explicit SizedForwardView(int* ptr)
      : ptr_(ptr)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return ForwardIter(ptr_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<ForwardIter>(ForwardIter(ptr_ + 8));
  }
};
static_assert(cuda::std::ranges::view<SizedForwardView>);
static_assert(cuda::std::ranges::forward_range<SizedForwardView>);
static_assert(cuda::std::ranges::sized_range<SizedForwardView>);

using RandomAccessIter = random_access_iterator<int*>;
struct SizedRandomAccessView : cuda::std::ranges::view_base
{
  int* ptr_;
  __host__ __device__ constexpr explicit SizedRandomAccessView(int* ptr)
      : ptr_(ptr)
  {}
  __host__ __device__ constexpr auto begin() const
  {
    return RandomAccessIter(ptr_);
  }
  __host__ __device__ constexpr auto end() const
  {
    return sized_sentinel<RandomAccessIter>(RandomAccessIter(ptr_ + 8));
  }
};
static_assert(cuda::std::ranges::view<SizedRandomAccessView>);
static_assert(cuda::std::ranges::random_access_range<SizedRandomAccessView>);
static_assert(cuda::std::ranges::sized_range<SizedRandomAccessView>);

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_TAKE_TYPES_H
