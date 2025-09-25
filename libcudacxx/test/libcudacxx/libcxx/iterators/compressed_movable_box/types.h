#ifndef TEST_LIBCXX_ITERATORS_COMPRESSED_MOVABLE_BOX_TYPES_H
#define TEST_LIBCXX_ITERATORS_COMPRESSED_MOVABLE_BOX_TYPES_H

#include <cuda/std/utility>

inline constexpr int MayThrow = 40;

struct TrivialEmpty
{
  __host__ __device__ friend constexpr bool operator==(const TrivialEmpty&, const int val) noexcept
  {
    return val == 42;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Default construction
////////////////////////////////////////////////////////////////////////////////
template <int Val>
struct NotTriviallyDefaultConstructible
{
  int val_ = Val;

  __host__ __device__ constexpr NotTriviallyDefaultConstructible(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ constexpr NotTriviallyDefaultConstructible() noexcept(Val != MayThrow)
      : val_(Val)
  {}
  NotTriviallyDefaultConstructible(const NotTriviallyDefaultConstructible&)            = default;
  NotTriviallyDefaultConstructible(NotTriviallyDefaultConstructible&&)                 = default;
  NotTriviallyDefaultConstructible& operator=(const NotTriviallyDefaultConstructible&) = default;
  NotTriviallyDefaultConstructible& operator=(NotTriviallyDefaultConstructible&&)      = default;
};

template <int Val>
struct NotTriviallyDefaultConstructibleEmpty
{
  __host__ __device__ constexpr NotTriviallyDefaultConstructibleEmpty(const int val) noexcept {}

  __host__ __device__ constexpr NotTriviallyDefaultConstructibleEmpty() noexcept(Val != MayThrow) {}
  NotTriviallyDefaultConstructibleEmpty(const NotTriviallyDefaultConstructibleEmpty&)            = default;
  NotTriviallyDefaultConstructibleEmpty(NotTriviallyDefaultConstructibleEmpty&&)                 = default;
  NotTriviallyDefaultConstructibleEmpty& operator=(const NotTriviallyDefaultConstructibleEmpty&) = default;
  NotTriviallyDefaultConstructibleEmpty& operator=(NotTriviallyDefaultConstructibleEmpty&&)      = default;

  __host__ __device__ friend constexpr bool
  operator==(const NotTriviallyDefaultConstructibleEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

struct NotDefaultConstructible
{
  int val_;

  __host__ __device__ constexpr NotDefaultConstructible(const int val) noexcept
      : val_(val)
  {}

  __host__ __device__ NotDefaultConstructible()                      = delete;
  NotDefaultConstructible(const NotDefaultConstructible&)            = default;
  NotDefaultConstructible(NotDefaultConstructible&&)                 = default;
  NotDefaultConstructible& operator=(const NotDefaultConstructible&) = default;
  NotDefaultConstructible& operator=(NotDefaultConstructible&&)      = default;
};

////////////////////////////////////////////////////////////////////////////////
// copy construction
////////////////////////////////////////////////////////////////////////////////
template <int Val>
struct NotTriviallyCopyConstructible
{
  int val_ = Val;

  __host__ __device__ constexpr NotTriviallyCopyConstructible(const int val) noexcept
      : val_(val)
  {}
  NotTriviallyCopyConstructible() = default;
  __host__ __device__ constexpr NotTriviallyCopyConstructible(const NotTriviallyCopyConstructible& other) noexcept(
    Val != MayThrow)
      : val_(other.val_)
  {}
  NotTriviallyCopyConstructible(NotTriviallyCopyConstructible&&)                 = default;
  NotTriviallyCopyConstructible& operator=(const NotTriviallyCopyConstructible&) = default;
  NotTriviallyCopyConstructible& operator=(NotTriviallyCopyConstructible&&)      = default;
};

template <int Val>
struct NotTriviallyCopyConstructibleEmpty
{
  __host__ __device__ constexpr NotTriviallyCopyConstructibleEmpty(const int) noexcept {}
  NotTriviallyCopyConstructibleEmpty() = default;
  __host__ __device__ constexpr NotTriviallyCopyConstructibleEmpty(const NotTriviallyCopyConstructibleEmpty&) noexcept(
    Val != MayThrow)
  {}
  NotTriviallyCopyConstructibleEmpty(NotTriviallyCopyConstructibleEmpty&&)                 = default;
  NotTriviallyCopyConstructibleEmpty& operator=(const NotTriviallyCopyConstructibleEmpty&) = default;
  NotTriviallyCopyConstructibleEmpty& operator=(NotTriviallyCopyConstructibleEmpty&&)      = default;

  __host__ __device__ friend constexpr bool operator==(const NotTriviallyCopyConstructibleEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

template <int Val>
struct NotCopyConstructible
{
  int val_ = Val;

  __host__ __device__ constexpr NotCopyConstructible(const int val) noexcept
      : val_(val)
  {}
  NotCopyConstructible()                                                          = default;
  __host__ __device__ constexpr NotCopyConstructible(const NotCopyConstructible&) = delete;
  NotCopyConstructible(NotCopyConstructible&&)                                    = default;
  NotCopyConstructible& operator=(const NotCopyConstructible&)                    = default;
  NotCopyConstructible& operator=(NotCopyConstructible&&)                         = default;
};

struct NotCopyConstructibleEmpty
{
  __host__ __device__ constexpr NotCopyConstructibleEmpty(const int) noexcept {}
  NotCopyConstructibleEmpty()                                                     = default;
  __host__ __device__ NotCopyConstructibleEmpty(const NotCopyConstructibleEmpty&) = delete;
  NotCopyConstructibleEmpty(NotCopyConstructibleEmpty&&)                          = default;
  NotCopyConstructibleEmpty& operator=(const NotCopyConstructibleEmpty&)          = default;
  NotCopyConstructibleEmpty& operator=(NotCopyConstructibleEmpty&&)               = default;

  __host__ __device__ friend constexpr bool operator==(const NotCopyConstructibleEmpty&, const int val) noexcept
  {
    return val == 42;
  }
};

////////////////////////////////////////////////////////////////////////////////
// move construction
////////////////////////////////////////////////////////////////////////////////
template <int Val>
struct NotTriviallyMoveConstructible
{
  int val_ = Val;

  __host__ __device__ constexpr NotTriviallyMoveConstructible(const int val) noexcept
      : val_(val)
  {}
  NotTriviallyMoveConstructible()                                     = default;
  NotTriviallyMoveConstructible(const NotTriviallyMoveConstructible&) = default;
  __host__
    __device__ constexpr NotTriviallyMoveConstructible(NotTriviallyMoveConstructible&& other) noexcept(Val != MayThrow)
      : val_(cuda::std::exchange(other.val_, -1))
  {}
  NotTriviallyMoveConstructible& operator=(const NotTriviallyMoveConstructible&) = default;
  NotTriviallyMoveConstructible& operator=(NotTriviallyMoveConstructible&&)      = default;
};

template <int Val>
struct NotTriviallyMoveConstructibleEmpty
{
  __host__ __device__ constexpr NotTriviallyMoveConstructibleEmpty(const int) noexcept {}
  NotTriviallyMoveConstructibleEmpty()                                          = default;
  NotTriviallyMoveConstructibleEmpty(const NotTriviallyMoveConstructibleEmpty&) = default;
  __host__ __device__ constexpr NotTriviallyMoveConstructibleEmpty(NotTriviallyMoveConstructibleEmpty&&) noexcept(
    Val != MayThrow)
  {}
  NotTriviallyMoveConstructibleEmpty& operator=(const NotTriviallyMoveConstructibleEmpty&) = default;
  NotTriviallyMoveConstructibleEmpty& operator=(NotTriviallyMoveConstructibleEmpty&&)      = default;

  __host__ __device__ friend constexpr bool operator==(const NotTriviallyMoveConstructibleEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

// movable box *must* be movable

////////////////////////////////////////////////////////////////////////////////
// copy assignable
////////////////////////////////////////////////////////////////////////////////
template <int Val>
struct NotTriviallyCopyAssignable
{
  int val_ = Val;

  __host__ __device__ constexpr NotTriviallyCopyAssignable(const int val) noexcept
      : val_(val)
  {}
  NotTriviallyCopyAssignable()                                  = default;
  NotTriviallyCopyAssignable(const NotTriviallyCopyAssignable&) = default;
  NotTriviallyCopyAssignable(NotTriviallyCopyAssignable&&)      = default;
  __host__ __device__ constexpr NotTriviallyCopyAssignable&
  operator=(const NotTriviallyCopyAssignable& other) noexcept(Val != MayThrow)
  {
    val_ = other.val_;
    return *this;
  }
  NotTriviallyCopyAssignable& operator=(NotTriviallyCopyAssignable&&) = default;
};

template <int Val>
struct NotTriviallyCopyAssignableEmpty
{
  __host__ __device__ constexpr NotTriviallyCopyAssignableEmpty(const int) noexcept {}
  NotTriviallyCopyAssignableEmpty()                                       = default;
  NotTriviallyCopyAssignableEmpty(const NotTriviallyCopyAssignableEmpty&) = default;
  NotTriviallyCopyAssignableEmpty(NotTriviallyCopyAssignableEmpty&&)      = default;
  __host__ __device__ constexpr NotTriviallyCopyAssignableEmpty&
  operator=(const NotTriviallyCopyAssignableEmpty& other) noexcept(Val != MayThrow)
  {
    return *this;
  }
  NotTriviallyCopyAssignableEmpty& operator=(NotTriviallyCopyAssignableEmpty&&) = default;

  __host__ __device__ friend constexpr bool operator==(const NotTriviallyCopyAssignableEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

template <int Val>
struct NotCopyAssignable
{
  int val_ = Val;

  __host__ __device__ constexpr NotCopyAssignable(const int val) noexcept
      : val_(val)
  {}
  NotCopyAssignable() = default;
  __host__ __device__ constexpr NotCopyAssignable(const NotCopyAssignable& other) noexcept(Val != MayThrow)
      : val_(other.val_)
  {}
  NotCopyAssignable(NotCopyAssignable&&)                 = default;
  NotCopyAssignable& operator=(const NotCopyAssignable&) = delete;
  NotCopyAssignable& operator=(NotCopyAssignable&&)      = default;
};

template <int Val>
struct NotCopyAssignableEmpty
{
  __host__ __device__ constexpr NotCopyAssignableEmpty(const int val) noexcept {}

  NotCopyAssignableEmpty() = default;
  __host__ __device__ constexpr NotCopyAssignableEmpty(const NotCopyAssignableEmpty&) noexcept(Val != MayThrow) {}
  NotCopyAssignableEmpty(NotCopyAssignableEmpty&&)                 = default;
  NotCopyAssignableEmpty& operator=(const NotCopyAssignableEmpty&) = delete;
  NotCopyAssignableEmpty& operator=(NotCopyAssignableEmpty&&)      = default;

  __host__ __device__ friend constexpr bool operator==(const NotCopyAssignableEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

struct NotCopyConstructibleOrAssignable
{
  int val_ = 42;

  __host__ __device__ constexpr NotCopyConstructibleOrAssignable(const int val) noexcept
      : val_(val)
  {}
  NotCopyConstructibleOrAssignable()                                                            = default;
  __host__ __device__ NotCopyConstructibleOrAssignable(const NotCopyConstructibleOrAssignable&) = delete;
  NotCopyConstructibleOrAssignable(NotCopyConstructibleOrAssignable&&)                          = default;
  NotCopyConstructibleOrAssignable& operator=(const NotCopyConstructibleOrAssignable&)          = delete;
  NotCopyConstructibleOrAssignable& operator=(NotCopyConstructibleOrAssignable&&)               = default;
};

struct NotCopyConstructibleOrAssignableEmpty
{
  __host__ __device__ constexpr NotCopyConstructibleOrAssignableEmpty(const int val) noexcept {}

  NotCopyConstructibleOrAssignableEmpty()                                                                 = default;
  __host__ __device__ NotCopyConstructibleOrAssignableEmpty(const NotCopyConstructibleOrAssignableEmpty&) = delete;
  NotCopyConstructibleOrAssignableEmpty(NotCopyConstructibleOrAssignableEmpty&&)                          = default;
  NotCopyConstructibleOrAssignableEmpty& operator=(const NotCopyConstructibleOrAssignableEmpty&)          = delete;
  NotCopyConstructibleOrAssignableEmpty& operator=(NotCopyConstructibleOrAssignableEmpty&&)               = default;
};

////////////////////////////////////////////////////////////////////////////////
// move assignable
////////////////////////////////////////////////////////////////////////////////
template <int Val>
struct NotTriviallyMoveAssignable
{
  int val_ = Val;

  __host__ __device__ constexpr NotTriviallyMoveAssignable(const int val) noexcept
      : val_(val)
  {}
  NotTriviallyMoveAssignable()                                  = default;
  NotTriviallyMoveAssignable(const NotTriviallyMoveAssignable&) = default;
  NotTriviallyMoveAssignable(NotTriviallyMoveAssignable&&)      = default;
  __host__ __device__ constexpr NotTriviallyMoveAssignable&
  operator=(const NotTriviallyMoveAssignable& other) noexcept(Val != MayThrow)
  {
    val_ = other.val_;
    return *this;
  }
  NotTriviallyMoveAssignable& operator=(NotTriviallyMoveAssignable&&) = default;
};

template <int Val>
struct NotTriviallyMoveAssignableEmpty
{
  __host__ __device__ constexpr NotTriviallyMoveAssignableEmpty(const int) noexcept {}
  NotTriviallyMoveAssignableEmpty()                                       = default;
  NotTriviallyMoveAssignableEmpty(const NotTriviallyMoveAssignableEmpty&) = default;
  NotTriviallyMoveAssignableEmpty(NotTriviallyMoveAssignableEmpty&&)      = default;
  __host__ __device__ constexpr NotTriviallyMoveAssignableEmpty&
  operator=(const NotTriviallyMoveAssignableEmpty& other) noexcept(Val != MayThrow)
  {
    return *this;
  }
  NotTriviallyMoveAssignableEmpty& operator=(NotTriviallyMoveAssignableEmpty&&) = default;

  __host__ __device__ friend constexpr bool operator==(const NotTriviallyMoveAssignableEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

template <int Val>
struct NotMoveAssignable
{
  int val_ = Val;

  __host__ __device__ constexpr NotMoveAssignable(const int val) noexcept
      : val_(val)
  {}
  NotMoveAssignable() = default;
  __host__ __device__ constexpr NotMoveAssignable(const NotMoveAssignable& other) noexcept(Val != MayThrow)
      : val_(other.val_)
  {}
  NotMoveAssignable(NotMoveAssignable&&)                 = default;
  NotMoveAssignable& operator=(const NotMoveAssignable&) = delete;
  NotMoveAssignable& operator=(NotMoveAssignable&&)      = default;
};

template <int Val>
struct NotMoveAssignableEmpty
{
  __host__ __device__ constexpr NotMoveAssignableEmpty(const int val) noexcept {}

  NotMoveAssignableEmpty() = default;
  __host__ __device__ constexpr NotMoveAssignableEmpty(const NotMoveAssignableEmpty&) noexcept(Val != MayThrow) {}
  NotMoveAssignableEmpty(NotMoveAssignableEmpty&&)                 = default;
  NotMoveAssignableEmpty& operator=(const NotMoveAssignableEmpty&) = delete;
  NotMoveAssignableEmpty& operator=(NotMoveAssignableEmpty&&)      = default;

  __host__ __device__ friend constexpr bool operator==(const NotMoveAssignableEmpty&, const int val) noexcept
  {
    return val == Val;
  }
};

template <class T>
__host__ __device__ constexpr bool operator==(const T& obj, const int val) noexcept
{
  return obj.val_ == val;
}

#endif // TEST_LIBCXX_ITERATORS_COMPRESSED_MOVABLE_BOX_TYPES_H
