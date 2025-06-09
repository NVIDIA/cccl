#include "test_workarounds.h"

#ifndef DEFINE_BASE
#  define DEFINE_BASE(Name) ::ArchetypeBases::NullBase
#  define DEFINE_INIT_LIST : ::ArchetypeBases::NullBase()
#endif
#ifndef DEFINE_EXPLICIT
#  define DEFINE_EXPLICIT
#endif
#ifndef DEFINE_NOEXCEPT
#  define DEFINE_NOEXCEPT
#endif
#ifndef DEFINE_CONSTEXPR
#  if defined(TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR) || defined(TEST_WORKAROUND_NO_ADDRESSOF)
#    define DEFINE_CONSTEXPR
#  else // TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR
#    define DEFINE_CONSTEXPR constexpr
#  endif // TEST_WORKAROUND_EDG_EXPLICIT_CONSTEXPR
#endif
#ifndef DEFINE_ASSIGN_CONSTEXPR
#  define DEFINE_ASSIGN_CONSTEXPR DEFINE_CONSTEXPR
#endif
#ifndef DEFINE_DEFAULT_CONSTEXPR
#  if TEST_COMPILER(NVRTC)
#    define DEFINE_DEFAULT_CONSTEXPR
#  else
#    define DEFINE_DEFAULT_CONSTEXPR DEFINE_CONSTEXPR
#  endif
#endif
#ifndef DEFINE_CTOR
#  define DEFINE_CTOR = default
#  undef DEFINE_INIT_LIST // defaulted constructors do not require explicit initializers for the base class
#endif
#ifndef DEFINE_CTOR_ANNOTATIONS
#  define DEFINE_CTOR_ANNOTATIONS
#endif
#ifndef DEFINE_DEFAULT_CTOR
#  define DEFINE_DEFAULT_CTOR DEFINE_CTOR
#endif
#ifndef DEFINE_ASSIGN
#  define DEFINE_ASSIGN = default
#endif
#ifndef DEFINE_ASSIGN_ANNOTATIONS
#  define DEFINE_ASSIGN_ANNOTATIONS
#endif
#ifndef DEFINE_DTOR
#  define DEFINE_DTOR(Name)
#endif
#ifndef DEFINE_INIT_LIST
#  define DEFINE_INIT_LIST
#endif

struct AllCtors : DEFINE_BASE(AllCtors)
{
  using Base = DEFINE_BASE(AllCtors);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  AllCtors(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  using Base::operator=;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR AllCtors() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR AllCtors(AllCtors const&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR AllCtors(AllCtors&&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_ASSIGN_ANNOTATIONS DEFINE_ASSIGN_CONSTEXPR AllCtors& operator=(AllCtors const&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_ASSIGN_ANNOTATIONS DEFINE_ASSIGN_CONSTEXPR AllCtors& operator=(AllCtors&&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_DTOR(AllCtors)
};

struct NoCtors : DEFINE_BASE(NoCtors)
{
  using Base                                              = DEFINE_BASE(NoCtors);
  DEFINE_EXPLICIT NoCtors() DEFINE_NOEXCEPT               = delete;
  DEFINE_EXPLICIT NoCtors(NoCtors const&) DEFINE_NOEXCEPT = delete;
  NoCtors& operator=(NoCtors const&) DEFINE_NOEXCEPT      = delete;
  DEFINE_DTOR(NoCtors)
};

struct NoDefault : DEFINE_BASE(NoDefault)
{
  using Base = DEFINE_BASE(NoDefault);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  NoDefault(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_EXPLICIT DEFINE_CONSTEXPR NoDefault() DEFINE_NOEXCEPT = delete;
  DEFINE_DTOR(NoDefault)
};

struct DefaultOnly : DEFINE_BASE(DefaultOnly)
{
  using Base = DEFINE_BASE(DefaultOnly);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  DefaultOnly(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR DefaultOnly() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DefaultOnly(DefaultOnly const&) DEFINE_NOEXCEPT            = delete;
  DefaultOnly& operator=(DefaultOnly const&) DEFINE_NOEXCEPT = delete;
  DEFINE_DTOR(DefaultOnly)
};

struct Copyable : DEFINE_BASE(Copyable)
{
  using Base = DEFINE_BASE(Copyable);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  Copyable(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR Copyable() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR Copyable(Copyable const&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_ASSIGN_ANNOTATIONS Copyable& operator=(Copyable const&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_DTOR(Copyable)
};

struct CopyOnly : DEFINE_BASE(CopyOnly)
{
  using Base = DEFINE_BASE(CopyOnly);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  CopyOnly(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR CopyOnly() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyOnly(CopyOnly const&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyOnly(CopyOnly&&) DEFINE_NOEXCEPT = delete;
  DEFINE_ASSIGN_ANNOTATIONS CopyOnly& operator=(CopyOnly const&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  CopyOnly& operator=(CopyOnly&&) DEFINE_NOEXCEPT = delete;
  DEFINE_DTOR(CopyOnly)
};

struct NonCopyable : DEFINE_BASE(NonCopyable)
{
  using Base = DEFINE_BASE(NonCopyable);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  NonCopyable(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR NonCopyable() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DEFINE_EXPLICIT DEFINE_CONSTEXPR NonCopyable(NonCopyable const&) DEFINE_NOEXCEPT = delete;
  NonCopyable& operator=(NonCopyable const&) DEFINE_NOEXCEPT                       = delete;
  DEFINE_DTOR(NonCopyable)
};

struct MoveOnly : DEFINE_BASE(MoveOnly)
{
  using Base = DEFINE_BASE(MoveOnly);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  MoveOnly(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR MoveOnly() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR MoveOnly(MoveOnly&&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_ASSIGN_ANNOTATIONS MoveOnly& operator=(MoveOnly&&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_DTOR(MoveOnly)
};

struct CopyAssignable : DEFINE_BASE(CopyAssignable)
{
  using Base = DEFINE_BASE(CopyAssignable);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  CopyAssignable(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyAssignable() DEFINE_NOEXCEPT = delete;
  DEFINE_ASSIGN_ANNOTATIONS CopyAssignable& operator=(CopyAssignable const&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_DTOR(CopyAssignable)
};

struct CopyAssignOnly : DEFINE_BASE(CopyAssignOnly)
{
  using Base = DEFINE_BASE(CopyAssignOnly);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  CopyAssignOnly(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_EXPLICIT DEFINE_CONSTEXPR CopyAssignOnly() DEFINE_NOEXCEPT = delete;
  DEFINE_ASSIGN_ANNOTATIONS CopyAssignOnly& operator=(CopyAssignOnly const&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  CopyAssignOnly& operator=(CopyAssignOnly&&) DEFINE_NOEXCEPT = delete;
  DEFINE_DTOR(CopyAssignOnly)
};

struct MoveAssignOnly : DEFINE_BASE(MoveAssignOnly)
{
  using Base = DEFINE_BASE(MoveAssignOnly);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621
  template <class... Args, typename = decltype(Base(cuda::std::declval<Args>()...))>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR
  MoveAssignOnly(Args&&... args) noexcept(noexcept(Base(cuda::std::declval<Args>()...)))
      : Base(cuda::std::forward<Args>(args)...)
  {}
#else
  using Base::Base;
#endif
  DEFINE_EXPLICIT DEFINE_CONSTEXPR MoveAssignOnly() DEFINE_NOEXCEPT = delete;
  MoveAssignOnly& operator=(MoveAssignOnly const&) DEFINE_NOEXCEPT  = delete;
  DEFINE_ASSIGN_ANNOTATIONS MoveAssignOnly& operator=(MoveAssignOnly&&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_DTOR(MoveAssignOnly)
};

struct ConvertingType : DEFINE_BASE(ConvertingType)
{
  using Base = DEFINE_BASE(ConvertingType);
#if TEST_COMPILER(NVRTC, <, 12, 5) // nvbug3961621, this one is special because of the converting ctor
#else
  using Base::Base;
#endif
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_DEFAULT_CONSTEXPR ConvertingType() DEFINE_NOEXCEPT DEFINE_DEFAULT_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType(ConvertingType const&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_CTOR_ANNOTATIONS DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType(ConvertingType&&)
    DEFINE_NOEXCEPT DEFINE_INIT_LIST DEFINE_CTOR;
  DEFINE_ASSIGN_ANNOTATIONS ConvertingType& operator=(ConvertingType const&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  DEFINE_ASSIGN_ANNOTATIONS ConvertingType& operator=(ConvertingType&&) DEFINE_NOEXCEPT DEFINE_ASSIGN;
  template <class... Args>
  __host__ __device__ DEFINE_EXPLICIT DEFINE_CONSTEXPR ConvertingType(Args&&...) DEFINE_NOEXCEPT DEFINE_INIT_LIST
  {}
  template <class Arg>
  __host__ __device__ ConvertingType& operator=(Arg&&) DEFINE_NOEXCEPT
  {
    return *this;
  }
  DEFINE_DTOR(ConvertingType)
};

template <template <class...> class List>
using ApplyTypes =
  List<AllCtors,
       NoCtors,
       NoDefault,
       DefaultOnly,
       Copyable,
       CopyOnly,
       NonCopyable,
       MoveOnly,
       CopyAssignable,
       CopyAssignOnly,
       MoveAssignOnly,
       ConvertingType>;

#undef DEFINE_BASE
#undef DEFINE_EXPLICIT
#undef DEFINE_NOEXCEPT
#undef DEFINE_CONSTEXPR
#undef DEFINE_ASSIGN_CONSTEXPR
#undef DEFINE_CTOR
#undef DEFINE_CTOR_ANNOTATIONS
#undef DEFINE_DEFAULT_CTOR
#undef DEFINE_ASSIGN
#undef DEFINE_ASSIGN_ANNOTATIONS
#undef DEFINE_DTOR
#undef DEFINE_INIT_LIST
