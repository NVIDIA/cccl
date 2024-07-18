#pragma once

#include <thrust/detail/static_assert.h>

#include <string>
#undef THRUST_STATIC_ASSERT
#undef THRUST_STATIC_ASSERT_MSG

#define THRUST_STATIC_ASSERT(B)          unittest::assert_static((B), __FILE__, __LINE__);
#define THRUST_STATIC_ASSERT_MSG(B, msg) unittest::assert_static((B), __FILE__, __LINE__);

namespace unittest
{
_CCCL_HOST_DEVICE void assert_static(bool condition, const char* filename, int lineno);
}

#include <thrust/device_delete.h>
#include <thrust/device_new.h>

#include <nv/target>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

#  define ASSERT_STATIC_ASSERT(X)                                                              \
    {                                                                                          \
      bool triggered                      = false;                                             \
      using ex_t                          = unittest::static_assert_exception;                 \
      thrust::device_ptr<ex_t> device_ptr = thrust::device_new<ex_t>();                        \
      ex_t* raw_ptr                       = thrust::raw_pointer_cast(device_ptr);              \
      ::cudaMemcpyToSymbol(unittest::detail::device_exception, &raw_ptr, sizeof(ex_t*));       \
      try                                                                                      \
      {                                                                                        \
        X;                                                                                     \
      }                                                                                        \
      catch (ex_t)                                                                             \
      {                                                                                        \
        triggered = true;                                                                      \
      }                                                                                        \
      if (!triggered)                                                                          \
      {                                                                                        \
        triggered = static_cast<ex_t>(*device_ptr).triggered;                                  \
      }                                                                                        \
      thrust::device_free(device_ptr);                                                         \
      raw_ptr = nullptr;                                                                       \
      ::cudaMemcpyToSymbol(unittest::detail::device_exception, &raw_ptr, sizeof(ex_t*));       \
      if (!triggered)                                                                          \
      {                                                                                        \
        unittest::UnitTestFailure f;                                                           \
        f << "[" << __FILE__ << ":" << __LINE__ << "] did not trigger a THRUST_STATIC_ASSERT"; \
        throw f;                                                                               \
      }                                                                                        \
    }

#else

#  define ASSERT_STATIC_ASSERT(X)                                                              \
    {                                                                                          \
      bool triggered = false;                                                                  \
      using ex_t     = unittest::static_assert_exception;                                      \
      try                                                                                      \
      {                                                                                        \
        X;                                                                                     \
      }                                                                                        \
      catch (ex_t)                                                                             \
      {                                                                                        \
        triggered = true;                                                                      \
      }                                                                                        \
      if (!triggered)                                                                          \
      {                                                                                        \
        unittest::UnitTestFailure f;                                                           \
        f << "[" << __FILE__ << ":" << __LINE__ << "] did not trigger a THRUST_STATIC_ASSERT"; \
        throw f;                                                                               \
      }                                                                                        \
    }

#endif

namespace unittest
{
class static_assert_exception
{
public:
  _CCCL_HOST_DEVICE static_assert_exception()
      : triggered(false)
  {}

  _CCCL_HOST_DEVICE static_assert_exception(const char* filename, int lineno)
      : triggered(true)
      , filename(filename)
      , lineno(lineno)
  {}

  bool triggered;
  const char* filename;
  int lineno;
};

namespace detail
{
#if defined(_CCCL_COMPILER_GCC) || defined(_CCCL_COMPILER_CLANG)
__attribute__((used))
#endif
_CCCL_DEVICE static static_assert_exception* device_exception = nullptr;
} // namespace detail

_CCCL_HOST_DEVICE void assert_static(bool condition, const char* filename, int lineno)
{
  if (!condition)
  {
    static_assert_exception ex(filename, lineno);

    NV_IF_TARGET(NV_IS_DEVICE, (*detail::device_exception = ex;), (throw ex;));
  }
}
} // namespace unittest
