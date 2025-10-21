#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <nv/target>

#include <unittest/unittest.h>

struct Foo
{
  _CCCL_DEVICE Foo()
      : destroyed{nullptr}
  {}

  Foo(const Foo&)            = delete;
  Foo& operator=(const Foo&) = delete;

  _CCCL_DEVICE ~Foo()
  {
    if (destroyed != nullptr)
    {
      *destroyed = true;
    }
  }

  bool* destroyed = nullptr;
};

void TestDeviceDeleteDestructorInvocation()
{
  thrust::device_ptr<Foo> foo_ptr = thrust::device_new<Foo>();

  thrust::device_vector<bool> destructor_flag(1, false);
  *thrust::device_ptr<bool*>(&foo_ptr.get()->destroyed) = destructor_flag.data().get();

  ASSERT_EQUAL(false, destructor_flag[0]);
  thrust::device_delete(foo_ptr);
  ASSERT_EQUAL(true, destructor_flag[0]);
}
DECLARE_UNITTEST(TestDeviceDeleteDestructorInvocation);

// based on: https://github.com/NVIDIA/cccl/issues/6132
struct base
{
  _CCCL_HOST_DEVICE virtual ~base()
  {
    if (base_destroyed != nullptr)
    {
      *base_destroyed = true;
    }
  }

  // Thrust does not support abstract class types inside vectors, so f() has a body instead of being abstract
  _CCCL_HOST_DEVICE virtual void f() {}

  bool* base_destroyed = nullptr;
};

struct derived : base
{
  _CCCL_HOST_DEVICE ~derived() override
  {
    if (derived_destroyed != nullptr)
    {
      *derived_destroyed = true;
    }
  }

  _CCCL_HOST_DEVICE void f() override {}

  bool* derived_destroyed = nullptr;
};

void TestDeviceDeleteVirtualDestructorInvocation()
{
  {
    thrust::device_ptr<derived> ptr = thrust::device_new<derived>();

    thrust::device_vector<bool> destructor_flags(2, false);
    *thrust::device_ptr<bool*>(&ptr.get()->base_destroyed)    = destructor_flags.data().get() + 0;
    *thrust::device_ptr<bool*>(&ptr.get()->derived_destroyed) = destructor_flags.data().get() + 1;

    thrust::device_ptr<derived> base_ptr = ptr;

    ASSERT_EQUAL(false, destructor_flags[0]);
    ASSERT_EQUAL(false, destructor_flags[1]);
    thrust::device_delete(base_ptr); // delete through the base pointer
    ASSERT_EQUAL(true, destructor_flags[0]);
    ASSERT_EQUAL(true, destructor_flags[1]);
  }
}
DECLARE_UNITTEST(TestDeviceDeleteVirtualDestructorInvocation);
