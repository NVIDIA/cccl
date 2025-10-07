#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <nv/target>

#include <unittest/unittest.h>

struct Foo
{
  _CCCL_HOST_DEVICE Foo()
      : set_me_upon_destruction{nullptr}
  {}

  _CCCL_HOST_DEVICE ~Foo()
  {
    NV_IF_TARGET(NV_IS_DEVICE, (if (set_me_upon_destruction != nullptr) { *set_me_upon_destruction = true; }));
  }

  bool* set_me_upon_destruction;
};

void TestDeviceDeleteDestructorInvocation()
{
  thrust::device_vector<bool> destructor_flag(1, false);

  thrust::device_ptr<Foo> foo_ptr = thrust::device_new<Foo>();

  Foo exemplar;
  exemplar.set_me_upon_destruction = thrust::raw_pointer_cast(&destructor_flag[0]);
  *foo_ptr                         = exemplar;

  ASSERT_EQUAL(false, destructor_flag[0]);

  thrust::device_delete(foo_ptr);

  ASSERT_EQUAL(true, destructor_flag[0]);
}
DECLARE_UNITTEST(TestDeviceDeleteDestructorInvocation);

// based on: https://github.com/NVIDIA/cccl/issues/6132
struct base
{
  _CCCL_HOST_DEVICE virtual void f() = 0;
  _CCCL_HOST_DEVICE virtual ~base() {}
};

struct derived : public base
{
  _CCCL_HOST_DEVICE void f() override {}
};

void TestDeviceDeleteVirtualDestructorInvocation()
{
  thrust::device_ptr<base> ptr = thrust::device_new<derived>();
  thrust::device_delete(ptr);
}
DECLARE_UNITTEST(TestDeviceDeleteVirtualDestructorInvocation);
