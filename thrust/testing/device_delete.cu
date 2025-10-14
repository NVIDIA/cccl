#include <thrust/device_delete.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <nv/target>

#include <unittest/unittest.h>

struct Foo
{
  _CCCL_DEVICE Foo()
      : set_me_upon_destruction{nullptr}
  {}

  Foo(const Foo&)            = delete;
  Foo& operator=(const Foo&) = delete;

  _CCCL_DEVICE ~Foo()
  {
    if (set_me_upon_destruction != nullptr)
    {
      *set_me_upon_destruction = true;
    }
  }

  bool* set_me_upon_destruction;
};

void TestDeviceDeleteDestructorInvocation()
{
  thrust::device_vector<bool> destructor_flag(1, false);

  thrust::device_ptr<Foo> foo_ptr = thrust::device_new<Foo>();
  *thrust::device_ptr<bool*>(&foo_ptr.get()->set_me_upon_destruction) =
    thrust::raw_pointer_cast(destructor_flag.data());

  ASSERT_EQUAL(false, destructor_flag[0]);
  thrust::device_delete(foo_ptr);
  ASSERT_EQUAL(true, destructor_flag[0]);
}
DECLARE_UNITTEST(TestDeviceDeleteDestructorInvocation);
