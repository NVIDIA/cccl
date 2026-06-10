#include <thrust/distance.h>
#include <thrust/functional.h>
#include <thrust/iterator/offset_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/std/iterator>

#include <unittest/unittest.h>

struct device_only_iterator
{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using difference_type   = cuda::std::ptrdiff_t;
  using value_type        = int;
  using pointer           = int*;
  using reference         = int&;

  _CCCL_HOST_DEVICE device_only_iterator(pointer ptr)
      : m_ptr(ptr)
  {}

  _CCCL_DEVICE reference operator*() const
  {
    return *m_ptr;
  }

  _CCCL_DEVICE device_only_iterator& operator++()
  {
    ++m_ptr;
    return *this;
  }

  _CCCL_DEVICE device_only_iterator operator++(int)
  {
    device_only_iterator tmp = *this;
    ++*this;
    return tmp;
  }

  _CCCL_DEVICE device_only_iterator& operator--()
  {
    --m_ptr;
    return *this;
  }

  _CCCL_DEVICE device_only_iterator operator--(int)
  {
    device_only_iterator tmp = *this;
    --*this;
    return tmp;
  }

  _CCCL_DEVICE device_only_iterator& operator+=(difference_type n)
  {
    m_ptr += n;
    return *this;
  }

  _CCCL_DEVICE friend device_only_iterator operator+(device_only_iterator it, difference_type d)
  {
    return it += d;
  }

  _CCCL_DEVICE friend difference_type operator-(const device_only_iterator& a, const device_only_iterator& b)
  {
    return a.m_ptr - b.m_ptr;
  }

  _CCCL_DEVICE friend bool operator==(const device_only_iterator& a, const device_only_iterator& b)
  {
    return a.m_ptr == b.m_ptr;
  }

  _CCCL_DEVICE friend bool operator!=(const device_only_iterator& a, const device_only_iterator& b)
  {
    return a.m_ptr != b.m_ptr;
  }

private:
  pointer m_ptr;
};

_CCCL_HOST_DEVICE void TestOffsetIteratorBoth(thrust::offset_iterator<device_only_iterator> iter)
{
  assert(iter.offset() == 0);
  ++iter;
  assert(iter.offset() == 1);
  iter++;
  assert(iter.offset() == 2);
  --iter;
  assert(iter.offset() == 1);
  iter--;
  assert(iter.offset() == 0);
  iter += 100;
  assert(iter.offset() == 100);
}

__global__ void TestOffsetIteratorDevice(thrust::offset_iterator<device_only_iterator> iter)
{
  TestOffsetIteratorBoth(iter);

  // access
  assert(*iter == 1);

  auto iter2 = iter;
  iter2 += 3;
  assert(*iter2 == 1);

  // difference
  assert(iter2 - iter == 3);

  // comparison
  assert(!(iter2 == iter));
  assert(iter2 != iter);
}

void TestOffsetIteratorWithDeviceOnlyIterator()
{
  thrust::device_vector<int> v{1, 2, 3, 4, 5};
  device_only_iterator base(thrust::raw_pointer_cast(v.data()));
  thrust::offset_iterator iter(base);
  TestOffsetIteratorBoth(iter);
  TestOffsetIteratorDevice<<<1, 1>>>(iter);
}
DECLARE_UNITTEST(TestOffsetIteratorWithDeviceOnlyIterator);
