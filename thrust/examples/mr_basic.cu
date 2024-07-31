#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/disjoint_pool.h>
#include <thrust/mr/new.h>
#include <thrust/mr/pool.h>

#include <cassert>

template <typename Vec>
void do_stuff_with_vector(typename Vec::allocator_type alloc)
{
  Vec v1(alloc);
  v1.push_back(1);
  assert(v1.back() == 1);

  Vec v2(alloc);
  v2 = v1;

  v1.swap(v2);

  v1.clear();
  v1.resize(2);
  assert(v1.size() == 2);
}

int main()
{
  thrust::mr::new_delete_resource memres;

  {
    // no virtual calls will be issued
    using Alloc = thrust::mr::allocator<int, thrust::mr::new_delete_resource>;
    Alloc alloc(&memres);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }

  {
    // virtual calls will be issued - wrapping in a polymorphic wrapper
    thrust::mr::polymorphic_adaptor_resource<void*> adaptor(&memres);
    using Alloc = thrust::mr::polymorphic_allocator<int, void*>;
    Alloc alloc(&adaptor);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }

  {
    // use the global device_ptr-flavored device memory resource
    using Resource = thrust::device_ptr_memory_resource<thrust::device_memory_resource>;
    thrust::mr::polymorphic_adaptor_resource<thrust::device_ptr<void>> adaptor(
      thrust::mr::get_global_resource<Resource>());
    using Alloc = thrust::mr::polymorphic_allocator<int, thrust::device_ptr<void>>;
    Alloc alloc(&adaptor);

    do_stuff_with_vector<thrust::device_vector<int, Alloc>>(alloc);
  }

  using Pool = thrust::mr::unsynchronized_pool_resource<thrust::mr::new_delete_resource>;
  Pool pool(&memres);
  {
    using Alloc = thrust::mr::allocator<int, Pool>;
    Alloc alloc(&pool);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }

  using DisjointPool =
    thrust::mr::disjoint_unsynchronized_pool_resource<thrust::mr::new_delete_resource, thrust::mr::new_delete_resource>;
  DisjointPool disjoint_pool(&memres, &memres);
  {
    using Alloc = thrust::mr::allocator<int, DisjointPool>;
    Alloc alloc(&disjoint_pool);

    do_stuff_with_vector<thrust::host_vector<int, Alloc>>(alloc);
  }
}
