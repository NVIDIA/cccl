---
parent: Memory access properties
grand_parent: Extended API
nav_order: 4
---

# `cuda::associate_access_property`

```cuda
template <class T, class Property>
__host__ __device__ 
T* associate_access_property(T* ptr, Property prop);
```

**Preconditions**:
* if `Property` is [`cuda::access_property::shared`] then it must be valid to cast the generic pointer `ptr` to a pointer to the shared memory address space.
* if `Property` is one of [`cuda::access_property::global`], [`cuda::access_property::persisting`], [`cuda::access_property::normal`], or [`cuda::access_property::streaming`] then it must be valid to cast the generic pointer `ptr` to a pointer to the global memory address space.
* if `Property` is a [`cuda::access_property`] of "range" kind, then `ptr` must be in the valid range. 

**Mandates**: `Property` is convertible to [`cuda::access_property`].

**Effects**: no effects. 

**_Hint_**: to associate an access property with the returned pointer, such that subsequent memory operations with the returned pointer _or_ pointers derived from it _may_ apply the access property. 

  * The "association" is _not_ part of the value representation of the pointer.
  * The compiler is allowed to drop the association; it does not have a functional consequence.
  * The association _may_ hold through simple expressions, sequence of simple statements, or fully inlined function calls where the pointer value or C++ reference is provably unchanged; this includes offset pointers used for array access. 
  * The association is _not_ expected to hold through the ABI of an unknown function call, e.g., when the pointer is passed through a separately-compiled function interface, unless link-time optimizations are used.

**Note**: currently `associate_access_property` is ignored by nvcc and nvc++ on the host; but this might change any time.

# Example

```cuda
#include <cuda/cooperative_groups.h>
__global__ void memcpy(int const* in_, int* out) {
    int const* in = cuda::associate_access_property(in_, cuda::access_property::streaming{});
    auto idx = cooperative_groups::this_grid().thread_rank();

    __shared__ int shmem[N];
    shmem[threadIdx.x] = in[idx]; // streaming access

    // compute...
}
```

[`cuda::access_propety`]: {{ "extended_api/memory_access_properties/access_property.html" | relative_url }}
[`cuda::access_property::persisting`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::streaming`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::normal`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::global`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::shared`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
