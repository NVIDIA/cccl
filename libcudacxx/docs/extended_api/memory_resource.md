## memory_resource

The `<cuda/memory_resource>` header provides a standard C++ interface for *heterogeneous*, *stream-ordered* memory allocation tailored to the needs of CUDA C++ developers. This design builds off of the success of the [RAPIDS Memory Manager (RMM)](https://github.com/rapidsai/rmm) project and evolves the design based on lessons learned. `<cuda/memory_resource>` is not intended to replace RMM, but instead moves the definition of the memory allocation interface to a more centralized home in CCCL. RMM will remain as a collection of implementations of the `cuda::mr` interfaces.

We are still experimenting with the design, so for now the contents of `<cuda/memory_resource>` are only available if `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE` is defined.

At a high level, the header provides:

1. the [`cuda::get_property`] infrastructure to tag a user defined type with a given property;
2. the [`cuda::mr::{async}_resource` and `cuda::mr::{async}_resource_with`] concepts that provide proper constraints for arbitrary memory resources;
3. the [`cuda::mr::{async}_resource_ref`] is a type-erased memory resource wrapper that enables consumers to specify properties of resources that they expect.

These features are an evolution of `std::pmr::memory_resource` that was introduced in C++17. While `std::pmr::memory_resource` provides a polymorphic memory resource that can be adopted through inheritance, it is not properly suited for heterogeneous systems. With the current design it ranges from cumbersome to impossible to verify whether a memory resource provides allocations that are e.g. accessible on device, or whether it can utilize other allocation mechanisms.

To better support asynchronous CUDA [stream-ordered allocations](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator) libcu++ provides [`cuda::stream_ref`] as a wrapper around `cudaStream_t`. The definition of `cuda::stream_ref` can be found in the `<cuda/stream_ref>` header.

[`cuda::get_property`]: {{ "extended_api/memory_resource/properties.html" | relative_url }}
[`cuda::mr::{async}_resource` and `cuda::mr::{async}_resource_with`]: {{ "extended_api/memory_resource/resource.html" | relative_url }}
[`cuda::mr::{async}_resource_ref`]: {{ "extended_api/memory_resource/resource_ref.html" | relative_url }}
[`cuda::stream_ref`]: {{ "extended_api/memory_resource/stream_ref.html" | relative_url }}
