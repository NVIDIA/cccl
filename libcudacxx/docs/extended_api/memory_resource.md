## memory_resource

The `cuda::memory_resource` header provides two main features:
1. the `cuda::resource` and `cuda::resource_with` concepts that provide proper constraints for arbitrary memory resoures
2. the `cuda::resource_ref` is a type-erased memory resource wrapper that enables consumers to specify properties of resources that they expect.

These features are an evolution of `std::pmr::memory_resource` that was introduced in C++17. While `std::pmr::memory_resource` provides a polymorphic memory resource that can be adopted through inheritance, it is not properly suited for heterogeneous systems. With the current design it ranges from cumbersome to impossible to verify whether a memory resource provides allocations that are e.g. accessible on device, or whether it can utilize other allocation mechanisms.

### Properties

Modern C++ programs use the type system to verify statically known properties of the code. It is undesirable to use run-time assertions to verify whether a function that accesses device memory is passed a memory resource that provides device-accessible allocations. A property system is provided for this purpose.

To tell the compiler that memory provided by `my_memory_resource` is device accessible, use the `cuda::mr::device_accessible` tag type with a free function `get_property` to declare that the resource provides device-accessible memory.
```c++
struct my_memory_resource {
    friend constexpr void get_property(const my_memory_resource&, cuda::mr::device_accessible) noexcept {}
};
```

A library can constrain interfaces with `cuda::has_property` to require that a passed memory resource provides the right kind of memory (See [here](https://godbolt.org/z/5hjoEnerb) for a minimal Compiler Explorer example).
```c++
template<class MemoryResource>
    requires cuda::has_property<MemoryResource, cuda::mr::device_accessible>
void function_that_dispatches_to_device(MemoryResource& resource);
```

If C++20 is not available, the function can instead be constrained via SFINAE (See [here](https://godbolt.org/z/11sGbr333) for a minimal Compiler Explorer example).
```c++
template<class MemoryResource, class = cuda::std::enable_if_t<cuda::has_property<MemoryResource, cuda::mr::device_accessible>>>
void function_that_dispatches_to_device(MemoryResource& resource);
```

For now, libcu++ provides two commonly used properties: `cuda::mr::device_accessible` and `cuda::mr::host_accessible`. More properties may be added as the library and the hardware capabilities evolve. However, a user library is free to define as many properties as needed to fully cover its API surface.

Note that the libcu++ provided properties are stateless. However, properties can also provide stateful information that is retrieved via the `get_property` free function. In order to communicate the desired type of the carried state, a stateful property must define the `value_type` alias. A library can constrain interfaces that require a stateful property with `cuda::has_property_with` as shown in the example below (See [here](https://godbolt.org/z/11sGbr333) for a minimal Compiler Explorer example).
```c++
struct required_alignment{
    using value_type = std::size_t;
};
struct my_memory_resource {
    friend constexpr std::size_t get_property(const my_memory_resource& resource, required_alignment) noexcept { return resource.required_alignment; }

    std::size_t required_alignment;
};
static_assert(cuda::has_property_with<my_memory_resource, required_alignment, std::size_t>);

template<class MemoryResource>
void* allocate_check_alignment(MemoryResource& resource, std::size_t size) {
    if constexpr(cuda::has_property_with<MemoryResource, required_alignment, std::size_t>) {
        return resource.allocate(size, get_property(resource, required_alignment{}));
    } else {
        // Use default alignment
        return resource.allocate(size, 42);
    }
}
```

### The `resource` concept

The `std::pmr::memory_resource` feature provides only a single `allocate` interface, which is sufficient for homogeneous memory systems. However, CUDA provides both synchronous and [stream-ordered allocation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-ordered-memory-allocator).


With `std::pmr::memory_resource` there is no way to tell whether a memory resource can utilize stream-ordered allocations. Even if the application knows it can, there is no way to properly tell the memory resource to use stream-ordered allocation. Ideally, this should not be something discovered through an assert at run time, but should be checked by the compiler.

The `cuda::mr::resource` concept provides basic type checks to ensure that a given memory resource provides the expected `allocate`/`deallocate` interface and is also equality comparable, which covers the whole API surface of `std::pmr::memory_resource`. See below for different memory resources and potential pitfals.

To demonstrate, the following example defines several resources, only some of which are valid implementations of the `cuda::mr::resource` concept. The `static_assertion`s will result in compile-time errors for the invalid resources.
```c++
struct valid_resource {
  void* allocate(std::size_t, std::size_t) { return nullptr; }
  void deallocate(void*, std::size_t, std::size_t) {}
  bool operator==(const valid_resource&) const { return true; }
  // NOTE: C++20 thankfully added operator rewrite rules so defining operator!= is not required.
  // However, if compiled with C++14 / C++17, operator != must also be defined.
  bool operator!=(const valid_resource&) const { return false; }
};
static_assert(cuda::mr::resource<valid_resource>, "");

struct invalid_argument {};
struct invalid_allocate_argument {
  void* allocate(invalid_argument, std::size_t) { return nullptr; }
  void deallocate(void*, std::size_t, std::size_t) {}
  bool operator==(const invalid_allocate_argument&) { return true; }
};
static_assert(!cuda::mr::resource<invalid_allocate_argument>, "");

struct invalid_allocate_return {
  int allocate(std::size_t, std::size_t) { return 42; }
  void deallocate(void*, std::size_t, std::size_t) {}
  bool operator==(const invalid_allocate_return&) { return true; }
};
static_assert(!cuda::mr::resource<invalid_allocate_return>, "");

struct invalid_deallocate_argument {
  void* allocate(std::size_t, std::size_t) { return nullptr; }
  void deallocate(void*, invalid_argument, std::size_t) {}
  bool operator==(const invalid_deallocate_argument&) { return true; }
};
static_assert(!cuda::mr::resource<invalid_deallocate_argument>, "");

struct non_comparable {
  void* allocate(std::size_t, std::size_t) { return nullptr; }
  void deallocate(void*, std::size_t, std::size_t) {}
};
static_assert(!cuda::mr::resource<non_comparable>, "");

struct non_eq_comparable {
  void* allocate(std::size_t, std::size_t) { return nullptr; }
  void deallocate(void*, std::size_t, std::size_t) {}
  bool operator!=(const non_eq_comparable&) { return false; }
};
static_assert(!cuda::mr::resource<non_eq_comparable>, "");
```

In addition to the `std::pmr::memory_resource` interface the `cuda::mr::async_resource` concept verifies that a memory resource also satisfies the `allocate_async` / `deallocate_async` interface. Requiring both the PMR interface and the async interface is a deliberate design decision.
```c++
struct valid_resource {
  void* allocate(std::size_t, std::size_t) { return nullptr; }
  void deallocate(void*, std::size_t, std::size_t) {}
  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref) { return nullptr; }
  void deallocate_async(void*, std::size_t, std::size_t, cuda::stream_ref) {}
  bool operator==(const valid_resource&) const { return true; }
  bool operator!=(const valid_resource&) const { return false; }
};
static_assert(cuda::mr::async_resource<valid_resource>, "");
```

A library can easily decide whether to use the async interface:
```c++
template<class MemoryResource>
    requires cuda::mr::resource<MemoryResource>
void* maybe_allocate_async(MemoryResource& resource, std::size_t size, std::size_t align, cuda::stream_ref stream) {
    if constexpr(cuda::mr::async_resource<MemoryResource>) {
        return resource.allocate_async(size, align, stream);
    } else {
        return resource.allocate(size, align);
    }
}
```

### Putting them together

Applications and libraries may want to combine type checks for arbitrary properties with the `{async_}resource` concept. The `{async_}resource_with` concept allows checking resources for arbitrary properties.
```c++
struct required_alignment{
    using value_type = std::size_t;
};
struct my_memory_resource {
    void* allocate(std::size_t, std::size_t) { return nullptr; }
    void deallocate(void*, std::size_t, std::size_t) {}
    bool operator==(const my_memory_resource&) const { return true; }
    bool operator!=(const my_memory_resource&) const { return false; }

    friend constexpr std::size_t get_property(const my_memory_resource& resource, required_alignment) noexcept { return resource.required_alignment; }

    std::size_t required_alignment;
};

template<class MemoryResource>
    requires cuda::mr::resource<MemoryResource>
void* maybe_allocate_async_check_alignment(MemoryResource& resource, std::size_t size, cuda::stream_ref stream) {
    if constexpr(cuda::mr::async_resource_with<MemoryResource, required_alignment>) {
        return resource.allocate_async(size, get_property(resource, required_alignment), stream);
    } else if constexpr (cuda::mr::async_resource<MemoryResource>) {
        return resource.allocate_async(size, my_default_alignment, stream);
    } else if constexpr (cuda::mr::resource_with<MemoryResource, required_alignment>) {
        return resource.allocate(size, get_property(resource, required_alignment));
    } else {
        return resource.allocate(size, my_default_alignment);
    }
}

// Potentially more concise
template<class MemoryResource>
    requires cuda::mr::resource<MemoryResource>
void* maybe_allocate_async_check_alignment2(MemoryResource& resource, std::size_t size, cuda::stream_ref stream) {
    constexpr std::size_t align = cuda::mr::resource_with<MemoryResource, required_alignment>
                                ? get_property(resource, required_alignment)
                                : my_default_alignment;
    if constexpr(cuda::mr::async_resource<MemoryResource>) {
        return resource.allocate_async(size, align, stream);
    } else {
        return resource.allocate(size, align);
    }
}
```

### `resource_ref`: a type-constrained resource wrapper

With the property design shown above a library has flexibility in checking constraints and querying custom properties. However, there is also a cost in providing function templates for a potentially wide range of inputs. Depending on the number of different memory resources, both compile time and binary size might increase considerably.

The type-erased `resource_ref` and `async_resource_ref` resource wrappers aid in efficiently coalesce such APIs into a single function.
```c++
void* do_allocate_async(cuda::mr::async_resource_ref<> resource, std::size_t size, std::size_t align, cuda::stream_ref stream) {
    return resource.allocate_async(size, align, stream);
}

my_async_memory_resource resource;
my_async_memory_resource* pointer_to_resource = &resource;

void* from_reference = do_allocate_async(resource, 1337, 256, cuda::stream_ref{});
void* from_ptr = do_allocate_async(pointer_to_resource, 1337, 256, cuda::stream_ref{});
```

Note that `do_allocate_async` is not a template anymore but a plain old function. The wrapper `cuda::mr::{async_}resource_ref<>` is constructible from any non-const reference or pointer to a memory resource that satisfies `cuda::mr::{async_}resource`.

Properties may also be passed to `cuda::mr::{async_}resource_ref` just as with `cuda::mr::resource_with`.
```c++
struct required_alignment{};
void* do_allocate_async_with_alignment(cuda::mr::async_resource_ref<required_alignment> resource, std::size_t size, cuda::stream_ref stream) {
    return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
}
```

However, the type erasure comes with the cost that arbitrary properties cannot be queried:
```c++
struct required_alignment{};
void* buggy_allocate_async_with_alignment(cuda::mr::async_resource_ref<> resource, std::size_t size, cuda::stream_ref stream) {
    if constexpr (cuda::has_property<required_alignment>) { // BUG: This will always be false
        return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
    } else {
        return resource.allocate_async(size, my_default_alignment, stream);
    }
}
```

So, choose wisely. If your library has a well-defined set of fixed properties that you expect to always be available, then `cuda::mr::{async_}resource_ref` is an amazing tool to improve compile times and binary size. If you need to your interface to be flexible then constraining trough `cuda::mr::{async_}resource_with` is the proper solution.
