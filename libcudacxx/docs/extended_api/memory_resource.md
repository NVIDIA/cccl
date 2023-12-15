## memory_resource

The `cuda::memory_resource` header provides two main features:
* The `cuda::resource` and `cuda::resource_with` concepts that provide proper constraints for arbitrary memory resoures
* The `cuda::resource_ref` resource wrapper as a type erased memory resource, that allows for simple efficient interfaces

These features are an evolution of `std::pmr::memory_resource` that was introduced in C++17. While `std::pmr::memory_resource` provides a polymorphic memory resource that can be adopted through inheritance, it is not properly suited for heterogeneous systems. With the current design it ranges from cumbersome to impossible to verify whether a memory resource provides allocations that are e.g. accessible on device, or whether it can utilize other allocation mechanisms.

### Properties

With C++ we want the type system to verify statically known properties of our code. If a memory resource provides allocations that are device accessible, we do not want to rely on runtime asserts to check whether we can indeed call a device function with that memory. Therefore, we are introducing a property system that is similar to rust traits.

Let's assume we want to tell the compiler that memory provided by `my_memory_resource` is indeed device accessible. For that we introduce a tag type:
```c++
namespace cuda::mr {
    struct device_accessible{};
}
```

We can utilize a free function `get_property` to mark `my_memory_resource` as providing device accessible memory
```c++
struct my_memory_resource {
    friend constexpr void get_property(const my_memory_resource&, cuda::mr::device_accessible) noexcept {}
};
```

A library can now constrain their interface with `cuda::mr::has_property` to verify that the passed memory resource is providing the right kind of memory (See [here](https://godbolt.org/z/MTTd6dqe1) for a minimal godbolt example)
```c++
template<class MemoryResource>
    requires cuda::mr::has_property<MemoryResource, cuda::mr::device_accessible>
void function_that_dispatches_to_device(MemoryResource& resource);
```

For now libcu++ provides two standard properties `cuda::mr::device_accessible` and `cuda::mr::host_accessible`. We envision to add a few more as the library and the hardware capabilities evolve. However, a user library is free to define as many properties as they want to fully cover their respective API surface.

Note that in the above example we went with a simple stateless property. However, it is valid to make them stateful to propagate more information.
```c++
struct required_alignment{};
struct my_memory_resource {
    friend constexpr std::size_t get_property(const my_memory_resource& resource, required_alignment) noexcept { return resource.required_alignment; }

    std::size_t required_alignment;
};

template<class MemoryResource>
void* allocate_check_alignment(MemoryResource& resource, std::size_t size) {
    if constexpr(cuda::mr::has_property<desired_alignment>) {
        return resource.allocate(size, get_property(resource, required_alignment));
    } else {
        // Use default alignment
        return resource.allocate(size, my_default_alignment);
    }
}
```

### The `resource` concept

The classical `std::pmr::memory_resource` feature only knows a single `allocate` interface, which is totally sufficient for homogeneous memory systems. However, we recently provided [stream ordered allocation](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/) interfaces that allow highly efficient asyncronous interfaces.

With `std::pmr::memory_resource` there is no possibility to differentiate whether a memory resource can utilize stream odered allocations and even if we would know, there is no way to properly tell the memory resource to use it or not. Again, this should not be something we discover through an assert at runtime but should be checked by the compiler.

Here, the `cuda::mr::resource` concept provides basic type checks to ensure that a given memory resource provides the expected `allocate`/`deallocate` interface and is also equality comparable, which covers the whole API surface of `std::pmr::memory_resource`. See below for different memory resources and potential pitfals
```c++
struct valid_resource {
  void* allocate(std::size_t, std::size_t) { return nullptr; }
  void deallocate(void*, std::size_t, std::size_t) {}
  bool operator==(const valid_resource&) const { return true; }
  // NOTE: C++20 thankfully added operator rewrite rules so that we do not need to define operator!= there.
  // However, if compiled with C++14 / C++17 we also require operator != to be defined.
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

In addition to the classical `std::pmr::memory_resource` interface the `cuda::mr::async_resource` concept also verifies that the memory resource satisfies the `allocate_async` / `deallocate_async` interface. We made the conscious decision that we always require classical interface in addition to the async one.
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

A library can now easily decide whether it wants to use the async interface:
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

Obviously, we want to combine type checks for arbitrary properties with the `{async_}resource` concept. This is what the `{async_}resource_with` concept is for. A library can now define arbitrary properties:
```c++
struct required_alignment{};
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
    if constexpr(cuda::mr::async_resource_with<MemoryResource, desired_alignment>) {
        return resource.allocate_async(size, get_property(resource, required_alignment), stream);
    } else if constexpr (cuda::mr::async_resource<MemoryResource>) {
        return resource.allocate_async(size, my_default_alignment, stream);
    } else if constexpr (cuda::mr::resource_with<MemoryResource, desired_alignment>) {
        return resource.allocate(size, get_property(resource, required_alignment));
    } else {
        return resource.allocate(size, my_default_alignment);
    }
}

// Potentially more consice
template<class MemoryResource>
    requires cuda::mr::resource<MemoryResource>
void* maybe_allocate_async_check_alignment2(MemoryResource& resource, std::size_t size, cuda::stream_ref stream) {
    constexpr std::size_t align = cuda::mr::resource_with<MemoryResource, desired_alignment>
                                ? get_property(resource, required_alignment)
                                : my_default_alignment;
    if constexpr(cuda::mr::async_resource<MemoryResource>) {
        return resource.allocate_async(size, align, stream);
    } else {
        return resource.allocate(size, align);
    }
}
```

### `resource_ref` a type constrained resource wrapper

With the property design shown above a library has maximal flexibility regarding constraints checks and querying for any custom property. However, there is also a cost in providing function templates for a potentially wide range of inputs. Depending on the number of different memory resources both compile times and binary size might increase considerably.

To efficiently coalesce such APIs into a single function we provide the type erased `resource_ref` and `async_resource_ref` resource wrappers.
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

Naturally we also support passing properties to `cuda::mr::{async_}resource_ref` just as we did with `cuda::mr::resource_with`
```c++
struct required_alignment{};
void* do_allocate_async_with_alignment(cuda::mr::async_resource_ref<required_alignment> resource, std::size_t size, cuda::stream_ref stream) {
    return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
}
```

However, the type erasure comes with a cost. We cannot query arbitrary properties anymore:
```c++
struct required_alignment{};
void* buggy_allocate_async_with_alignment(cuda::mr::async_resource_ref<> resource, std::size_t size, cuda::stream_ref stream) {
    if constexpr (cuda::mr::has_property<required_alignment>) { // BUG: This will always be false
        return resource.allocate_async(size, cuda::mr::get_property(resource, required_alignment), stream);
    } else {
        return resource.allocate_async(size, my_default_alignment, stream);
    }
}
```

So, choose wisely. If your library has a well defined set of fixed properties that you expect to always be available, then `cuda::mr::{async_}resource_ref` is an amazing tool to improve compile times and binary size. If you need to your interface to be flexible then constraining trough `cuda::mr::{async_}resource_with` is the proper solution.
