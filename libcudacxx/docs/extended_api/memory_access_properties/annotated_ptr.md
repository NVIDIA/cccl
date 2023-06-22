---
parent: Memory access properties
grand_parent: Extended API
nav_order: 1
---

# `cuda::annotated_ptr`

Defined in header `<cuda/annotated_ptr>`:

```cuda
namespace cuda {
template <typename Type, typename Property>
class annotated_ptr<Type, Property>;
} // namespace cuda
```

**Mandates**: `Property` is one of:

* [`cuda::access_property::shared`],
* [`cuda::access_property::global`],
* [`cuda::access_property::persisting`],
* [`cuda::access_property::normal`],
* [`cuda::access_property::streaming`], or
* [`cuda::access_property`] (a type-erased property with a runtime value).

_Note_: if `Property` is [`cuda::access_property`], i.e. a dynamic property with a runtime value, then `sizeof(cuda::annotated_ptr<Type, cuda::access_property>) == 2 * sizeof(Type*)`. Otherwise, its size is `sizeof(Type*)`.

The class template [`cuda::annotated_ptr`] is a pointer annotated with an access property that _may_ be applied to memory operations performed through the [`cuda::annotated_ptr`].

In contrast with [`cuda::associate_access_property`], [`cuda::annotated_ptr`] maintains the association when passed through ABI boundaries, e.g., calling a non-inlined library function with a [`cuda::annotated_ptr`] argument.

It implements a pointer-like interface:

| Pointer Expression  | `cuda::annotated_ptr<T, P>`               | Description                                 |
|=====================|===========================================|=============================================|
| `T* a`              | `cuda::annotated_ptr<T, P> a`             | non-`const` pointer to non-`const` memory   |
| `T const * a`       | `cuda::annotated_ptr<T const, P> a`       | non-`const` pointer to `const` memory       |
| `T* const a`        | `const cuda::annotated_ptr<T, P> a`       | `const` pointer to non-`const` memory       |
| `T const* const a`  | `const cuda::annotated_ptr<T const, P> a` | `const` pointer to `const` memory           |
| `val = *a;`         | `val = *a;`                               | dereference operator to load an element     |
| `*a = val;`         | `*a = val;`                               | dereference operator to store an element    |
| `val = a[n];`       | `val = a[n];`                             | subscript operator to load an element       |
| `a[n] = val;`       | `a[n] = val;`                             | subscript operator to store an element      |
| `T* a = nullptr;`   | `annotated_ptr<T, P> a = nullptr;`        | `nullptr` initialization                    |
| `n = a - b;`        | `n = a - b;`                              | difference operator                         |
| `if (a) { ... }`    | `if (a) { ... }`                          | explicit bool conversion                    |

But it is not a drop-in replacement for pointers since, among others, it does not:

* model any [`Iterator`] concept,
* implement [`std::pointer_traits`], [`std::iterator_traits`], etc.
* have the same variance as pointer.

```cuda
namespace cuda {

template<class Type, class Property>
class annotated_ptr {
public:
  using value_type = Type;
  using size_type = std::size_t;
  using reference = value_type &;
  using pointer = value_type *;
  using const_pointer = value_type const *;
  using difference_type = std::ptrdiff_t;

  __host__ __device__ constexpr annotated_ptr() noexcept;
  __host__ __device__ constexpr annotated_ptr(annotated_ptr const&) noexcept = default;
  __host__ __device__ constexpr annotated_ptr& operator=(annotated_ptr const&) noexcept = default;
  __host__ __device__ explicit annotated_ptr(pointer);
  template <class RuntimeProperty>
  __host__ __device__ annotated_ptr(pointer, RuntimeProperty);
  template <class T, class P>
  __host__ __device__ annotated_ptr(annotated_ptr<T,P> const&);

  __host__ __device__ constexpr explicit operator bool() const noexcept;
  __host__ __device__ pointer get() const noexcept;

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
  __host__ __device__ reference operator[](std::ptrdiff_t) const;
  __host__ __device__ constexpr difference_type operator-(annotated_ptr);

private:
  pointer ptr;   // exposition only
  Property prop; // exposition only
};

} // namespace cuda
```

## Constructors and assignment

### Default constructor

```cuda
constexpr annotated_ptr() noexcept;
```

**Effects**:  as if constructed by `annotated_ptr(nullptr)`;

### Constructor from pointer

```cuda
constexpr explicit annotated_ptr(pointer ptr);
```

**Preconditions**:

* if `Property` is [`cuda::access_property::shared`] then `ptr` must be a generic pointer that is valid to cast to a pointer to the shared memory address space.
* if `Property` is [`cuda::access_property::global`], [`cuda::access_property::normal`], [`cuda::access_property::streaming`], [`cuda::access_property::persisting`], or [`cuda::access_property`]  then `ptr` must be a generic pointer that is valid to cast to a pointer to the global memory address space.

**Effects**:  Constructs an `annotated_ptr` requesting associating `ptr` with `Property`. 
If `Property` is [`cuda::access_property`] then `prop` is initialized with [`cuda::access_property::global`].

**Note**: in **Preconditions** "valid" means that casting the generic pointer to the corresponding address space does not introduce undefined behavior.

### Constructor from pointer and access property

```cuda
template <class RuntimeProperty>
annotated_ptr(pointer ptr, RuntimeProperty prop);
```

**Mandates**:

* `Property` is [`cuda::access_property`].
* `RuntimeProperty` is any of [`cuda::access_property::global`], [`cuda::access_property::normal`], [`cuda::access_property::streaming`], [`cuda::access_property::persisting`], or [`cuda::access_property`].

**Preconditions**: `ptr` is a pointer to a valid allocation in the global memory address space.

**Effects**:  Constructs an `annotated_ptr` requesting the association of `ptr` with the property `prop`.

# Copy constructor from a different `annotated_ptr`

```cuda
template <class T, class P>
constexpr annotated_ptr(annotated_ptr<T,P> const& a);
```

**Mandates**:

* `annotated_ptr<Type, Property>::pointer` is assignable from `annotated_ptr<T, P>::pointer`.
* `Property` is either [`cuda::access_property`] or `P`.
* `Property` and `P` specify the same memory space.

**Preconditions**: `pointer` is compatible with `Property`.

**Effects**: Constructs an `annotated_ptr` for the same pointer as the input `annotated_ptr`.


## Explicit conversion operator to `bool`

```cuda
constexpr operator bool() const noexcept;
```

**Returns**: `false` if the pointer is a `nullptr`, `true` otherwise.


## Raw pointer access

```cuda
pointer get() const noexcept;
```

**Returns**: A pointer derived from the `annotated_ptr`.

## Operators

### Dereference

```cuda
reference operator*() const;
```

**Preconditions**: The `annotated_ptr` is not null and points to a valid `T` value.

**Returns**: [`*cuda::associate_access_property(ptr, prop)`][`cuda::associate_access_property`]

### Pointer-to-member

```cuda
pointer operator->() const;
```

**Preconditions**: the `annotated_ptr` is not null.

**Returns**: [`cuda::associate_access_property(ptr, prop)`][`cuda::associate_access_property`]

### Subscript

```cuda
reference operator[](ptrdiff_t i) const;
```

**Preconditions**: `ptr` points to a valid allocation of at least size `[ptr, ptr+i]`.

**Returns**: [`*cuda::associate_access_property(ptr+i,prop)`][`cuda::associate_access_property`]

### Pointer distance

```cuda
constexpr difference_type operator-(annotated_ptr p) const;
```

**Preconditions**: `ptr` and `p` point to the same allocation.

**Returns**: as-if `get() - p.get()`.

## Example

Given three input and output vectors `x`, `y`, and `z`, and two arrays of coefficients `a` and `b`, all of length `N`:

```cuda
size_t N;
int* x, *y, *z;
int* a, *b;
```

the grid-strided kernel:

```cuda
__global__ void update(int* const x, int const* const a, int const* const b, size_t N) {
    auto g = cooperative_groups::this_grid();
    for (int i = g.thread_rank(); idx < N; idx += g.size()) {
        x[i] = a[i] * x[i] + b[i];
    }
}
```

updates `x`, `y`, and `z` as follows:

```cuda
update<<<grid, block>>>(x, a, b, N);
update<<<grid, block>>>(y, a, b, N);
update<<<grid, block>>>(z, a, b, N);
```

The elements of `a` and `b` are used in all kernels.
If `N` is large enough, elements of `a` and `b` might be evicted from the L2 cache, requiring these to be re-loaded from memory in the next `update`.

We can make the `update` kernel generic to allow the caller to pass [`cuda::annotated_ptr`] objects that hint at how memory will be accessed:

```cuda
template <typename PointerX, typename PointerA, typename PointerB>
__global__ void update_template(PointerX x, PointerA a, PointerB b, size_t N) {
    auto g = cooperative_groups::this_grid();
    for (int idx = g.thread_rank(); idx < N; idx += g.size()) {
        x[idx] = a[idx] * x[idx] + b[idx];
    }
}
```

With [`cuda::annotated_ptr`], the caller can then specify the temporal locality of the memory accesses:

```cuda
// Frequent accesses to "a" and "b"; infrequent accesses to "x" and "y":
cuda::annotated_ptr<int const, cuda::access_property::persisting> a_p {a}, b_p{b};
cuda::annotated_ptr<int, cuda::access_property::streaming> x_s{x}, y_s{y};
update_template<<<grid, block>>>(x_s, a_p, b_p, N);
update_template<<<grid, block>>>(y_s, a_p, b_p, N);

// Infrequent accesses to "a" and "b"; frequent acceses to "z":
cuda::annotated_ptr<int const, cuda::access_property::streaming> a_s {a}, b_s{b};
cuda::annotated_ptr<int, cuda::access_property::persisting> z_p{z};
update_template<<<grid, block>>>(z_p, a_s, b_s, N);

// Different kernel, "update_z", uses "z" again one last time.
// Since "z" was accessed as "persisting" by the previous kernel,
// parts of it are more likely to have previously survived in the L2 cache.
update_z<<<grid, block>>>(z, ...);
```

Notice how the raw pointers to `a` and `b` can be wrapped by both `annotated_ptr<T, persistent>` and `annotated_ptr<T, streaming>`, and accesses through each pointer applies the corresponding access property.

[`Iterator`]: https://en.cppreference.com/w/cpp/iterator
[`std::pointer_traits`]: https://en.cppreference.com/w/cpp/memory/pointer_traits
[`std::iterator_traits`]: https://en.cppreference.com/w/cpp/iterator/iterator_traits

[`cuda::annotated_ptr`]: {{ "extended_api/memory_access_properties/annotated_ptr.html" | relative_url }}
[`cuda::access_propety`]: {{ "extended_api/memory_access_properties/access_property.html" | relative_url }}
[`cuda::associate_access_property`]: {{ "extended_api/memory_access_properties/associate_access_property.html" | relative_url }}
[`cuda::apply_access_property`]: {{ "extended_api/memory_access_properties/apply_access_property.html" | relative_url }}
[`cuda::access_property::shared`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::global`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::persisting`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::normal`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
[`cuda::access_property::streaming`]: {{ "extended_api/memory_access_properties/access_property.html#kinds-of-access-properties" | relative_url }}
