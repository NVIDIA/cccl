.. _libcudacxx-extended-api-mdspan-restrict-accessor:

``restrict`` ``mdspan`` and ``accessor``
========================================

.. code:: cpp

  template <typename Accessor>
  using restrict_accessor;

An alias type to create an accessor with the *restrict aliasing policy* starting from an existing accessor.

More information related to the *restrict aliasing policy* can be found in the CUDA programming guide: `__restrict__ keyword <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#restrict>`_.

----

.. code:: cpp

  template <typename ElementType,
            typename Extents,
            typename LayoutPolicy   = cuda::std::layout_right,
            typename AccessorPolicy = cuda::std::default_accessor<_ElementType>>
  using restrict_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, restrict_accessor<AccessorPolicy>>;

An alias type to create an ``mdspan`` with a *restrict aliasing policy* accessor.

----

Traits:

.. code:: cpp

    template <typename T>
    inline constexpr bool is_restrict_accessor_v = /*true if T is a restrict accessor, false otherwise*/;

    template <typename T>
    inline constexpr bool is_restrict_mdspan_v = /*true if T is a restrict mdspan, false otherwise*/;

----

**Constraints**:

- Accessor ``data_handle_type`` must be a pointer type.

Example
-------

.. code:: cuda

    #include <cuda/mdspan>

    using restrict_mdspan = cuda::restrict_mdspan<int, cuda::std::dims<1>>;

    __host__ __device__ void
    compute(restrict_mdspan a, restrict_mdspan b, restrict_mdspan c) {
        c[0] = a[0] * b[0];
        c[1] = a[0] * b[0];
        c[2] = a[0] * b[0] * a[1];
        c[3] = a[0] * a[1];
        c[4] = a[0] * b[0];
        c[5] = b[0];
    }

    int main() {
        using  dim      = cuda::std::dims<1>;
        using  mdspan   = cuda::std::mdspan<int, dim>;
        int    arrayA[] = {1, 2};
        int    arrayB[] = {5};
        int    arrayC[] = {9, 10, 11, 12, 13, 14};
        mdspan mdA{arrayA, dim{1}};
        mdspan mdB{arrayB, dim{5}};
        mdspan mdC{arrayC, dim{6}};
        compute(mdA, mdB, mdC);

        using restrict_aligned_accesor = cuda::std::restrict_accessor<cuda::std::aligned_accessor<int, 8>>;
        using restrict_aligned_mdspan  = cuda::std::mdspan<int, dim, layout_right, restrict_aligned_accesor>;
        restrict_aligned_mdspan mdD{mdC};
    }

`See it on Godbolt 🔗 <https://godbolt.org/z/Wjco996z8>`_
