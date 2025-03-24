.. _libcudacxx-extended-api-mdspan-host-device-accessor:

``host/device/managed`` ``mdspan`` and ``accessors``
====================================================

*Host*, *device*, and *managed* ``mdspan`` allow to express multi-dimensional views of the respective CUDA memory spaces as *vocabulary types* and prevent potential errors.

Types and Traits
----------------

.. code:: cpp

  template <typename Accessor>
  using host_accessor;

  template <typename Accessor>
  using device_accessor;

  template <typename Accessor>
  using managed_accessor;

Alias types to create accessors tailored for the *host*, *device*, or *managed* memory spaces.

----

.. code:: cpp

  template <typename ElementType,
            typename Extents,
            typename LayoutPolicy   = cuda::std::layout_right,
            typename AccessorPolicy = cuda::std::default_accessor<_ElementType>>
  using host_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, host_accessor<AccessorPolicy>>;

  template <typename ElementType,
            typename Extents,
            typename LayoutPolicy   = cuda::std::layout_right,
            typename AccessorPolicy = cuda::std::default_accessor<_ElementType>>
  using device_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, device_accessor<AccessorPolicy>>;

  template <typename ElementType,
            typename Extents,
            typename LayoutPolicy   = cuda::std::layout_right,
            typename AccessorPolicy = cuda::std::default_accessor<_ElementType>>
  using managed_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, managed_accessor<AccessorPolicy>>;

Alias types to create ``mdspan`` with *host*, *device*, or *managed* accessors.

----

.. code:: cpp

  template <typename T>
  inline constexpr bool is_host_accessor_v = /* true if T is a host accessor, false otherwise */

  template <typename T>
  inline constexpr bool is_device_accessor_v = /* true if T is a device accessor, false otherwise */

  template <typename T>
  inline constexpr bool is_managed_accessor_v = /* true if T is a managed accessor, false otherwise */

  template <typename T>
  inline constexpr bool is_host_accessible_v = /* true if T is a mdspan/accessor accessible from the host, false otherwise */

  template <typename T>
  inline constexpr bool is_device_accessible_v = /* true if T is a mdspan/accessor accessible from the device, false otherwise */

----

Features
--------

**Memory spaces**

*Host*, *device*, and *managed* ``mdspan`` can be created and "sliced" (``cuda::std::submdspan``) on any memory space. However, access to a specific memory space is restricted to the respective *accessor* type.

+----------------------------------+------------------+-------------------+
| ``mdspan`` / memory space access | Host memory      | Device memory     |
+==================================+==================+===================+
| ``host_mdspan``                  | Allowed          | *Compile error*   |
+----------------------------------+------------------+-------------------+
| ``device_mdspan``                | *Compile error*  | Allowed           |
+----------------------------------+------------------+-------------------+
| ``managed_mdspan``               | Allowed *****    | Allowed *****     |
+----------------------------------+------------------+-------------------+

***** the validity of the *managed* memory space is checked at run-time in debug mode (host-side).

**Conversions**

+-----------------------------+------------------+-------------------+---------------------+
|                             | ``host_mdspan``  | ``device_mdspan`` | ``managed_mdspan``  |
+=============================+==================+===================+=====================+
| ``host_mdspan``             | Allowed          | *Compile error*   | *Compile error*     |
+-----------------------------+------------------+-------------------+---------------------+
| ``device_mdspan``           | *Compile error*  | Allowed           | *Compile error*     |
+-----------------------------+------------------+-------------------+---------------------+
| ``managed_mdspan``          | Allowed          | Allowed           | Allowed             |
+-----------------------------+------------------+-------------------+---------------------+
| Other mdspan                | Allowed          | Allowed           | Allowed             |
+-----------------------------+------------------+-------------------+---------------------+

*Note:* the conversion is ``explicit`` if the base accessor is not directly convertible.

Example 1
---------

``cuda::host_mdspan`` and ``cuda::device_mdspan`` usage:

.. code:: cpp

    #include <cuda/mdspan>

    using dim = cuda::std::dims<1>;

    __global__ void kernel_d(cuda::device_mdspan<int, dim> md) {
        md[0] = 0;
    }
    __global__ void kernel_h(cuda::host_mdspan<int, dim> md) {
        // md[0] = 0;  // compile error
    }

    __host__ void host_function_h(cuda::host_mdspan<int, dim> md) {
        md[0] = 0;
    }
    __host__ void host_function_d(cuda::device_mdspan<int, dim> md) {
        // md[0] = 0;  // compile error
    }
    __host__ void host_function_m(cuda::managed_mdspan<int, dim> md) {
        md[0] = 0;
    }

    int main() {
        int* d_ptr;
        cudaMalloc(&d_ptr, 4 * sizeof(int));
        int                 h_ptr[4];
        cuda::host_mdspan   h_md{h_ptr};
        cuda::device_mdspan d_md{d_ptr, 4};
        kernel_d<<<1, 1>>>(d_md);    // ok
        // kernel_d<<<1, 1>>>(h_md); // compile error
        host_function_h(h_md);       // ok
        host_function_d(h_md);       // compile error
        // host_function_m(h_md);    // compile error
        cudaFree(d_ptr);
    }

`See example 1 on Godbolt 🔗 <https://godbolt.org/z/fezxsbjaq>`_

Example 2
---------

``cuda::managed_mdspan`` usage:

.. code:: cpp

    #include <cuda/mdspan>

    using dim = cuda::std::dims<1>;

    __global__ void kernel_d(cuda::device_mdspan<int, dim> md) {
        md[0] = 0;
    }

    __host__ void host_function_h(cuda::host_mdspan<int, dim> md) {
        md[0] = 0;
    }

    int main() {
        int* m_ptr;
        cudaMallocManaged(&m_ptr, 4 * sizeof(int));
        cuda::managed_mdspan m_md{m_ptr, 4};
        kernel_d<<<1, 1>>>(m_md); // ok
        host_function_h(m_md);    // ok

        cuda::managed_mdspan m_md2{d_ptr, 4};
        m_md2[0]; // run-time error
        cudaFree(d_ptr);
    }

`See example 2 on Godbolt 🔗 <https://godbolt.org/z/Kj39Pe4vP>`_


Example 3
---------

Conversion from other accessors:

.. code:: cpp

    #include <cuda/mdspan>

    using dim = cuda::std::dims<1>;

    int main() {
        using cuda::std::layout_right;
        using cuda::std::aligned_accessor;
        int               h_ptr[4];
        cuda::std::mdspan md{h_ptr};
        cuda::host_mdspan h_md = md; // ok

        cuda::std::mdspan<int, dim, layout_right, aligned_accessor<int, 8>> md_a{h_ptr, 4};
        // cuda::host_mdspan h_md = md_a; // compile-error
        cuda::host_mdspan    h_md{md_a};  // ok
    }

`See example 3 on Godbolt 🔗 <https://godbolt.org/z/7dq7vcTWP>`_
