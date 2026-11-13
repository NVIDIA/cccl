.. _libcudacxx-extended-api-utility-in-range:

``in_range``
============

Defined in the ``<cuda/utility>`` header.

.. code:: cuda

    namespace cuda {

    template <typename T>
    [[nodiscard]] __host__ __device__ constexpr
    bool in_range(T value, T start, T end) noexcept;

    } // namespace cuda

Checks whether a value ``value`` is within the range ``[start, end]``, inclusive.

**Template Parameters**

- ``T``: the type of the value and the range bounds.

**Parameters**

- ``value``: the value to check if it's within the range.
- ``start``: the lower bound of the range.
- ``end``: the upper bound of the range.

**Return Value**

- ``true`` if ``value`` is in the range ``[start, end]``, ``false`` otherwise.

**Constraints**

- ``T`` must be an integer or floating point type.

**Preconditions**

- ``end`` must be greater than or equal to ``start``.

.. note::

  For floating-point types, ``NaN`` is considered a valid value. The function returns ``false`` if any input is ``NaN``.

**Performance considerations**

- The function is optimized when ``value`` is an unsigned integer type. The optimization is useful when ``start`` and ``end`` are known at compile-time, or when ``in_range`` is used multiple times with the same range.

Example
-------

.. code:: cuda

    #include <cuda/utility>
    #include <cuda/std/cassert>

    __global__ void in_range_kernel() {
        // unsigned integers
        assert(cuda::in_range(5u, 1u, 10u));   // 5  is in the range     [1, 10]
        assert(!cuda::in_range(15u, 1u, 10u)); // 15 is NOT in the range [1, 10]
        assert(cuda::in_range(1u, 1u, 10u));   // 1  is in the range     [1, 10]
        assert(cuda::in_range(10u, 1u, 10u));  // 10 is in the range     [1, 10]
        assert(cuda::in_range(10u, 10u, 10u)); // 10 is in the range     [10, 10]

        // signed integers
        assert(cuda::in_range(-5, -10, 0));    // -5 is in the range    [-10, 0]
        assert(!cuda::in_range(5, -10, 0));    // 5 is NOT in the range [-10, 0]
    }

    int main() {
        in_range_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

`See the example in Compiler Explorer 🔗 <https://godbolt.org/z/WjG4rWT3G>`_
