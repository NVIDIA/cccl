.. _libcudacxx-extended-api-functional-always-true-false:

``cuda::always_true`` and ``cuda::always_false``
================================================

Defined in the header ``<cuda/functional>``.

.. code:: cuda

    struct always_true {
        template <typename... Ts>
        [[nodiscard]] __host__ __device__ constexpr bool operator()(Ts&&...) const noexcept;
    };

    struct always_false {
        template <typename... Ts>
        [[nodiscard]] __host__ __device__ constexpr bool operator()(Ts&&...) const noexcept;
    };

``cuda::always_true`` is a function object that always returns ``true`` regardless of the number and type of arguments
passed. ``cuda::always_false`` is a function object that always returns ``false`` regardless of the number and type of
arguments passed.

Both types are empty, trivially copyable, and their ``operator()`` is ``constexpr`` and ``noexcept``.

Example
-------

.. code:: cuda

    #include <cuda/functional>

    __global__ void example_kernel() {
        cuda::always_true  pred_true{};
        cuda::always_false pred_false{};

        // Returns true regardless of arguments
        static_assert(pred_true());
        static_assert(pred_true(1, 2, 3));

        // Returns false regardless of arguments
        static_assert(!pred_false());
        static_assert(!pred_false(1, 2, 3));
    }
