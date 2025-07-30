CUB Tests
###########################

.. warning::
    CUB is in the progress of migrating to [Catch2](https://github.com/catchorg/Catch2) framework.

CUB tests rely on `CPM <https://github.com/cpm-cmake/CPM.cmake>`_ to fetch
`Catch2 <https://github.com/catchorg/Catch2>`_ that's used as our main testing framework.

Currently,
legacy tests coexist with Catch2 ones.
This guide is focused on new tests.

.. important::
    Instead of including ``<catch2/catch.hpp>`` directly, use ``catch2_test_helper.h``.

.. code-block:: c++

    #include <cub/block/block_scan.cuh>
    #include <c2h/vector.h>
    #include <c2h/catch2_test_helper.h>

Directory and File Naming
*************************************

Our tests can be found in the ``test`` directory.
Legacy tests have the following naming scheme: ``test_SCOPE_FACILITY.cu``.
For instance, here are the reduce tests:

.. code-block:: c++

    test/test_warp_reduce.cu
    test/test_block_reduce.cu
    test/test_device_reduce.cu

Catch2-based tests have a different naming scheme: ``catch2_test_SCOPE_FACILITY.cu``.

The prefix is essential since that's how CMake finds tests
and distinguishes new tests from legacy ones.

Test Structure
*************************************

Base case
=====================================
Let's start with a simple example.
Say there's no need to cover many types with your test.

.. code-block:: c++

    // 0) Define test name and tags
    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]")
    {
      using type = std::int32_t;
      constexpr int threads_per_block = 256;
      constexpr int num_items = threads_per_block;

      // 1) Allocate device input
      c2h::device_vector<type> d_input(num_items);

      // 2) Generate 3 random input arrays using Catch2 helper
      c2h::gen(C2H_SEED(3), d_input);

      // 3) Allocate output array
      c2h::device_vector<type> d_output(d_input.size());

      // 4) Copy device input to host
      c2h::host_vector<key_t> h_reference = d_input;

      // 5) Compute reference output
      std::ALGORITHM(
          thrust::raw_pointer_cast(h_reference.data()),
          thrust::raw_pointer_cast(h_reference.data()) + h_reference.size());

      // 6) Compute CUB output
      SCOPE_ALGORITHM<threads_per_block>(d_input.data(),
                                         d_output.data(),
                                         d_input.size());

      // 7) Compare device and host results
      REQUIRE( d_input == d_output );
    }

We introduce test cases with the ``C2H_TEST`` macro in (0).
This macro always takes two string arguments - a free-form test name and
one or more tags. Then, in (1), we allocate device memory using ``c2h::device_vector``.
``c2h::device_vector`` and ``c2h::host_vector`` behave similarly to their Thrust counterparts,
but are modified to provide more stable behavior in some testing edge cases.

.. important::
    Always use ``c2h::host_vector<T>``/``c2h::device_vector<T>``
    instead of ``thrust::host_vector<T>``/``thrust::device_vector<T>``,
    unless the test code is being used for documentation examples.

Similarly, any thrust algorithms that executed on the device must be invoked with the
`c2h::device_policy` execution policy (not shown here) to support the same edge cases.
The memory is filled with random data in (2).

Generator ``c2h::gen`` takes at least two parameters.
The first one is a random generator seed.
Instead of providing a single value, we use the ``C2H_SEED`` macro.
The macro expects a number of seeds that has to be generated.
In the example above, we require three random seeds to be generated.
This leads to the whole test being executed three times
with different seed values.

Later, in (3), we allocate device output and host reference.
In (4), we allocate and populate the host input data.
Then, we perform the reference computation on the host in (5).

.. important::
    Standard library algorithms (``std::``) have to be used where possible when computing reference solutions.

Afterwards, we launch the corresponding CUB algorithm in (6).
At this point, we have a reference solution on the CPU and a CUB solution on the GPU.
The two can be compared using Catch2's ``REQUIRE`` macro, which stops execution upon failure (preferred).
Catch2 also offers the ``CHECK`` macro, which continues test execution if the check fails.

If your test has to cover floating point types,
it's sufficient to replace ``REQUIRE( a == b )`` with ``REQUIRE_APPROX_EQ(a, b)``.

.. important::
    Using ``c2h::gen`` for producing input data is strongly advised.

Do not use ``assert`` in tests, which is usually only enabled in Debug mode,
and we run CUB tests in Release mode.

If a custom (non-fundamental) type has to be tested, the following helper class template should be used:

.. code-block:: c++

    using type = c2h::custom_type_t<c2h::accumulateable_t,
                                    c2h::equal_comparable_t>;

Here we enumerate all the type properties that we are interested in.
The produced type ends up having ``operator==`` (from ``equal_comparable_t``)
and ``operator+`` (from ``accumulateable_t``).
More properties are available.
If a property is missing, please add it to the existing set in ``c2h``
instead of writing a custom type from scratch.

Generators
=====================================

We often need to test CUB algorithms against different inputs or problem sizes.
If these are **runtime values**, we can use the Catch2 ``GENERATE`` macro:

.. code-block:: c++

    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]")
    {
      int num_items = GENERATE(1, 100, 1'000'000); // 0) Init. a variable with a generator
      // ...
    }

This will lead to the test being executed three times, once for each argument to ``GENERATE(...)``.
Multiple generators in a test inside the same scope will form the cartesian product of all combinations.
Please consult the `Catch2 documentation <https://github.com/catchorg/Catch2/blob/devel/docs/generators.md>`_
for more details.

``C2H_SEED(3)`` uses a generator expression internally.


Type Lists
=====================================

Since CUB is a generic library,
it's often required to test CUB algorithms against many types.
To do so,
it's sufficient to define a type list and provide it to the ``C2H_TEST`` macro.
This is useful for **compile-time** parameterization of tests.

.. code-block:: c++

    // 0) Define type list
    using types = c2h::type_list<std::uint8_t, std::int32_t>;

    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]",
            types) // 1) Provide it to the test case
    {
      // 2) Access current type with `c2h::get`
      using type = typename c2h::get<0, TestType>;
      // ...
    }

This will lead to the test being compiled (instantiated) and run twice.
The first run will cause ``type`` to be ``std::uint8_t``.
The second one will cause ``type`` to be ``std::uint32_t``.

.. warning::
    It's important to use types from the ``<cstdint>`` header
    instead of built-in types like ``char`` and ``int``.

Multidimensional Configuration Spaces
=====================================

In most cases, the input data type is not the only compile-time parameter we want to vary.
For instance, you might need to test a block algorithm for different data types
**and** different thread block sizes.
To do so, you can add another type list as follows:

.. code-block:: c++

    using block_sizes = c2h::enum_type_list<int, 128, 256>;
    using types = c2h::type_list<std::uint8_t, std::int32_t>;

    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]",
             types, block_sizes)
    {
      using type = typename c2h::get<0, TestType>;
      constexpr int threads_per_block = c2h::get<1, TestType>::value;
      // ...
    }

The code above leads to the following combinations being compiled:

- ``type = std::uint8_t``, ``threads_per_block = 128``
- ``type = std::uint8_t``, ``threads_per_block = 256``
- ``type = std::int32_t``, ``threads_per_block = 128``
- ``type = std::int32_t``, ``threads_per_block = 256``

As an example, the following test case includes both multidimensional configuration spaces
and multiple random sequence generations.

.. code-block:: c++

    using block_sizes = c2h::enum_type_list<int, 128, 256>;
    using types = c2h::type_list<std::uint8_t, std::int32_t>;

    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]",
             types, block_sizes)
    {
      using type = typename c2h::get<0, TestType>;
      constexpr int threads_per_block = c2h::get<1, TestType>::value;
      // ...
      c2h::device_vector<type> d_input(5);
      c2h::gen(C2H_SEED(2), d_input);
    }

The code above leads to the following combinations being compiled:

- ``type = std::uint8_t``, ``threads_per_block = 128``, 1st random generated input sequence
- ``type = std::uint8_t``, ``threads_per_block = 256``, 1st random generated input sequence
- ``type = std::int32_t``, ``threads_per_block = 128``, 1st random generated input sequence
- ``type = std::int32_t``, ``threads_per_block = 256``, 1st random generated input sequence
- ``type = std::uint8_t``, ``threads_per_block = 128``, 2nd random generated input sequence
- ``type = std::uint8_t``, ``threads_per_block = 256``, 2nd random generated input sequence
- ``type = std::int32_t``, ``threads_per_block = 128``, 2nd random generated input sequence
- ``type = std::int32_t``, ``threads_per_block = 256``, 2nd random generated input sequence

Each new generator multiplies the number of execution times by its number of seeds. That means
that if there were further more sequence generators (``c2h::gen(C2H_SEED(X), ...)``) on the
example above the test would execute X more times and so on.

Speedup Compilation Time
=====================================

Since type lists in the ``C2H_TEST`` form a Cartesian product,
compilation time grows quickly with every new dimension.
To keep the compilation process parallelized,
it's possible to rely on our ``%PARAM%`` machinery:

.. code-block:: c++

    // %PARAM% BLOCK_SIZE bs 128:256
    using block_sizes = c2h::enum_type_list<int, BLOCK_SIZE>;
    using types = c2h::type_list<std::uint8_t, std::int32_t>;

    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]",
             types, block_sizes)
    {
      using type = typename c2h::get<0, TestType>;
      constexpr int threads_per_block = c2h::get<1, TestType>::value;
      // ...
    }

The comment with ``%PARAM%`` is recognized by our CMake scripts.
It leads to multiple executables being produced from a single test source.

.. code-block:: bash

    bin/cub.test.scope_algorithm.bs_128
    bin/cub.test.scope_algorithm.bs_256

Multiple ``%PARAM%`` comments can be specified forming another Cartesian product.

Final Test
=====================================

Let's consider the final test that illustrates all of the tools we discussed above:

.. code-block:: c++

    // %PARAM% BLOCK_SIZE bs 128:256
    using block_sizes = c2h::enum_type_list<int, BLOCK_SIZE>;
    using types = c2h::type_list<std::uint8_t, std::int32_t>;

    C2H_TEST("SCOPE FACILITY works with CONDITION", "[FACILITY][SCOPE]",
             types, block_sizes)
    {
      using type = typename c2h::get<0, TestType>;
      constexpr int threads_per_block = c2h::get<1, TestType>::value;
      constexpr int max_num_items = threads_per_block;

      c2h::device_vector<type> d_input(
        GENERATE_COPY(take(2, random(0, max_num_items))));
      c2h::gen(C2H_SEED(3), d_input);

      c2h::device_vector<type> d_output(d_input.size());

      SCOPE_ALGORITHM<threads_per_block>(d_input.data(),
                                        d_output.data(),
                                        d_input.size());

      REQUIRE( d_input == d_output );

      const type expected_sum = 4;
      const type sum = thrust::reduce(c2h::device_policy, d_output.cbegin(), d_output.cend());
      REQUIRE( sum == expected_sum);
    }

Apart from discussed tools, here we also rely on ``Catch2`` to generate random input sizes
in the range ``[0, max_num_items]`` for our input vector ``d_input``.
Overall, the test will produce two executables.
Each of these executables is going to generate ``2`` input problem sizes.
For each problem size, ``3`` random vectors are generated.
As a result, we have ``12`` different tests.
The code also demonstrates the syntax and usage of ``c2h::device_policy`` with a Thrust algorithm.
