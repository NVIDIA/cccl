.. _libcudacxx-extended-api-functional-operator-properties:

Operator Properties
===================

Defined in the header ``<cuda/functional>``.

The operator properties traits provide compile-time information about algebraic properties of binary operators.
These traits are useful for generic algorithms that can apply optimizations based on operator properties,
such as parallel reductions that can reorder operations for associative operators.

Associativity
-------------

.. code:: cuda

   namespace cuda {

   template <class Op, class T>
   struct is_associative;

   template <class Op, class T>
   inline constexpr bool is_associative_v = is_associative<Op, T>::value;

   } // namespace cuda

Determines whether a binary operator ``Op`` is associative for type ``T``, meaning ``op(op(a, b), c) == op(a, op(b, c))``
for all values ``a``, ``b``, ``c`` of type ``T``. This allows the implementation to reorder operations.

**Supported operators and types:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Operator
     - Integer Types
     - Floating-Point Types
   * - ``cuda::std::plus``
     - ``true``
     - ``false`` (due to rounding errors)
   * - ``cuda::std::multiplies``
     - ``true``
     - ``false`` (due to rounding errors)
   * - ``cuda::std::minus``
     - ``false``
     - ``false``
   * - ``cuda::std::divides``
     - ``false``
     - ``false``
   * - ``cuda::std::modulus``
     - ``false``
     - N/A
   * - ``cuda::std::bit_and``
     - ``true``
     - N/A
   * - ``cuda::std::bit_or``
     - ``true``
     - N/A
   * - ``cuda::std::bit_xor``
     - ``true``
     - N/A
   * - ``cuda::std::logical_and``
     - ``true`` (``bool`` only)
     - N/A
   * - ``cuda::std::logical_or``
     - ``true`` (``bool`` only)
     - N/A
   * - ``cuda::minimum``
     - ``true``
     - ``true``
   * - ``cuda::maximum``
     - ``true``
     - ``true``

.. note::

   In the strictest sense of the term, the operations of plus and multiplication for integral values may result in undefined behaviour due to overflow. However, in the context of parallel algorithms, they are considered to be associative.

Commutativity
-------------

.. code:: cuda

   namespace cuda {

   template <class Op, class T>
   struct is_commutative;

   template <class Op, class T>
   inline constexpr bool is_commutative_v = is_commutative<Op, T>::value;

   } // namespace cuda

Determines whether a binary operator ``Op`` is commutative for type ``T``, meaning ``op(a, b) == op(b, a)``
for all values ``a``, ``b`` of type ``T``.

**Supported operators and types:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Operator
     - Integer Types
     - Floating-Point Types
   * - ``cuda::std::plus``
     - ``true``
     - ``true``
   * - ``cuda::std::multiplies``
     - ``true``
     - ``true``
   * - ``cuda::std::minus``
     - ``false``
     - ``false``
   * - ``cuda::std::divides``
     - ``false``
     - ``false``
   * - ``cuda::std::modulus``
     - ``false``
     - N/A
   * - ``cuda::std::bit_and``
     - ``true``
     - N/A
   * - ``cuda::std::bit_or``
     - ``true``
     - N/A
   * - ``cuda::std::bit_xor``
     - ``true``
     - N/A
   * - ``cuda::std::logical_and``
     - ``true`` (``bool`` only)
     - N/A
   * - ``cuda::std::logical_or``
     - ``true`` (``bool`` only)
     - N/A
   * - ``cuda::minimum``
     - ``true``
     - ``true``
   * - ``cuda::maximum``
     - ``true``
     - ``true``

Element Existence
-----------------

.. code:: cuda

   namespace cuda {

   template <template <class...> class Trait, class... Tp>
   inline constexpr bool element_exists = /* see below */;

   } // namespace cuda

A helper trait that evaluates to ``true`` if the given trait template ``Trait`` instantiated with types ``Tp...``
has a valid ``value`` member.

.. code:: cuda

   // Check if identity_element is defined for plus<int> and int
   static_assert(cuda::element_exists<cuda::identity_element, cuda::std::plus<int>, int>);

   // Check if absorbing_element is defined for plus<int> and int (it's not)
   static_assert(!cuda::element_exists<cuda::absorbing_element, cuda::std::plus<int>, int>);

Identity Element
----------------

.. code:: cuda

   namespace cuda {

   template <class Op, class T>
   struct identity_element;

   template <class Op, class T>
   constexpr auto get_identity_element() noexcept;

   template <class Op, class T>
   inline constexpr bool has_identity_element = /* see below */;

   } // namespace cuda

Provides the identity element for operator ``Op`` and type ``T``. The identity element ``e`` satisfies
``op(e, x) == op(x, e) == x`` for all values ``x`` of type ``T``.

The function ``get_identity_element<Op, T>()`` returns ``identity_element<Op, T>::value``.

``has_identity_element`` evaluates to ``true`` if an identity element is defined for the given operator and type.

**Identity elements by operator:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Operator
     - Integer Types
     - Floating-Point Types
   * - ``cuda::std::plus``
     - ``T{0}``
     - ``-0.0`` (negative zero)
   * - ``cuda::std::multiplies``
     - ``T{1}``
     - ``1.0``
   * - ``cuda::std::bit_and``
     - ``~T{0}`` (all bits set)
     - N/A
   * - ``cuda::std::bit_or``
     - ``T{0}``
     - N/A
   * - ``cuda::std::bit_xor``
     - ``T{0}``
     - N/A
   * - ``cuda::std::logical_and``
     - ``true`` (``bool`` only)
     - N/A
   * - ``cuda::std::logical_or``
     - ``false`` (``bool`` only)
     - N/A
   * - ``cuda::minimum``
     - ``numeric_limits<T>::max()``
     - ``+infinity``
   * - ``cuda::maximum``
     - ``numeric_limits<T>::lowest()``
     - ``-infinity``

.. note::

   For floating-point ``plus``, the identity element is negative zero (``-0.0``) rather than positive zero.
   This preserves the sign when adding: ``-0.0 + (-0.0) == -0.0``.

.. note::

   ``cuda::std::minus``, ``cuda::std::divides``, and ``cuda::std::modulus`` do not have identity elements.

Absorbing Element
-----------------

.. code:: cuda

   namespace cuda {

   template <class Op, class T>
   struct absorbing_element;

   template <class Op, class T>
   constexpr auto get_absorbing_element() noexcept;

   template <class Op, class T>
   inline constexpr bool has_absorbing_element = /* see below */;

   } // namespace cuda

Provides the absorbing (annihilating) element for operator ``Op`` and type ``T``. The absorbing element ``z`` satisfies
``op(z, x) == op(x, z) == z`` for all values ``x`` of type ``T``.

The function ``get_absorbing_element<Op, T>()`` returns ``absorbing_element<Op, T>::value``.

``has_absorbing_element`` evaluates to ``true`` if an absorbing element is defined for the given operator and type.

**Absorbing elements by operator:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Operator
     - Integer Types
     - Floating-Point Types
   * - ``cuda::std::multiplies``
     - ``T{0}``
     - N/A (see note)
   * - ``cuda::std::bit_and``
     - ``T{0}``
     - N/A
   * - ``cuda::std::bit_or``
     - ``~T{0}`` (all bits set)
     - N/A
   * - ``cuda::std::logical_and``
     - ``false`` (``bool`` only)
     - N/A
   * - ``cuda::std::logical_or``
     - ``true`` (``bool`` only)
     - N/A
   * - ``cuda::minimum``
     - ``numeric_limits<T>::lowest()``
     - ``-infinity``
   * - ``cuda::maximum``
     - ``numeric_limits<T>::max()``
     - ``+infinity``

.. note::

   Floating-point ``multiplies`` does not have an absorbing element because:

   - ``0 * NaN = NaN`` (not ``0``)
   - ``0 * infinity = NaN`` (not ``0``)
   - ``(-1) * (+0) = -0`` (not ``+0``)

.. note::

   ``cuda::std::plus``, ``cuda::std::minus``, ``cuda::std::divides``, ``cuda::std::modulus``, and ``cuda::std::bit_xor``
   do not have absorbing elements.

Supported Types
---------------

The functionality supports all integer and floating-point types, including extended floating-point types.

Example
-------

.. code:: cuda

   #include <cuda/functional>
   #include <cuda/std/cstdio>

   template <class Op, class T>
   __host__ __device__ void print_properties() {
        printf("Associative: %s\n", cuda::is_associative_v<Op, T> ? "yes" : "no");
        printf("Commutative: %s\n", cuda::is_commutative_v<Op, T> ? "yes" : "no");

        if constexpr (cuda::has_identity_element<Op, T>) {
            printf("Identity element exists\n");
        }
        if constexpr (cuda::has_absorbing_element<Op, T>) {
            printf("Absorbing element exists\n");
        }
   }

   __global__ void example_kernel() {
        // Integer plus: associative, commutative, identity=0, no absorbing
        print_properties<cuda::std::plus<int>, int>();

        // Integer multiplies: associative, commutative, identity=1, absorbing=0
        print_properties<cuda::std::multiplies<int>, int>();

        // Floating-point plus: commutative but NOT associative
        static_assert(!cuda::is_associative_v<cuda::std::plus<float>, float>);
        static_assert(cuda::is_commutative_v<cuda::std::plus<float>, float>);

        // Use identity element for reduction initialization
        int sum_identity = cuda::get_identity_element<cuda::std::plus<int>, int>();       // 0
        int mul_identity = cuda::get_identity_element<cuda::std::multiplies<int>, int>(); // 1
   }
