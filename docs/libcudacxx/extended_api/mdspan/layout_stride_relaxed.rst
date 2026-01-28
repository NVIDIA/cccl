.. _libcudacxx-extended-api-mdspan-layout-stride-relaxed:

``layout_stride_relaxed``
=========================

Defined in the ``<cuda/mdspan>`` header.

``layout_stride_relaxed`` is a *LayoutMappingPolicy* which provides a layout mapping where the strides are user-defined and can be negative or zero.

Unlike ``cuda::std::layout_stride``, this layout allows:

- **Negative strides** for reverse iteration.
- **Zero strides** for broadcasting.
- **A base offset** to accommodate negative strides.
- **Compile-time stride values** for static arrays.

.. note::

    This layout is NOT always *unique*, *exhaustive*, or *strided* in the C++ standard sense.

----

Synopsis
--------

.. code:: cpp

    namespace cuda {

    // Tag value for dynamic stride (analogous to dynamic_extent)
    inline constexpr ptrdiff_t dynamic_stride = /* implementation-defined */;

    // Strides class template
    template <class OffsetType, ptrdiff_t... Strides>
    class strides;

    // Alias for all-dynamic strides
    template <class OffsetType, size_t Rank>
    using dstrides = strides<OffsetType, /* Rank dynamic_stride values */>;

    // Alias for steps (synonym for dstrides)
    template <size_t Rank, class OffsetType = ptrdiff_t>
    using steps = dstrides<OffsetType, Rank>;

    // Layout policy
    struct layout_stride_relaxed {
        template <class Extents,
                  class Stride = dstrides<make_signed_t<typename Extents::index_type>,
                                          Extents::rank()>>
        class mapping;
    };

    } // namespace cuda

----

``strides``
-----------

Class template to describe the strides of a multi-dimensional array layout. Similar to ``extents``, but for strides. Supports both *static* (compile-time known) and *dynamic* (runtime) stride values.

.. code:: cpp

    // Strides class template
    template <class OffsetType, ptrdiff_t... Strides>
    class strides;

    // Alias for all-dynamic strides
    template <class OffsetType, size_t Rank>
    using dstrides = strides<OffsetType, /* Rank dynamic_stride values */>;

    // Alias for steps (synonym for dstrides)
    template <size_t Rank, class OffsetType = ptrdiff_t>
    using steps = dstrides<OffsetType, Rank>;

**Template Parameters**

- ``OffsetType``: A signed integer type for stride values (supports negative strides).
- ``Strides...``: The stride values, where ``dynamic_stride`` indicates a runtime value.
- ``Rank``: The number of dimensions.

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Member Types**
      - Definition
    * - ``offset_type``
      - ``OffsetType``
    * - ``size_type``
      - ``cuda::std::make_unsigned_t<offset_type>``
    * - ``rank_type``
      - ``cuda::std::size_t``

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Static Member Functions**
      - Description
    * - ``rank()``
      - Returns the number of dimensions.
    * - ``rank_dynamic()``
      - Returns the number of dynamic strides.
    * - ``static_stride(rank_type r)``
      - Returns the static stride at dimension ``r``, or ``dynamic_stride`` if dynamic.

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Member Functions**
      - Description
    * - ``stride(rank_type r)``
      - Returns the stride at dimension ``r``.

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Non-member Functions**
      - Description
    * - ``operator==``
      - ``true`` if ranks are equal and all stride values compare equal; ``false`` otherwise.
    * - ``operator!=``
      - ``false`` if ranks are equal and all stride values compare equal; ``true`` otherwise.

**Constructors**

.. code:: cpp

    // (1) default constructor
    constexpr strides() noexcept = default;

    // (2) constructor from values
    template <class... OtherIndexTypes>
    constexpr explicit strides(OtherIndexTypes... values) noexcept;

    // (3) constructor from span
    template <class OtherIndexType, size_t Size>
    constexpr explicit(/*see below*/) strides(cuda::std::span<OtherIndexType, Size> strs) noexcept;

    // (4) constructor from array
    template <class OtherIndexType, size_t Size>
    constexpr explicit(/*see below*/) strides(const cuda::std::array<OtherIndexType, Size>& strs) noexcept;

    // (5) constructor from strides
    template <class OtherIndexType, ptrdiff_t... OtherStrides>
    constexpr explicit(/*see below*/) strides(const strides<OtherIndexType, OtherStrides...>& other) noexcept;

- **(1)** Default constructor. Value-initializes all dynamic strides to zero.
- **(2)** Initializes the strides with the provided values.

  - *Constraints*:

    - ``sizeof...(OtherIndexTypes)`` equals ``rank()`` or ``rank_dynamic()``.
    - ``OtherIndexTypes...`` is convertible and nothrow constructible to ``offset_type``.

  - *Preconditions*:

    - Each value is representable as ``offset_type``.

- **(3)**, **(4)** Initializes the strides from ``span`` or ``array`` values.

  - *Constraints*:

    - ``Size`` equals ``rank_dynamic()`` (implicit) or ``Size`` equals ``rank()`` and ``Size != rank_dynamic()`` (explicit).
    - ``OtherIndexType`` is convertible and nothrow constructible to ``offset_type``.

  - *Preconditions*:

    - Each value is representable as ``offset_type``.

- **(5)** Initializes the strides from another ``strides`` object.

  - The constructor is ``explicit`` if any static stride in ``Strides...`` corresponds to a ``dynamic_stride`` in ``OtherStrides...``.
  - *Constraints*:

    - ``sizeof...(OtherStrides)`` equals ``sizeof...(Strides)``.
    - For each dimension, either stride is ``dynamic_stride``, or both strides are equal.

  - *Preconditions*:

    - Static strides must match their compile-time values.
    - Each dynamic stride value is representable as ``offset_type``.

``layout_stride_relaxed::mapping``
----------------------------------

The class template ``layout_stride_relaxed::mapping`` controls how multidimensional indices are mapped with user-defined strides (including negative and zero strides) and an optional offset to a one-dimensional value representing the offset.

.. code:: cpp

    template <class Extents,
              class Stride = dstrides<make_signed_t<typename Extents::index_type>, Extents::rank()>>
    class layout_stride_relaxed::mapping;

**Template Parameters**

- ``Extents``: Specifies number of dimensions, their sizes, and which are known at compile time. Must be a specialization of ``cuda::std::extents``.
- ``Stride``: Specifies the strides for each dimension. Must be a specialization of ``cuda::strides``. Defaults to all-dynamic strides.

**Constraints**

- ``Extents`` must be a specialization of ``cuda::std::extents``.
- ``Extents::rank()`` must equal ``Stride::rank()``.

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Member Types**
      - Definition
    * - ``extents_type``
      - ``Extents``
    * - ``strides_type``
      - ``Stride``
    * - ``index_type``
      - ``extents_type::index_type``
    * - ``size_type``
      - ``extents_type::size_type``
    * - ``rank_type``
      - ``extents_type::rank_type``
    * - ``offset_type``
      - ``strides_type::offset_type``
    * - ``layout_type``
      - ``layout_stride_relaxed``

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Static Member Functions**
      - Description
    * - ``is_always_unique()``
      - Returns ``false``. Uniqueness is not guaranteed due to zero/negative strides.
    * - ``is_always_exhaustive()``
      - Returns ``false``. Exhaustiveness is not guaranteed due to zero/negative strides.
    * - ``is_always_strided()``
      - Returns ``false``. Standard strided behavior is not guaranteed due to offset.

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Member Functions**
      - Description
    * - ``extents()``
      - Returns the extents object.
    * - ``strides()``
      - Returns the strides object.
    * - ``offset()``
      - Returns the base offset (nonnegative).
    * - ``required_span_size()``
      - Returns the required span size to cover all valid indices.
    * - ``operator()(Indices... indices)``
      - Maps multidimensional indices to a linear index.
    * - ``is_unique()``
      - Returns ``false`` (conservative).
    * - ``is_exhaustive()``
      - Returns ``false`` (conservative).
    * - ``is_strided()``
      - Returns ``true`` if offset is zero, ``false`` otherwise.
    * - ``stride(rank_type r)``
      - Returns the stride along dimension ``r``.

.. list-table::
    :widths: 40 60
    :header-rows: 1

    * - **Non-member Functions**
      - Description
    * - ``operator==``
      - ``true`` if extents, strides, and offsets are equal; ``false`` otherwise.
    * - ``operator!=``
      - ``false`` if extents, strides, and offsets are equal; ``true`` otherwise.

**Constructors**

.. code:: cpp

    // (1) default constructor
    constexpr mapping() noexcept;

    // (2) copy constructor
    constexpr mapping(const mapping&) noexcept = default;

    // (3) constructor from strides
    constexpr mapping(const extents_type& ext,
                      const strides_type& strides,
                      offset_type offset = 0) noexcept;

    // (4) converting constructor from mapping
    template <class OtherMapping>
    constexpr explicit(/*see below*/) mapping(const OtherMapping& other) noexcept;

- **(1)** Default constructs the mapping, delegating to a ``layout_right`` mapping with default extents.
- **(2)** Copy constructor.
- **(3)** Direct-non-list-initializes the extents, strides, and offset with the provided arguments.

  - *Preconditions*:

    - ``offset`` is nonnegative.
    - ``required_span_size()`` is representable as ``index_type``.

- **(4)** Constructs the mapping by copying extents and strides from ``other``. For ``layout_stride_relaxed`` sources, also copies the offset.

  - The constructor is ``explicit`` if ``OtherMapping::extents_type`` is not convertible to ``extents_type``, or (for non-``layout_stride_relaxed`` mappings) the source is not a ``layout_left``, ``layout_right``, or ``layout_stride`` mapping.
  - *Constraints*:

    - ``OtherMapping`` is a layout mapping.
    - ``extents_type`` is constructible from ``OtherMapping::extents_type``.

  - *Preconditions*:

    - ``offset`` is nonnegative.
    - ``required_span_size()`` is representable as ``index_type``.

Examples
--------

**Compile-time strides (column-major layout)**

.. code:: cpp

    #include <cuda/mdspan>
    #include <cassert>

    int main() {
        // 3x4 matrix in column-major order
        int data[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

        using extents_t = cuda::std::extents<int, 3, 4>;
        // Compile-time strides: stride 1 for rows, stride 3 for columns
        using strides_t = cuda::strides<int, 1, 3>;
        using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t, strides_t>;

        mapping_t mapping(extents_t{}, strides_t{});

        // Column-major: consecutive elements in same column are adjacent
        assert(mapping(0, 0) == 0);
        assert(mapping(1, 0) == 1);
        assert(mapping(2, 0) == 2);
        assert(mapping(0, 1) == 3);
    }
1
**Negative strides (reverse iteration)**

.. code:: cpp

    #include <cuda/mdspan>
    #include <cassert>

    int main() {
        int data[] = {1, 2, 3, 4, 5};
        // Create a reversed view using negative stride
        // Offset points to the last element, stride is -1
        using extents_t = cuda::std::extents<int, 5>;
        using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
        using stride_t  = mapping_t::strides_type;
        mapping_t mapping(extents_t{}, stride_t(-1), 4); // offset=4, stride=-1
        // Access pattern: mapping(i) = 4 + i * (-1) = 4 - i
        assert(mapping(0) == 4);
        assert(mapping(1) == 3);
        assert(mapping(2) == 2);
        assert(mapping(3) == 1);
        assert(mapping(4) == 0);
    }

**Zero strides (broadcasting)**

.. code:: cpp

    #include <cuda/mdspan>
    #include <cassert>

    int main() {
        int scalar = 42;
        // Create a broadcast view: single value appears at all indices
        using extents_t = cuda::std::extents<int, 4, 4>;
        using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
        using stride_t  = mapping_t::strides_type;

        mapping_t mapping(extents_t{}, stride_t(0, 0)); // zero strides
        // All indices map to offset 0
        assert(mapping(0, 0) == 0);
        assert(mapping(1, 2) == 0);
        assert(mapping(3, 3) == 0);
    }

**Mixed strides**

.. code:: cpp

    #include <cuda/mdspan>
    #include <cassert>

    int main() {
        // 2D array with column-major layout but reversed rows
        // Data: row 0 at end, row 1 in middle, row 2 at start
        int data[3][4] = {
            {1, 2,  3,  4},
            {5, 6,  7,  8},
            {9, 10, 11, 12}
        };
        using extents_t = cuda::std::extents<int, 3, 4>;
        using mapping_t = cuda::layout_stride_relaxed::mapping<extents_t>;
        using stride_t  = mapping_t::strides_type;
        // Reverse rows: stride of -4 in row dimension, +1 in column
        // Offset to start at last row
        mapping_t mapping(extents_t{}, stride_t(-4, 1), 8);
        // Access gives reversed row order
        assert(mapping(0, 0) == 8); // data[2][0]
        assert(mapping(1, 0) == 4); // data[1][0]
        assert(mapping(2, 0) == 0); // data[0][0]
    }
