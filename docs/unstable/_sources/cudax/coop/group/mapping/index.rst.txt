.. _cudax-coop-group-mapping:

Group Mappings
==============

.. toctree::
  :hidden:

  identity_mapping
  group_by
  group_as
  take
  binary_partition
  composite_mapping

Group mappings define how a parent group's units are assigned to subgroups, independently of the synchronization mechanism. To communicate this information, they are required to accept and return types that satisfy the *group mapping result* concept.

Group Mapping Result Concept
----------------------------

*Group mapping result* is an implementation-defined concept that defines the interface for data exchange between groups and mappings. It is defined as:

.. code:: c++

  template <typename T>
  concept /*group-mapping-result*/ = requires (const T& t)
  {
    requires(cuda::std::is_copy_constructible_v<T>);

    // Group within the parent group info.
    { T::static_group_count() } noexcept -> cuda::std::same_as<cuda::std::size_t>; // (1), must be constexpr
    { t.group_count() } noexcept -> cuda::std::same_as<cuda::std::uint32_t>;       // (2)
    { t.group_rank() } noexcept -> cuda::std::same_as<cuda::std::uint32_t>;        // (3)

    // Unit within the group info.
    { T::static_unit_count() } noexcept -> cuda::std::same_as<cuda::std::size_t>;  // (4), must be constexpr
    { t.unit_count() } noexcept -> cuda::std::same_as<cuda::std::uint32_t>;        // (5)
    { t.unit_rank() } noexcept -> cuda::std::same_as<cuda::std::uint32_t>;         // (6)

    // Properties of the mapping result.
    { T::is_always_exhaustive() } noexcept -> cuda::std::same_as<bool>;            // (7), must be constexpr
    { T::is_always_contiguous() } noexcept -> cuda::std::same_as<bool>;            // (8), must be constexpr
    { t.is_valid() } noexcept -> cuda::std::same_as<bool>;                         // (9)

    // Other group-related info.
    { t.lane_mask() } noexcept -> cuda::std::same_as<cuda::device::lane_mask>;     // (10)
  };

Where:

#. The number of groups known at compile time. When it is equal to ``cuda::std::dynamic_extent``, the group count is known only at runtime.
#. The number of groups known at runtime. It is equal to ``static_cast<cuda::std::uint32_t>(static_group_count())`` when ``static_group_count() != cuda::std::dynamic_extent``.
#. The current unit's group rank.
#. The number of units in every group known at compile time. When equal to ``cuda::std::dynamic_extent``, the number of units is known only at runtime and may differ between groups.
#. The number of units within the unit's group known at runtime. It is equal to ``static_cast<cuda::std::uint32_t>(static_unit_count())`` when ``static_unit_count() != cuda::std::dynamic_extent``.
#. The current unit's rank.
#. A property known at compile time that specifies whether all units are always part of a group.
#. A property known at compile time that guarantees that each group occupies a contiguous range of physical units, with ``unit_rank()`` numbering them in ascending physical order starting at zero. This guarantee must hold after mapping composition as well.
#. A property that defines whether the mapping result is valid for the current unit. If not, it means that the unit won't be part of the newly created group.
#. The mask of threads within this warp that are part of this group.

.. note:: When a mapping result is not valid (``.is_valid() == false``), calling any other non-static query is undefined behavior. Predefined mappings preserve invalid results, so a unit excluded by an earlier mapping remains excluded by subsequent mappings.

Group Mapping Process
---------------------

The group mapping process has 3 steps:

1. The initial mapping result describes the parent group as one group: the static and runtime group counts are ``1``, the group rank is ``0``, and the unit count and rank are queried for the requested unit level. The exhaustive property is ``true`` relative to the parent group, and the contiguous property and lane mask are copied from the parent.
2. The mapping's ``.map(...)`` member function is invoked with the initial mapping result. When a non-exhaustve mapping is done, the ``cuda::experimental::coop::non_exhaustive`` tag is passed as the first parameter to the ``.map(...)`` method.
3. The transformed mapping result is used to initialize the newly created group.

Only units that belong to the parent group may create a child group. For a unit larger than a thread, all threads in that unit must agree on its assignment and participation.

Mappings preserve as much information as possible at compile time. Cooperative algorithms and group queries can use this information to select more efficient implementations. A property whose value is ``false`` means that the type does not guarantee the property; a particular runtime partition may still satisfy it.

Group Mapping Interface
-----------------------

Every group mapping type must implement the group mapping interface, which can be defined as the *group-mapping* concept:

.. code:: c++

  template <typename T, typename Unit, typename ParentGroup, typename PrevMappingResult>
  concept /*group-mapping*/ = requires (T& t,
                                        const Unit& unit,
                                        const ParentGroup& parent_group,
                                        const PrevMappingResult& prev_mapping_result)
  {
    { t.map(unit, parent_group, prev_mapping_result) } -> /*group-mapping-result*/;
    { t.map(non_exhaustive, unit, parent_group, prev_mapping_result) } -> /*group-mapping-result*/;
  };

Where:

- ``unit`` is the unit-level of the newly created group.
- ``parent_group`` is the parent group of the newly created group. It can be used to synchronize all units.
- ``prev_mapping_result`` is the input mapping result being transformed.
- ``non_exhaustive`` is the ``cuda::experimental::coop::non_exhaustive`` tag.

The overload that takes the ``non_exhaustive`` tag does the non-exhaustive mapping.

Mappings can be chained with ``|`` to transform the result from left to right. This is implemented by :ref:`cudax-coop-group-mapping-composite-mapping`.

Predefined Mappings
-------------------

The implementation comes with several predefined mappings that should cover most of the use cases. The documentation hides the group mapping interface behind a ``/*group-mapping-interface-for*/`` parent class.

- :doc:`identity_mapping <identity_mapping>` preserves the previous mapping result without further partitioning.
- :doc:`group_by <group_by>` splits each group into subgroups with the same specified number of units.
- :doc:`group_as <group_as>` splits each group into subgroups with individually specified sizes.
- :doc:`take <take>` selects the first specified number of units from each group.
- :doc:`binary_partition <binary_partition>` splits threads within a warp into two groups based on a predicate.
- :doc:`composite_mapping <composite_mapping>` applies multiple mappings in sequence, composed with the pipe operator (``|``).
