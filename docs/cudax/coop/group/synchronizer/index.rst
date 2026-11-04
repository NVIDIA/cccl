.. _cudax-coop-group-synchronizer:

Group Synchronizers
===================

Group synchronizers are types that implement the synchronizer interface. They define what synchronization mechanism shall be used to synchronize the group. The type needs to implement the ``.make_instance(...)`` method that returns a type that satisfies the *group synchronizer instance* concept.

Group Synchronizer Instance Concept
-----------------------------------

*Group synchronizer instance* is an implementation-defined concept that defines the interface for a group synchronizer instance. It is defined as:

.. code:: c++

  template <typename T, typename MappingResult, typename Hierarchy>
  concept /*group-synchronizer-instance*/ = requires (T& t,
                                                      const T& ct,
                                                      const MappingResult& mapping_result,
                                                      const Hierarchy& hierarchy,
                                                      typename T::arrival_token&& token)
  {
    // Member types.
    requires /*is-arrival-token*/<typename T::arrival_token>;

    // Lifetime-related methods.
    { t.deinit(mapping_result, hierarchy) } -> cuda::std::same_as<void>;           // (1)
    { ct.view() } -> /*group-synchronizer-instance*/<MappingResult, Hierarchy>;    // (2)

    // Synchronization interface.
    { ct.do_sync(mapping_result, hierarchy) } -> cuda::std::same_as<void>;         // (3)
    { ct.do_sync_aligned(mapping_result, hierarchy) } -> cuda::std::same_as<void>; // (4)

    { ct.do_arrive(mapping_result, hierarchy) }
      -> cuda::std::same_as<typename T::arrival_token>;                            // (5)
    { ct.do_arrive_aligned(mapping_result, hierarchy) }
      -> cuda::std::same_as<typename T::arrival_token>;                            // (6)

    { ct.do_wait(mapping_result, hierarchy, cuda::std::move(token)) }
      -> cuda::std::same_as<void>;                                                 // (7)
    { ct.do_wait_aligned(mapping_result, hierarchy, cuda::std::move(token)) }
      -> cuda::std::same_as<void>;                                                 // (8)
  };

Where:

#. Deinitializes the synchronizer instance.
#. Creates a non-owning view over the synchronizer instance resources. It must also satisfy the *group-synchronizer-instance* concept and have the same ``arrival_token``.
#. Runs the synchronization. It doesn't require the units to be contextually aligned.
#. Runs the synchronization. The caller guarantees that all of the units are contextually aligned (execute the exact same statement in the code).
#. Arrives at the synchronization point and returns an arrival token.
#. Same as ``do_arrive(...)``, with contextually aligned units.
#. Waits for synchronization to complete using the token from the matching arrival on the same instance.
#. Same as ``do_wait(...)``, with contextually aligned units.

Group Synchronizer Instantiation Process
----------------------------------------

If a group takes a synchronizer, it must instantiate it with its ``Unit``, ``ParentGroup`` and ``MappingResult`` by calling the ``.make_instance(...)`` method.

Group Synchronizer Interface
----------------------------

Every group synchronizer type must implement the *group-synchronizer* interface:

.. code:: c++

  template <typename T, typename Unit, typename ParentGroup, typename MappingResult>
  concept /*group-synchronizer*/ = requires (T& t,
                                             const Unit& unit,
                                             const ParentGroup& parent_group,
                                             const MappingResult& mapping_result)
  {
    { t.make_instance(unit, parent_group, mapping_result) }
      -> /*group-synchronizer-instance*/<MappingResult, typename ParentGroup::hierarchy_type>;
  };

Where:

- ``Unit`` is the unit level type of the newly created group.
- ``ParentGroup`` is the parent group of the newly created group. It can be used to synchronize all units.
- ``MappingResult`` is the final mapping result that describes the newly created group.

Predefined Synchronizers
------------------------

.. toctree::
  :maxdepth: 1

  level_synchronizer
  lane_synchronizer
  interwarp_synchronizer
  barrier_synchronizer
