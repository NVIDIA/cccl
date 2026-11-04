.. _cudax-coop-group-synchronizer-lane-synchronizer:

``cuda::experimental::coop::lane_synchronizer``
===============================================

.. code:: cuda

  namespace cuda::experimental::coop {

  class lane_synchronizer
  {
  public:
    explicit lane_synchronizer() = default;

    template <typename Unit, typename ParentGroup, typename MappingResult>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const;
  };

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::lane_synchronizer`` is a synchronizer that uses ``__syncwarp(lane_mask)`` to synchronize the group, where ``lane_mask`` is the final group mapping result's lane mask. It can only be used with groups that group threads within the warp level.

The instance's ``do_sync(...)`` and ``do_sync_aligned(...)`` both call ``__syncwarp(lane_mask)``.

The ``do_arrive()`` and ``do_wait(...)`` (and aligned variants) are emulated by calling ``do_sync(...)`` in ``do_wait(...)``.

The type is empty and is explicitly default constructible.
