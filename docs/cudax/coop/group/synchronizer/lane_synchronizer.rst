.. _cudax-coop-group-synchronizer-lane-synchronizer:

``cudax::coop::lane_synchronizer``
==================================

.. code:: cuda

  namespace cudax::coop {

  class lane_synchronizer
  {
  public:
    explicit lane_synchronizer() = default;

    template <typename Unit, typename ParentGroup, typename MappingResult>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const noexcept;
  };

  } // namespace cudax::coop

Overview
--------

``cudax::coop::lane_synchronizer`` is a synchronizer that uses ``__syncwarp(lane_mask)`` to synchronize the group, where ``lane_mask`` is the final group mapping result's lane mask. It can only be used with groups that group threads within the warp level.

The type is empty and is explicitly default constructible.
