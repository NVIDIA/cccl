.. _cudax-coop-group-synchronizer-level-synchronizer:

``cuda::experimental::coop::level_synchronizer``
================================================

.. code:: cuda

  namespace cuda::experimental::coop {

  class level_synchronizer
  {
  public:
    explicit level_synchronizer() = default;

    template <typename Unit, typename ParentGroup, typename MappingResult>
    [[nodiscard]] __device__
    auto make_instance(const Unit&, const ParentGroup&, const MappingResult&) const noexcept;
  };

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::level_synchronizer`` is a synchronizer that synchronizes all threads within the group's level. The synchronization behaviour is equivalent to:

+--------------+---------------------------------------------------------------------+-----------------------------------------------------------------+
| Level        | ``.do_sync(...)``                                                   | ``.do_sync_aligned(...)``                                       |
+==============+=====================================================================+=================================================================+
| Thread       | Noop                                                                | Noop                                                            |
+--------------+---------------------------------------------------------------------+-----------------------------------------------------------------+
| Warp         | ``__syncwarp()``                                                    | ``__syncwarp()``                                                |
+--------------+---------------------------------------------------------------------+-----------------------------------------------------------------+
| Block        | ``__barrier_sync(0)``                                               | ``__syncthreads()``                                             |
+--------------+---------------------------------------------------------------------+-----------------------------------------------------------------+
| Cluster      | Cluster barrier arrive + wait                                       | Aligned cluster barrier arrive + wait                           |
+--------------+---------------------------------------------------------------------+-----------------------------------------------------------------+
| Grid         | ``__barrier_sync(0)`` + global barrier sync + ``__barrier_sync(0)`` | ``__syncthreads()`` + global barrier sync + ``__syncthreads()`` |
+--------------+---------------------------------------------------------------------+-----------------------------------------------------------------+

.. note:: Grid synchronization requires a cooperative kernel launch on a device that supports it. With ``cuda::launch``, include ``cuda::cooperative_launch{}`` in the configuration passed to ``cuda::make_config``.

The type is empty and is explicitly default constructible.
