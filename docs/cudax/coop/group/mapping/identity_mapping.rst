.. _cudax-coop-group-mapping-identity-mapping:

``cuda::experimental::coop::identity_mapping``
==============================================

.. code:: cuda

  namespace cuda::experimental::coop {

  class identity_mapping : public /*group-mapping-interface-for*/<identity_mapping>
  {
  public:
    explicit identity_mapping() = default;
  };

  } // namespace cuda::experimental::coop

Overview
--------

``cuda::experimental::coop::identity_mapping`` returns a copy of the previous mapping result, preserving its values, type, and properties. It is useful in generic code when no further partitioning is needed.

The type is empty and explicitly default constructible.
