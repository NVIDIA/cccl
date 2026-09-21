#pragma once

#include <thrust/detail/config.h>

#include <nv/target>

// A type whose copy constructor records whether it ran as device or host code. Used to verify
// that algorithms like uninitialized_copy/uninitialized_fill actually construct elements on the
// system they claim to target.
struct CopyConstructTest
{
  CopyConstructTest() = default;

  _CCCL_HOST_DEVICE CopyConstructTest(const CopyConstructTest&)
  {
    NV_IF_TARGET(NV_IS_DEVICE,
                 (copy_constructed_on_device = true; copy_constructed_on_host = false;),
                 (copy_constructed_on_device = false; copy_constructed_on_host = true;));
  }

  CopyConstructTest& operator=(const CopyConstructTest&) = default;

  bool copy_constructed_on_host{false};
  bool copy_constructed_on_device{false};
};

struct is_copy_constructed_on_device
{
  _CCCL_HOST_DEVICE bool operator()(const CopyConstructTest& t) const
  {
    return t.copy_constructed_on_device;
  }
};

struct is_copy_constructed_on_host
{
  _CCCL_HOST_DEVICE bool operator()(const CopyConstructTest& t) const
  {
    return t.copy_constructed_on_host;
  }
};
