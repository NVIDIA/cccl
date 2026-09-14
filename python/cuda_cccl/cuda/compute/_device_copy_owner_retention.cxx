// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <atomic>

#include <_device_copy_owner_retention.h>

struct cccl_device_copy_owner_retention
{
  PyObject* source;
  PyObject* destination;
  cccl_device_copy_owner_retention* next;
};

namespace
{
std::atomic<cccl_device_copy_owner_retention*> completed_owners{nullptr};

void push_completed_owner(cccl_device_copy_owner_retention* const owners) noexcept
{
  auto* head = completed_owners.load();
  do
  {
    owners->next = head;
  } while (!completed_owners.compare_exchange_weak(head, owners));
}

int drain_completed_owners_pending(void*) noexcept
{
  (void) cccl_device_copy_drain_completed_owners_impl();
  return 0;
}

void CUDA_CB owners_ready(void* const user_data) noexcept
{
  auto* const owners = static_cast<cccl_device_copy_owner_retention*>(user_data);
  push_completed_owner(owners);

  // Object destruction is deferred to an interpreter thread. In particular,
  // a DLPack deleter may call CUDA APIs, which is forbidden from a CUDA host callback.
  (void) Py_AddPendingCall(drain_completed_owners_pending, nullptr);
}
} // namespace

extern "C" {
cccl_device_copy_owner_retention*
cccl_device_copy_create_owner_retention(PyObject* const source, PyObject* const destination) noexcept
{
  auto* const owners =
    static_cast<cccl_device_copy_owner_retention*>(PyMem_Malloc(sizeof(cccl_device_copy_owner_retention)));
  if (owners == nullptr)
  {
    PyErr_NoMemory();
    return nullptr;
  }

  Py_INCREF(source);
  Py_INCREF(destination);
  owners->source      = source;
  owners->destination = destination;
  owners->next        = nullptr;
  return owners;
}

void cccl_device_copy_release_owner_retention(cccl_device_copy_owner_retention* const owners) noexcept
{
  Py_DECREF(owners->destination);
  Py_DECREF(owners->source);
  PyMem_Free(owners);
}

CUresult
cccl_device_copy_schedule_owner_release(CUstream const stream, cccl_device_copy_owner_retention* const owners) noexcept
{
  return cuLaunchHostFunc(stream, owners_ready, owners);
}

Py_ssize_t cccl_device_copy_drain_completed_owners_impl() noexcept
{
  auto* owners     = completed_owners.exchange(nullptr);
  Py_ssize_t count = 0;

  while (owners != nullptr)
  {
    auto* const next = owners->next;
    cccl_device_copy_release_owner_retention(owners);
    owners = next;
    ++count;
  }

  return count;
}
} // extern "C"
