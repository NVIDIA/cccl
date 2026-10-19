import cupy as cp
from cuda.compute._device_copy_spike import copy_into, make_device_copy

src = cp.arange(16, dtype=cp.int32)
dst = cp.empty_like(src)

copy_into(src, dst)
cp.cuda.runtime.deviceSynchronize()

print(dst)
assert cp.all(dst == src)

# Reuse path for loop-like usage:
copy = make_device_copy(src, dst)
try:
    dst.fill(0)
    copy(src, dst)
    cp.cuda.runtime.deviceSynchronize()
    assert cp.all(dst == src)
finally:
    copy.close()

print("ok")
