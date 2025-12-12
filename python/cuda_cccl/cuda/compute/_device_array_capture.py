"""
Extends Numba CUDA to enable global capture of objects implementing
__cuda_array_interface__.

HOW IT WORKS:
1. Patches typeof_impl to check for DeviceArrayLike objects
2. Defines a DeviceArrayType (subclass of Array) to represent DeviceArrayLike objects
3. Registers data model and lower_constant for the type
"""

import numpy as np
from numba.cuda import types
from numba.cuda.core.imputils import builtin_registry
from numba.cuda.datamodel.models import ArrayModel
from numba.cuda.datamodel.registry import register_default
from numba.cuda.np import numpy_support
from numba.cuda.typing import typeof as typeof_module

from .typing import DeviceArrayLike

# =============================================================================
# Step 1: Define a custom Array subclass for DeviceArrayLike objects
# =============================================================================


class DeviceArrayType(types.Array):
    """
    Type for any DeviceArrayLike object.

    By subclassing Array and overriding copy() to ignore readonly,
    we ensure these arrays remain mutable when captured from globals.
    """

    def __init__(self, dtype, ndim, layout, readonly=False, aligned=True):
        type_name = "device_array"
        if readonly:
            type_name = "readonly " + type_name
        name = f"{type_name}({dtype}, {ndim}d, {layout})"
        super().__init__(
            dtype, ndim, layout, readonly=readonly, name=name, aligned=aligned
        )

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        # IGNORE readonly parameter - CAI objects are device memory references
        # that should remain mutable
        return DeviceArrayType(
            dtype=dtype if dtype is not None else self.dtype,
            ndim=ndim if ndim is not None else self.ndim,
            layout=layout if layout is not None else self.layout,
            readonly=not self.mutable,  # Preserve original mutability
            aligned=self.aligned,
        )


# =============================================================================
# Step 2: Define _typeof_device_array to type DeviceArrayLike objects
# =============================================================================


def _typeof_device_array(val, c):
    """Type of a DeviceArrayLike object when captured as a constant."""

    # don't affect typing of arguments:
    if c.purpose == typeof_module.Purpose.argument:
        return None

    interface = val.__cuda_array_interface__

    dtype = numpy_support.from_dtype(np.dtype(interface["typestr"]))
    shape = interface["shape"]
    ndim = len(shape)
    strides = interface.get("strides")

    # Determine layout
    if ndim == 0:
        layout = "C"
    elif strides is None:
        layout = "C"
    else:
        itemsize = np.dtype(interface["typestr"]).itemsize
        # Check C-contiguous
        c_strides = []
        stride = itemsize
        for i in range(ndim - 1, -1, -1):
            c_strides.insert(0, stride)
            stride *= shape[i]

        if tuple(strides) == tuple(c_strides):
            layout = "C"
        else:
            # Check F-contiguous
            f_strides = []
            stride = itemsize
            for i in range(ndim):
                f_strides.append(stride)
                stride *= shape[i]
            if tuple(strides) == tuple(f_strides):
                layout = "F"
            else:
                layout = "A"

    readonly = interface["data"][1]
    return DeviceArrayType(dtype, ndim, layout, readonly=readonly)


# =============================================================================
# Step 3: Patch typeof_impl to handle CAI objects
# =============================================================================

# Get the original generic fallback
_original_typeof_impl = typeof_module.typeof_impl
_original_generic = _original_typeof_impl.dispatch(object)


def _patched_generic(val, c):
    """
    Patched generic typeof handler that checks for DeviceArrayLike objects.
    """
    # Check for CAI FIRST
    if isinstance(val, DeviceArrayLike):
        return _typeof_device_array(val, c)

    # Fall back to original behavior
    return _original_generic(val, c)


# Register our patched handler for the generic object case
_original_typeof_impl.register(object)(_patched_generic)


# =============================================================================
# Step 4: Register data model for our type
# =============================================================================


register_default(DeviceArrayType)(ArrayModel)
# =============================================================================
# Step 5: Register lower_constant for our type
# =============================================================================


@builtin_registry.lower_constant(DeviceArrayType)
def lower_device_array(context, builder, ty, pyval):
    """
    Lower DeviceArrayLike objects by embedding the device pointer as a constant.
    """
    interface = pyval.__cuda_array_interface__

    shape = interface["shape"]
    strides = interface.get("strides")
    data_ptr = interface["data"][0]
    typestr = interface["typestr"]

    # Calculate strides if not provided (C-contiguous)
    if strides is None:
        itemsize = np.dtype(typestr).itemsize
        ndim = len(shape)
        strides = []
        stride = itemsize
        for i in range(ndim - 1, -1, -1):
            strides.insert(0, stride)
            stride *= shape[i]
        strides = tuple(strides)

    # Embed device pointer as constant
    llvoidptr = context.get_value_type(types.voidptr)
    data = context.get_constant(types.uintp, data_ptr).inttoptr(llvoidptr)

    # Build array structure
    ary = context.make_array(ty)(context, builder)
    kshape = [context.get_constant(types.intp, s) for s in shape]
    kstrides = [context.get_constant(types.intp, s) for s in strides]
    itemsize = np.dtype(typestr).itemsize

    context.populate_array(
        ary,
        data=builder.bitcast(data, ary.data.type),
        shape=kshape,
        strides=kstrides,
        itemsize=context.get_constant(types.intp, itemsize),
        parent=None,
        meminfo=None,
    )

    return ary._getvalue()
