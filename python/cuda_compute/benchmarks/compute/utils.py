import cupy as cp
import numpy as np

import cuda.bench as bench

ALL_TYPES = {
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "U16": np.uint16,
    "U32": np.uint32,
    "U64": np.uint64,
    "F32": np.float32,
    "F64": np.float64,
}

SIGNED_TYPES = {k: ALL_TYPES[k] for k in ("I8", "I16", "I32", "I64", "F32", "F64")}
FLOAT_TYPES = {k: ALL_TYPES[k] for k in ("F32", "F64")}

# Matches C++ integral_types = {int8_t, int16_t, int32_t, int64_t}
INTEGRAL_TYPES = {k: ALL_TYPES[k] for k in ("I8", "I16", "I32", "I64")}

# Matches C++ fundamental_types = {int8..int64, [int128,] float, double}
# int128 is excluded because it is not supported by numpy/cupy.
FUNDAMENTAL_TYPES = {k: ALL_TYPES[k] for k in ("I8", "I16", "I32", "I64", "F32", "F64")}

ENTROPY_TO_STEPS = {
    "1.000": 0,
    "0.811": 1,
    "0.544": 2,
    "0.337": 3,
    "0.201": 4,
    "0.000": 0,
}

ENTROPY_TO_PROB = {
    "1.000": 1.0,
    "0.811": 0.811,
    "0.544": 0.544,
    "0.337": 0.337,
    "0.201": 0.201,
    "0.000": 0.0,
}


def as_cupy_stream(cs: bench.CudaStream) -> cp.cuda.Stream:
    """Convert nvbench CudaStream to CuPy Stream."""
    return cp.cuda.ExternalStream(cs.addressof())


def lerp_min_max(dtype, probability):
    """Interpolate between min/max for dtype like nvbench_helper.cuh."""
    if probability == 1.0:
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).max
        return np.finfo(dtype).max

    if np.issubdtype(dtype, np.integer):
        min_val = float(np.iinfo(dtype).min)
        max_val = float(np.iinfo(dtype).max)
    else:
        min_val = float(np.finfo(dtype).min)
        max_val = float(np.finfo(dtype).max)

    return dtype(min_val + probability * (max_val - min_val))


def _bitwise_and(a, b, dtype):
    if np.issubdtype(dtype, np.floating):
        view_dtype = cp.uint32 if dtype == np.float32 else cp.uint64
        return (a.view(view_dtype) & b.view(view_dtype)).view(dtype)
    return a & b


def _uniform_random(num_elements, dtype, min_val, max_val):
    rand = cp.random.random(num_elements)
    if np.issubdtype(dtype, np.floating):
        return ((float(max_val) - float(min_val)) * rand + float(min_val)).astype(dtype)
    min_f = float(min_val)
    max_f = float(max_val)
    return cp.floor((max_f - min_f + 1) * rand + min_f).astype(dtype)


def generate_data_with_entropy(
    num_elements, dtype, entropy_str, stream, min_val=None, max_val=None
):
    """Generate data with nvbench_helper-style bit entropy."""
    if min_val is None or max_val is None:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            default_min = info.min
            default_max = info.max
        else:
            info = np.finfo(dtype)
            default_min = info.tiny
            default_max = info.max
        min_val = default_min if min_val is None else min_val
        max_val = default_max if max_val is None else max_val

    steps = ENTROPY_TO_STEPS[entropy_str]

    with stream:
        if entropy_str == "0.000":
            scalar = _uniform_random(1, dtype, min_val, max_val)[0]
            data = cp.full(num_elements, scalar, dtype=dtype)
        else:
            data = _uniform_random(num_elements, dtype, min_val, max_val)
            for _ in range(steps):
                tmp = _uniform_random(num_elements, dtype, min_val, max_val)
                data = _bitwise_and(data, tmp, dtype)

    return data


def generate_uniform_segment_offsets(num_elements, min_segment_size, max_segment_size):
    num_elements = int(num_elements)
    if min_segment_size <= 0:
        raise ValueError("min_segment_size must be positive")
    if max_segment_size < min_segment_size:
        raise ValueError("max_segment_size must be >= min_segment_size")

    num_segments_est = int(np.ceil(num_elements / min_segment_size))
    if num_segments_est > np.iinfo(np.int32).max:
        raise MemoryError("Too many segments for int32 offsets")

    sizes = cp.random.randint(
        min_segment_size,
        max_segment_size + 1,
        size=num_segments_est,
        dtype=cp.int64,
    )
    cumsum = cp.cumsum(sizes)
    cutoff = int(
        cp.searchsorted(
            cumsum, cp.asarray(num_elements, dtype=cp.int64), side="left"
        ).item()
    )
    sizes = sizes[: cutoff + 1]
    prev = 0 if cutoff == 0 else int(cumsum[cutoff - 1].item())
    sizes[cutoff] = num_elements - prev

    offsets = cp.empty(cutoff + 2, dtype=cp.int64)
    offsets[0] = 0
    offsets[1:] = cp.cumsum(sizes)
    offsets[-1] = num_elements
    return offsets


def generate_power_law_offsets(num_elements, num_segments):
    if num_segments <= 0:
        return cp.asarray([0, num_elements], dtype=cp.int64)

    # Mirror nvbench_helper power-law generation:
    # draw log-normal samples, normalize to total elements,
    # floor to integer segment sizes, then distribute remainder
    # across the first `diff` segments.
    samples = cp.random.lognormal(3.0, 1.2, size=num_segments)
    if int(cp.count_nonzero(samples).item()) == 0:
        samples = cp.ones(num_segments, dtype=cp.float64)

    sample_sum = float(samples.sum().item())
    sizes = cp.floor(samples * num_elements / sample_sum).astype(cp.int64)

    diff = int(num_elements - sizes.sum().item())
    if diff > 0:
        sizes[:diff] += 1

    offsets = cp.empty(num_segments + 1, dtype=cp.int64)
    offsets[0] = 0
    offsets[1:] = cp.cumsum(sizes)
    return offsets


def generate_fixed_segment_offsets(num_elements, segment_size, stream):
    num_segments = max(1, num_elements // segment_size)
    actual_elements = num_segments * segment_size

    with stream:
        start_offsets = cp.arange(0, actual_elements, segment_size, dtype=np.int64)
        end_offsets = cp.arange(
            segment_size, actual_elements + 1, segment_size, dtype=np.int64
        )

    return start_offsets, end_offsets, num_segments, actual_elements


def generate_key_segments(
    num_elements, key_dtype, min_segment_size, max_segment_size, stream
):
    """Generate GPU key segments (runs of equal keys) matching C++ generate.uniform.key_segments.

    All computation stays on GPU via CuPy. We avoid ``cp.repeat`` (which
    doesn't accept a device array for *repeats*) by building a segment-id
    array through ``cp.searchsorted`` on cumulative offsets instead.
    """
    num_elements = int(num_elements)
    if min_segment_size <= 0:
        raise ValueError("min_segment_size must be positive")
    if max_segment_size < min_segment_size:
        raise ValueError("max_segment_size must be >= min_segment_size")

    num_segments_est = int(np.ceil(num_elements / min_segment_size))
    if num_segments_est > np.iinfo(np.int32).max:
        raise MemoryError("Too many segments for int32 offsets")

    with stream:
        sizes = cp.random.randint(
            min_segment_size,
            max_segment_size + 1,
            size=num_segments_est,
            dtype=cp.int64,
        )
        cumsum = cp.cumsum(sizes)

        # Find how many full segments fit within num_elements
        cutoff = int(
            cp.searchsorted(
                cumsum, cp.asarray(num_elements, dtype=cp.int64), side="left"
            ).item()
        )
        sizes = sizes[: cutoff + 1]
        prev = 0 if cutoff == 0 else int(cumsum[cutoff - 1].item())
        sizes[cutoff] = num_elements - prev

        # Build cumulative offsets for the final segments
        offsets = cp.empty(cutoff + 2, dtype=cp.int64)
        offsets[0] = 0
        offsets[1:] = cp.cumsum(sizes)

        # Instead of cp.repeat (which doesn't support device repeats),
        # use searchsorted to map each element index to its segment id.
        indices = cp.arange(num_elements, dtype=cp.int64)
        # searchsorted(offsets[1:], indices, side="right") gives the segment id
        segment_ids = cp.searchsorted(offsets[1:], indices, side="right")

        # Map segment ids to key values, wrapping within dtype range
        if np.issubdtype(key_dtype, np.integer):
            info = np.iinfo(key_dtype)
            if np.dtype(key_dtype).itemsize < 8:
                range_size = int(info.max) - int(info.min) + 1
                keys = ((segment_ids % range_size) + int(info.min)).astype(
                    key_dtype, copy=False
                )
            else:
                # For int64, avoid Python overflow: just cast directly
                keys = segment_ids.astype(key_dtype, copy=False)
        else:
            keys = segment_ids.astype(key_dtype, copy=False)

    return keys
