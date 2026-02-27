import cupy as cp
import numpy as np

import cuda.bench as bench

TYPE_MAP = {
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

SIGNED_TYPES = {k: TYPE_MAP[k] for k in ("I8", "I16", "I32", "I64", "F32", "F64")}
INTEGER_TYPES = {k: TYPE_MAP[k] for k in ("I8", "I16", "I32", "I64")}
FLOAT_TYPES = {k: TYPE_MAP[k] for k in ("F32", "F64")}

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


def generate_data_with_entropy(num_elements, dtype, entropy_str, stream):
    """Generate data with entropy-controlled range approximations."""
    probability = ENTROPY_TO_PROB[entropy_str]

    with stream:
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            if probability == 1.0:
                if dtype == np.int64:
                    data = cp.random.randint(
                        int(info.min), int(info.max), size=num_elements, dtype=np.int64
                    )
                else:
                    data = cp.random.randint(
                        int(info.min),
                        int(info.max) + 1,
                        size=num_elements,
                        dtype=np.int64,
                    ).astype(dtype)
            else:
                range_size = int((int(info.max) - int(info.min)) * probability)
                if range_size < 1:
                    range_size = 1
                if dtype == np.int64:
                    max_high = int(info.max)
                    if range_size > max_high:
                        range_size = max_high
                data = cp.random.randint(
                    0, range_size, size=num_elements, dtype=np.int64
                ).astype(dtype)
        else:
            info = np.finfo(dtype)
            if probability == 1.0:
                data = cp.random.uniform(-1, 1, size=num_elements).astype(dtype)
                data = data * info.max * 0.5
            else:
                scale = probability * info.max * 0.5
                data = cp.random.uniform(-scale, scale, size=num_elements).astype(dtype)

    return data


def generate_uniform_segment_offsets(num_elements, min_segment_size, max_segment_size):
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
    cutoff = int(cp.searchsorted(cumsum, num_elements, side="left").item())
    sizes = sizes[: cutoff + 1]
    prev = 0 if cutoff == 0 else int(cumsum[cutoff - 1].item())
    sizes[cutoff] = num_elements - prev

    offsets = cp.empty(cutoff + 2, dtype=cp.int64)
    offsets[0] = 0
    offsets[1:] = cp.cumsum(sizes)
    offsets[-1] = num_elements
    return offsets


def generate_power_law_offsets(num_elements, num_segments, zipf_param=1.5):
    if num_segments <= 0:
        return cp.asarray([0, num_elements], dtype=cp.int64)

    sizes = cp.random.zipf(zipf_param, size=num_segments).astype(cp.int64)
    sizes = cp.maximum(sizes, 1)

    min_total = num_segments
    remaining = num_elements - min_total
    if remaining < 0:
        remaining = 0

    scaled = sizes / sizes.sum() * remaining
    sizes = cp.floor(scaled).astype(cp.int64)
    remainder = int(remaining - sizes.sum().item())
    if remainder > 0:
        sizes[:remainder] += 1

    sizes += 1

    offsets = cp.empty(num_segments + 1, dtype=cp.int64)
    offsets[0] = 0
    offsets[1:] = cp.cumsum(sizes)
    offsets[-1] = num_elements
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
    cutoff = int(cp.searchsorted(cumsum, num_elements, side="left").item())
    sizes = sizes[: cutoff + 1]
    prev = 0 if cutoff == 0 else int(cumsum[cutoff - 1].item())
    sizes[cutoff] = num_elements - prev

    num_segments = sizes.size
    key_ids = cp.arange(num_segments, dtype=cp.int64)
    if np.issubdtype(key_dtype, np.integer):
        info = np.iinfo(key_dtype)
        range_size = int(info.max) - int(info.min) + 1
        key_ids = (key_ids % range_size) + int(info.min)

    key_ids = key_ids.astype(key_dtype, copy=False)

    with stream:
        keys = cp.repeat(key_ids, sizes)

    return keys
