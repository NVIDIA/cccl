from typing import List

import numpy as np


def random_array(size, dtype) -> np.typing.NDArray:
    rng = np.random.default_rng()
    if np.isdtype(dtype, "integral"):
        max_value = np.iinfo(dtype).max
        return rng.integers(max_value, size=size, dtype=dtype)
    elif np.isdtype(dtype, "real floating"):
        return rng.random(size=size, dtype=dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def type_to_problem_sizes(dtype) -> List[int]:
    if dtype in [np.uint8, np.int8]:
        return [2, 4, 5, 6]
    elif dtype in [np.uint16, np.int16]:
        return [4, 8, 12, 14]
    elif dtype in [np.uint32, np.int32, np.float32]:
        return [16, 20, 24, 28]
    elif dtype in [np.uint64, np.int64, np.float64]:
        return [16, 20, 24, 28]
    else:
        raise ValueError("Unsupported dtype")
