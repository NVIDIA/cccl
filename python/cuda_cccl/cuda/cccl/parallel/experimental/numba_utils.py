from numba import cuda


def get_inferred_return_type(op, args: tuple):
    _, return_type = cuda.compile(op, args)
    return return_type
