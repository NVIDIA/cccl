# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


class BaseArray:
    __slots__ = ()

    def __new__(cls):
        msg = f"{cls.__name__} cannot be instantiated directly"
        raise NotImplementedError(msg)

    @staticmethod
    def _array(shape, dtype, alignment=None):
        from functools import reduce
        from operator import mul

        from ._common import normalize_dim_param

        dim = normalize_dim_param(shape)
        size = reduce(mul, dim)
        return [None] * size if size > 0 else []


class shared(BaseArray):
    @staticmethod
    def array(shape, dtype, alignment=None):
        """
        Create a shared memory array with the specified shape and dtype.

        Parameters
        ----------
        shape : int or tuple of int
            The shape of the array.
        dtype : data-type
            The desired data-type for the array.
        alignment : int, optional
            The alignment of the array in bytes.  If not specified,
            defaults to 1.

        Returns
        -------
        cuda.shared.array
            A shared memory array with the specified shape and dtype.
        """
        return BaseArray._array(shape, dtype, alignment)


class local(BaseArray):
    @staticmethod
    def array(shape, dtype, alignment=None):
        """
        Create a local memory array with the specified shape and dtype.

        Parameters
        ----------
        shape : int or tuple of int
            The shape of the array.
        dtype : data-type
            The desired data-type for the array.
        alignment : int, optional
            The alignment of the array in bytes.  If not specified,
            defaults to 1.

        Returns
        -------
        cuda.local.array
            A local memory array with the specified shape and dtype.
        """
        return BaseArray._array(shape, dtype, alignment)
