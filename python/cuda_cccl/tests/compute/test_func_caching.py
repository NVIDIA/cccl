import numpy as np

from cuda.compute._caching import CachableFunction, _make_hashable

global_x = 1


def test_func_caching_basic():
    def func(x):
        return x

    f1 = CachableFunction(func)

    def func(x):
        return x

    f2 = CachableFunction(func)

    assert f1 == f2


def test_func_caching_different_names():
    def func(x):
        return x

    f1 = CachableFunction(func)

    def func2(x):
        return x

    f2 = CachableFunction(func2)

    assert f1 != f2


def test_func_caching_different_code():
    def func(x):
        return x

    f1 = CachableFunction(func)

    def func(x):
        return x + 1

    f2 = CachableFunction(func)
    assert f1 != f2


def test_func_caching_with_closure():
    def factory(x):
        def func(y):
            return x + y

        return func

    f1 = CachableFunction(factory(1))
    f2 = CachableFunction(factory(1))
    assert f1 == f2

    f3 = CachableFunction(factory(2))
    assert f1 != f3


def test_func_caching_with_numpy_numeric_scalar_closure():
    def factory(indexlength, regularsize):
        index_dtype = np.int64
        idx_len = index_dtype(indexlength)
        reg_size = index_dtype(regularsize)

        def func(counter):
            return counter % idx_len + reg_size

        return func

    f1 = CachableFunction(factory(100_000, 16))
    f2 = CachableFunction(factory(100_000, 16))
    assert f1 == f2

    f3 = CachableFunction(factory(100_000, 32))
    assert f1 != f3


def test_make_hashable_python_scalars_keyed_by_value():
    # Regression test for gh-9626: plain Python int/float/bool scalars used to
    # fall through to ``id(value)``, so two equal-valued but distinct (non-
    # interned) objects produced different cache keys and missed the cache.
    # int(str(...)) forces fresh, non-interned objects.
    a = int(str(10**6))
    b = int(str(10**6))
    assert a is not b
    assert _make_hashable(a) == _make_hashable(b)

    x = float(str(3.5))
    y = float(str(3.5))
    assert _make_hashable(x) == _make_hashable(y)

    # Distinct values, types, and bool-vs-int must not collide.
    assert _make_hashable(a) != _make_hashable(int(str(10**6 + 1)))
    assert _make_hashable(1) != _make_hashable(1.0)
    assert _make_hashable(True) != _make_hashable(1)


def test_func_caching_with_python_scalar_closure():
    # gh-9626: closures capturing equal-valued Python scalars must compare
    # equal so the algorithm build cache hits instead of rebuilding every call.
    def factory(indexlength, regularsize):
        # int(str(...)) forces fresh, non-interned int objects.
        idx_len = int(str(indexlength))
        reg_size = int(str(regularsize))

        def func(counter):
            return counter % idx_len + reg_size

        return func

    f1 = CachableFunction(factory(100_000, 16))
    f2 = CachableFunction(factory(100_000, 16))
    assert f1 == f2

    f3 = CachableFunction(factory(100_000, 32))
    assert f1 != f3


def test_func_caching_with_global_variable():
    global global_x

    def func(y):
        return global_x + y

    f1 = CachableFunction(func)
    f2 = CachableFunction(func)
    assert f1 == f2

    global_x = 2
    f3 = CachableFunction(func)
    assert f1 != f3


def test_func_caching_wrapped_cuda_jit_function():
    import numba.cuda

    def make_func():
        @numba.cuda.jit
        def inner(x):
            return x

        def func(x):
            return inner(x) + 1

        return func

    def make_func2():
        @numba.cuda.jit
        def inner(x):
            return 2 * x

        def func(x):
            return inner(x) + 1

        return func

    func1 = make_func()
    func2 = make_func()
    func3 = make_func2()

    assert CachableFunction(func1) == CachableFunction(func2)
    assert CachableFunction(func1) != CachableFunction(func3)


def test_func_caching_with_global_np_ufunc():
    def make_func():
        def func(x):
            return np.argmin(x) + 1

        return func

    def make_func2():
        def func(x):
            return np.argmax(x) + 1

        return func

    func1 = make_func()
    func2 = make_func2()

    assert CachableFunction(func1) != CachableFunction(func2)


def test_func_caching_with_aliased_np_ufunc():
    def make_func1():
        amin = np.argmin

        def func(x):
            return amin(x) + 1

        return func

    def make_func2():
        amax = np.argmax

        def func(x):
            return amax(x) + 1

        return func

    func1 = make_func1()
    func2 = make_func2()

    assert CachableFunction(func1) != CachableFunction(func2)
