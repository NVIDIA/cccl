from cuda.compute._caching import CachableFunction

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

    func1 = make_func()
    func2 = make_func()

    assert CachableFunction(func1) == CachableFunction(func2)
