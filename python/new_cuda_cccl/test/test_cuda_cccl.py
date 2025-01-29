import cuda.cccl as c4l


def test_headers():
    inc_paths = c4l.get_include_paths()
    assert hasattr(inc_paths, "cuda")
    assert hasattr(inc_paths, "cub")
    assert hasattr(inc_paths, "libcudacxx")
    assert hasattr(inc_paths, "thrust")
    tpl = inc_paths.as_tuple()
    assert len(tpl) == 4
    thrust_, cub_, cudacxx_, cuda_ = tpl
    assert cuda_ == inc_paths.cuda
    assert cub_ == inc_paths.cub
    assert cudacxx_ == inc_paths.libcudacxx
    assert thrust_ == inc_paths.thrust


def test_version():
    v = c4l.__version__
    assert isinstance(v, str)
