import pytest

# Check that tests marked no_numba fail fast if they import numba.
pytestmark = pytest.mark.no_numba


@pytest.mark.no_numba
def test_import_numba_raises():
    with pytest.raises(
        ImportError, match="This test is marked 'no_numba' but attempted to import it"
    ):
        import numba.cuda  # noqa: F401
