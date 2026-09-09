import pytest
 
 
def pytest_addoption(parser):
    parser.addoption(
        "--path",
        action="store",
        default=None,
        help="Scratch directory containing system.xyz / dist_traj.pt for MD comparison tests",
    )
 
 
@pytest.fixture
def md_test_path(request):
    path = request.config.getoption("--path")
    if path is None:
        pytest.skip("--path not provided; run with `pytest --path <dir> ...`")
    return path
