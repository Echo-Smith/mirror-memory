"""Debug plugin: print marker visibility for fixture_writes-marked tests."""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    marker = item.get_closest_marker("fixture_writes")
    if marker is not None:
        print(f"PROBE marker found at setup: {item.name} kwargs={marker.kwargs}")
    else:
        print(f"PROBE no marker at setup: {item.name}")
    yield
