
import pytest

def pytest_collectstart(collector):
    collector.skip_compare += 'image/png', 'image/svg+xml',