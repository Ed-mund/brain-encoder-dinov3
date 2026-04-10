import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: real dataset / slow GPU checks"
    )
