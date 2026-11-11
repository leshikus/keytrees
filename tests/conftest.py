"""Mark the exhaustive enumerations that take more than a few seconds as slow.

`pytest` skips them by default; `pytest -m ""` or `scripts/reproduce.sh` runs all.
"""
import pytest

SLOW = (
    "test_gradient_analytic_4.py::test_exact_enumeration[5-5]",
    "test_gradient_analytic_4.py::test_exact_enumeration[5-4]",
    "test_gradient_analytic_4.py::test_exact_enumeration[4-6]",
    "test_gradient_analytic_4.py::test_exact_enumeration[4-7]",
    "test_gradient_analytic_4.py::test_exact_enumeration[4-8]",
    "test_gradient_analytic_4.py::test_exact_enumeration[4-9]",
    "test_gradient_analytic_4.py::test_exact_enumeration[4-10]",
    "test_gradient_analytic_3.py::test_c1_bound_below_exact_window",
    "test_elimination_keys.py::test_exact_means_against_brute_force[dist-4-5-1]",
    "test_elimination_keys.py::test_exact_means_against_brute_force[all-3-4-1]",
    "test_gradient_analytic_7.py::test_pollution_averaging[4-4]",
)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: exhaustive check taking more than a few seconds")


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.nodeid.split("tests/")[-1].endswith(SLOW):
            item.add_marker(pytest.mark.slow)
