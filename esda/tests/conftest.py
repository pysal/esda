import pytest


def pytest_configure(config):  # noqa: ARG001

    pytest.WARN_ALT_HYPOTHESIS_DEPR = pytest.warns(
        DeprecationWarning,
        match=(
            "The alternative hypothesis for conditional randomization "
            "is changing in the next major release of esda."
        ),
    )
