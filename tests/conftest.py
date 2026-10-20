"""
With SCATTER_SMOOTH_REQUIRE_R set (as in CI's R job), the R comparisons must
run: R and rpy2 are checked first, so a missing R is an error, not a silent
skip (the tests skip only when rpy2 can't be imported or R can't start).
"""
import os

import pytest


def pytest_configure(config):
    if os.environ.get('SCATTER_SMOOTH_REQUIRE_R'):
        try:
            import rpy2.robjects as ro
            ro.r('1')
        except Exception as e:
            raise pytest.UsageError(f'SCATTER_SMOOTH_REQUIRE_R is set but R (rpy2) is not usable: {e}')
