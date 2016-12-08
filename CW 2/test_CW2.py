"""
testing for the CW2
"""

import pytest
from main_CW2 import *


def test_P():
    for i in range(10):
        t = numpy.random.random()
        dy = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        val = a * dy * dy + b * (t * t - 1) * dy * dy * dy
        val_P = P(t, dy, a, b)
        assert type(val_P) == float,\
            "P does not create proper type values."
        assert numpy.allclose(val, val_P), \
            "P does not create proper values."


if __name__ == "__main__":
    pytest.main()
