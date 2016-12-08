"""
testing for the CW2 base function.
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

def test_L():
    for i in range(10):
        t = numpy.random.random()
        dy = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        y = numpy.random.random()
        val = a * dy * dy + b * (t * t - 1) * dy * dy * dy - y
        val_L = L(t, y, dy, a, b)
        assert type(val_L) == float, \
            "L does not create proper type values."
        assert numpy.allclose(val, val_L), \
            "L does not create proper values."

def test_dL_dy():
    for i in range(10):
        t = numpy.random.random()
        h = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        q = numpy.random.random(2)
        dy = q[1]
        y = q[0]
        val = (a * dy * dy + b * (t * t - 1) * dy * dy * dy - (y+h) - \
            (a * dy * dy + b * (t * t - 1) * dy * dy * dy - (y-h))) / (2*h)
        val_dl = dL_dy(L, t, q, h, a, b)
        assert type(val_dl) == float or type(val_dl) == numpy.float64, \
            "dL_dy didn't create proper type"
        assert numpy.allclose(val, val_dl), \
            "values don't match"


def test_d2L_dtdydot():
    for i in range(10):
        t = numpy.random.random()
        h = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        q = numpy.random.random(2)
        dy = q[1]
        y = q[0]
        val_dl = d2L_dtdydot(L, t, q, h, a, b)
        t1 = t + h
        t2 = t - h
        dy1 = q[1]+h
        dy2 = q[1]-h
        k1 = a * dy1 * dy1 + b * (t1 * t1 - 1) * dy1 * dy1 * dy1 - y
        k2 = a * dy2 * dy2 + b * (t1 * t1 - 1) * dy2 * dy2 * dy2 - y
        k3 = a * dy1 * dy1 + b * (t2 * t2 - 1) * dy1 * dy1 * dy1 - y
        k4 = a * dy2 * dy2 + b * (t2 * t2 - 1) * dy2 * dy2 * dy2 - y

        val = (k1 - k2 - k3 + k4) / (4 * h**2)

        assert type(val_dl) == float or type(val_dl) == numpy.float64, \
            "d2L_dtdydot didn't create proper type"
        assert numpy.allclose(val, val_dl), \
            "values don't match"


def test_d2L_dydydot():
    for i in range(10):
        t = numpy.random.random()
        h = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        q = numpy.random.random(2)
        dy = q[1]
        y = q[0]
        val_dl = d2L_dydydot(L, t, q, h, a, b)
        y1 = t + h
        y2 = t - h
        dy1 = q[1]+h
        dy2 = q[1]-h
        k1 = a * dy1 * dy1 + b * (t * t - 1) * dy1 * dy1 * dy1 - y1
        k2 = a * dy1 * dy1 + b * (t * t - 1) * dy1 * dy1 * dy1 - y2
        k3 = a * dy2 * dy2 + b * (t * t - 1) * dy2 * dy2 * dy2 - y2
        k4 = a * dy2 * dy2 + b * (t * t - 1) * dy2 * dy2 * dy2 - y1

        val = q[1] * (k1 - k2 + k3 - k4) / (4*h**2)

        assert type(val_dl) == float or type(val_dl) == numpy.float64, \
            "dL_dydydot didn't create proper type"
        assert numpy.allclose(val, val_dl), \
            "values don't match"


def test_d2L_dydot2():
    for i in range(10):
        t = numpy.random.random()
        h = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        q = numpy.random.random(2)
        dy = q[1]
        y = q[0]
        val_dl = d2L_dydot2(L, t, q, h, a, b)
        dy0 = q[1]
        dy1 = q[1]+h
        dy2 = q[1]-h
        k1 = a * dy1 * dy1 + b * (t * t - 1) * dy1 * dy1 * dy1 - y
        k2 = a * dy0 * dy0 + b * (t * t - 1) * dy0 * dy0 * dy0 - y
        k3 = a * dy2 * dy2 + b * (t * t - 1) * dy2 * dy2 * dy2 - y

        val = (k1 - 2 * k2 + k3) / h**2

        assert type(val_dl) == float or type(val_dl) == numpy.float64, \
            "dL_dydot2 didn't create proper type"
        assert numpy.allclose(val, val_dl), \
            "values don't match"


def test_f():
    for i in range(10):
        t = numpy.random.random()
        h = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        q = numpy.random.random(2)
        dy = q[1]
        y = q[0]
        val_f = f(L, t, q, h, a, b)

        k1 = dL_dy(L, t, q, h, a, b)
        k2 = d2L_dtdydot(L, t, q, h, a, b)
        k3 = d2L_dydydot(L, t, q, h, a, b)
        k4 = d2L_dydot2(L, t, q, h, a, b)
        val = (k1 - k2 - k3) / k4

        assert type(val_f) == float or type(val_f) == numpy.float64, \
            "f didn't create proper type"
        assert numpy.allclose(val, val_f), \
            "values don't match"


def test_dq_dt():
    for i in range(10):
        t = numpy.random.random()
        h = numpy.random.random()
        a = numpy.random.random()
        b = numpy.random.random()
        q = numpy.random.random(2)
        dy = q[1]
        y = q[0]
        dq = dq_dt(q, t, L, h, a, b)
        ddy = f(L, t, q, h, a, b)

        assert (dq.shape == q.shape), \
            "not the proper shape."
        assert numpy.allclose(dy, dq[0]), \
            "dq_dt doest not create proper ydot val"
        assert numpy.allclose(ddy, dq[1]), \
            "dq_dt doest not create proper yddot val"


if __name__ == "__main__":
    pytest.main()
