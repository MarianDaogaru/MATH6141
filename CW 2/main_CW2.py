

import numpy


def P(t, dy, alpha, beta):
    return alpha * dy**2 + beta * (t**2 - 1) * dy**3


def L(t, dy, y, alpha, beta):
    return P(t, dy, alpha, beta) - y

