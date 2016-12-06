import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
from scipy.optimize import newton
import types

# the boundary values given in the exercise
y_A = 1.
y_B = 0.9


def P(t, y_dot, alpha, beta):
    """
    The penalty function which describes how the factory maintenance cost
    will increase if the usage is kept high.
    it is used to calculate the optimum path for decreasing the factory
    capacity, while maintaining high profit.

    Parameters
    t - float
        time at which the function is evaluated

    y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    p (val) - float
            the value of Penalty of the machine at t & y_dot
    """
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in P. Current type is: {}.".format(type(t))
    assert type(y_dot) == float or type(y_dot) == numpy.float64 or type(y_dot) == numpy.float, \
        "y_dot is not the supported type in P. Current type is: {}.".format(type(y_dot))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in P. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in P. Current type is: {}.".format(type(beta))

    p = alpha * y_dot**2 + beta * (t**2 - 1) * y_dot**3
    assert type(p) == float or type(p) == numpy.float64 or type(p) == numpy.float, \
        "p is not the supported type in P. Current type is: {}.".format(type(p))

    return p


def L(t, y, y_dot, alpha, beta):
    """
    The Lagrangiang function L(t, y(t), y_dot(t), ...) which has to be
    minimised such that the profit is maximised. S, the loss of the factory,
    is described as:
        S = integrate(P - y) dt from 0 to 1
    Thus leading to L(t, y(t), y_dot(t), .. ) = P - y.
    This function returs the value of L at a certain time t.

    Parameters:
    t - float
        time at which the function is evaluated

    y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)

    y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    L(val) - float
            the value of the lagrangian function at a particular time t.
    """
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in L. Current type is: {}.".format(type(t))
    assert type(y) == float or type(y) == numpy.float64 or type(y) == numpy.float, \
        "y is not the supported type in L. Current type is: {}.".format(type(y))
    assert type(y_dot) == float or type(y_dot) == numpy.float64 or type(y_dot) == numpy.float, \
        "y_dot is not the supported type in L. Current type is: {}.".format(type(y_dot))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in L. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in L. Current type is: {}.".format(type(beta))

    l = P(t, y_dot, alpha, beta) - y
    assert type(l) == float or type(l) == numpy.float64 or type(l) == numpy.float, \
        "l is not the supported type in L. Current type is: {}.".format(type(l))

    return l


def dL_dy(L, t, q, h, alpha, beta):
    """
    This function returns the value of the central differencing of L wrt y,
    at a given time t, with a step-size h.

    As L is a function of t, y, y_dot, dL/dy using the central differencing
    implemented in this function results in:

    dL/dy = (L(t, y+h, y_dot) - L(t, y-h, y_dot))/(2h)

    Parameters:
    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    t - float
        time at which the function is evaluated

    q - numpy array (2,)
        array containing the functions y & y_dot
        y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)
        y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    dL/dy - float
            the value of the central differencing of L wrt y
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in dL_dy. It is: {}.".format(type(L))
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in dL_dy. Current type is: {}.".format(type(t))
    assert type(q) == numpy.ndarray or type(q) == list, \
        "q is not the supported type in dL_dy. Current type is: {}.".format(type(q))
    assert type(q[0]) == float or type(q[0]) == numpy.float64 or type(q[0]) == numpy.float, \
        "y is not the supported type in dL_dy. Current type is: {}.".format(type(q[0]))
    assert type(q[1]) == float or type(q[1]) == numpy.float64 or type(q[1]) == numpy.float, \
        "y_dot is not the supported type in dL_dy. Current type is: {}.".format(type(q[1]))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in dL_dy. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in dL_dy. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in dL_dy. Current type is: {}.".format(type(beta))

    dl = (L(t, q[0]+h, q[1], alpha, beta) - L(t, q[0]-h, q[1], alpha, beta)) / (2*h)
    assert type(dl) == float or type(dl) == numpy.float64 or type(dl) == numpy.float, \
        "dl is not the supported type in dL_dy. Current type is: {}.".format(type(dl))

    return dl


def d2L_dtdydot(L, t, q, h, alpha, beta):
    """
    This function returns the value of the partial derivative of L wrt t &
    y_dot, by using the central differencing method.

    As L is a function of (t, y, y_dot), the partial derivate wrt t & y_dot
    using central differencing takes the form:

    dL/dtdy_dot = (L(t+h, y, y_dot+h) + L(t-h, y, y_dot-h) -
                   L(t-h, y, y_dot+h) - L(t+h, y, y_dot-h)) / (4*h^2)

    Parameters:
    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    t - float
        time at which the function is evaluated

    q - numpy array (2,)
        array containing the functions y & y_dot
        y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)
        y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    dL- float
            the value of the partial differentiation of L wrt t and y_dot,
            using central differencing
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in dL_dtdydot. It is: {}.".format(type(L))
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in dL_dtdydot. Current type is: {}.".format(type(t))
    assert type(q) == numpy.ndarray or type(q) == list, \
        "q is not the supported type in dL_dtdydot. Current type is: {}.".format(type(q))
    assert type(q[0]) == float or type(q[0]) == numpy.float64 or type(q[0]) == numpy.float, \
        "y is not the supported type in dL_dtdydot. Current type is: {}.".format(type(q[0]))
    assert type(q[1]) == float or type(q[1]) == numpy.float64 or type(q[1]) == numpy.float, \
        "y_dot is not the supported type in dL_dtdydot. Current type is: {}.".format(type(q[1]))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in dL_dtdydot. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in dL_dtdydot. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in dL_dtdydot. Current type is: {}.".format(type(beta))

    # calculate each component of the central differencing
    k1 = L(t+h, q[0], q[1]+h, alpha, beta)
    k2 = L(t+h, q[0], q[1]-h, alpha, beta)
    k3 = L(t-h, q[0], q[1]+h, alpha, beta)
    k4 = L(t-h, q[0], q[1]-h, alpha, beta)

    dL = (k1 - k2 - k3 + k4) / (4 * h**2)

    return dL


def d2L_dydydot(L, t, q, h, alpha, beta):
    """
    This function returns the value of the partial derivative of L wrt y &
    y_dot, by using the central differencing method.

    As L is a function of (t, y, y_dot), the partial derivate wrt y & y_dot
    using central differencing takes the form:

    dL/dydy_dot = (L(t, y+h, y_dot+h) + L(t, y-h, y_dot-h) -
                   L(t, y-h, y_dot+h) - L(t, y+h, y_dot-h)) / (4*h^2)

    However, as this function is used to calculate the y double dot from
    the lagrangian equation in the exercise, from the Euler-Langrange Equation
    the result of this partial derivative is multiplied by y_dot, to
    correctly display the formula.

    Parameters:
    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    t - float
        time at which the function is evaluated

    q - numpy array (2,)
        array containing the functions y & y_dot
        y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)
        y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    dL- float
            the value of the partial differentiation of L wrt y and y_dot,
            using central differencing
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in dL_dydydot. It is: {}.".format(type(L))
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in dL_dydydot. Current type is: {}.".format(type(t))
    assert type(q) == numpy.ndarray or type(q) == list, \
        "q is not the supported type in dL_dydydot. Current type is: {}.".format(type(q))
    assert type(q[0]) == float or type(q[0]) == numpy.float64 or type(q[0]) == numpy.float, \
        "y is not the supported type in dL_dydydot. Current type is: {}.".format(type(q[0]))
    assert type(q[1]) == float or type(q[1]) == numpy.float64 or type(q[1]) == numpy.float, \
        "y_dot is not the supported type in dL_dydydot. Current type is: {}.".format(type(q[1]))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in dL_dydydot. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in dL_dydydot. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in dL_dydydot. Current type is: {}.".format(type(beta))

    k1 = L(t, q[0]+h, q[1]+h, alpha, beta)
    k2 = L(t, q[0]-h, q[1]+h, alpha, beta)
    k3 = L(t, q[0]-h, q[1]-h, alpha, beta)
    k4 = L(t, q[0]+h, q[1]-h, alpha, beta)

    dL = q[1] * (k1 - k2 + k3 - k4) / (4*h**2)
    assert type(dL) == float or type(dL) == numpy.float64 or type(dL) == numpy.float, \
        "dL is not the supported type in dL_dydydot. Current type is: {}.".format(type(dL))

    return dL


def d2L_dydot2(L, t, q, h, alpha, beta):
    """
    This function returns the value of the second derivative of L wrt
    y_dot, as a double partial derivative of L wrt to y_dt,
    by using the central differencing method.

    As L is a function of (t, y, y_dot), the partial derivate wrt y_dot & y_dot
    using central differencing takes the form:

    dL/dydy_dot = (L(t, y, y_dot+h) + L(t, y, y_dot-h) -
                   2 * L(t, y, y_dot)) / (h^2)


    Parameters:
    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    t - float
        time at which the function is evaluated

    q - numpy array (2,)
        array containing the functions y & y_dot
        y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)
        y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    dL- float
            the value of the partial differentiation of L wrt y_dot and y_dot,
            using central differencing
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in d2L_dydot2. It is: {}.".format(type(L))
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in d2L_dydot2. Current type is: {}.".format(type(t))
    assert type(q) == numpy.ndarray or type(q) == list, \
        "q is not the supported type in d2L_dydot2. Current type is: {}.".format(type(q))
    assert type(q[0]) == float or type(q[0]) == numpy.float64 or type(q[0]) == numpy.float, \
        "y is not the supported type in d2L_dydot2. Current type is: {}.".format(type(q[0]))
    assert type(q[1]) == float or type(q[1]) == numpy.float64 or type(q[1]) == numpy.float, \
        "y_dot is not the supported type in d2L_dydot2. Current type is: {}.".format(type(q[1]))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in d2L_dydot2. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in d2L_dydot2. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in d2L_dydot2. Current type is: {}.".format(type(beta))

    k1 = L(t, q[0], q[1]+h, alpha, beta)
    k2 = L(t, q[0], q[1], alpha, beta)
    k3 = L(t, q[0], q[1]-h, alpha, beta)

    dL = (k1 - 2 * k2 + k3) / h**2
    assert type(dL) == float or type(dL) == numpy.float64 or type(dL) == numpy.float, \
        "dL is not the supported type in d2L_dydot2. Current type is: {}.".format(type(dL))

    return dL


def f(L, t, q, h, alpha, beta):
    """
    Based on the chain rule expansion of the Euler-Lagrange equation given in
    the exercise, y double dot can be rearranged to obtain:

    y_ddot = (dL/dy - y_dot * d2L/dydydot - dL/dtdydot)/d2L_dydot2

    Parameters:
    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    t - float
        time at which the function is evaluated

    q - numpy array (2,)
        array containing the functions y & y_dot
        y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)
        y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    y_ddot - float
            the value of the second derivative of y wrt t, based on the
            rearrangement of the expansion of the Euler-Lagrande equation
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in f. It is: {}.".format(type(L))
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in f. Current type is: {}.".format(type(t))
    assert type(q) == numpy.ndarray or type(q) == list, \
        "q is not the supported type in f. Current type is: {}.".format(type(q))
    assert type(q[0]) == float or type(q[0]) == numpy.float64 or type(q[0]) == numpy.float, \
        "y is not the supported type in f. Current type is: {}.".format(type(q[0]))
    assert type(q[1]) == float or type(q[1]) == numpy.float64 or type(q[1]) == numpy.float, \
        "y_dot is not the supported type in f. Current type is: {}.".format(type(q[1]))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in f. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in f. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in f. Current type is: {}.".format(type(beta))

    y_ddot = (dL_dy(L, t, q, h, alpha, beta) -
              d2L_dtdydot(L, t, q, h, alpha, beta) - \
              d2L_dydydot(L, t, q, h, alpha, beta)) / \
              d2L_dydot2(L, t, q, h, alpha, beta)
    assert type(y_ddot) == float or type(y_ddot) == numpy.float64 or type(y_ddot) == numpy.float, \
        "y_ddot is not the supported type in f. Current type is: {}.".format(type(y_ddot))

    return y_ddot


def dq_dt(q, t, L, h, alpha, beta):
    """
    A function used when calculating both the root and when using shooting
    method. Based on the array q, which contains y & several derivatives of
    y wrt time, this function creates another array dqdt which containes the
    derivaties wrt time of each element in q.

    Parameters:
    t - float
        time at which the function is evaluated

    q - numpy array (2,)
        array containing the functions y & y_dot
        y - float
        the value of the function that describes the output of the machinery.
        This output is trying to be decreased, such that y(0)=1 & y(1)=0.9,
        while maintaing the highest profit possible (the highest value of y)
        y_dot - float
            the first derivate wrt time of the function y. y describes
            the output of the machinery, such that y(0)=1 & y(1)=0.9.

    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    Return
    dqdt - numpy array (2, )
            an array similar to q, which contains the derivative wrt time of
            each element of q. As q contains (y, y_dot), dqdt will contain
            (y_dot, y_ddot).
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in dq_dt. It is: {}.".format(type(L))
    assert type(t) == float or type(t) == numpy.float64 or type(t) == numpy.float, \
        "t is not the supported type in dq_dt. Current type is: {}.".format(type(t))
    assert type(q) == numpy.ndarray or type(q) == list, \
        "q is not the supported type in dq_dt. Current type is: {}.".format(type(q))
    assert type(q[0]) == float or type(q[0]) == numpy.float64 or type(q[0]) == numpy.float, \
        "y is not the supported type in dq_dt. Current type is: {}.".format(type(q[0]))
    assert type(q[1]) == float or type(q[1]) == numpy.float64 or type(q[1]) == numpy.float, \
        "y_dot is not the supported type in dq_dt. Current type is: {}.".format(type(q[1]))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in dq_dt. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in dq_dt. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in dq_dt. Current type is: {}.".format(type(beta))

    dqdt = numpy.zeros_like(q)
    dqdt[0] = q[1] # y dot
    dqdt[1] = f(L, t, q, h, alpha, beta) # y ddot
    assert dqdt.shape == q.shape, \
        "dqdt does not have the same shape as q in dq_dt. There is a big problem."
    assert (dqdt != 0).any() , \
        "dqdt was not made properly in dq_dt. All value are equal to 0."
    return dqdt


def shooting_ivp(z, L, h, alpha, beta, time):
    """
    This function represents the algorithm for for the Initial Value Problem
    (IVP) used in the shooting method.

    With the initial quess (and further approximated values of this until the
    proper root is found) z and with the boundary condition at A (the start),
    we integrate the function dq_dt which contains the values of
    (y_dot, y_doubledot), to obtain the values of (y, y_dot). over a period
    of time. After this integration is finished, we take the last value
    obtained from the integration, for point B, and substract the boundary
    value(know), and return this difference as the root of the problem.
    This algorithm is repeated until tolerance in achieved.

    Parameters
    z - float
        initial guess for y'(A) used in shooting method

    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    time - numpy array (N, )
        array containing the period of time over which shooting should be
        investigated, from A to B, distributed into an N length array

    Return
    phi - float
        as the shooting method is based on an initial guess, finding
        the apropriate z requires foot finding. As such, in order to
        obtain the proper z, this function returns the value obtained
        during root finding minus the exact value required.
    """
    assert type(z) == float or type(z) == int or type(z) == numpy.float64 or type(z) == numpy.float, \
        "z is not the supported type in shooting_ivp. Current type is: {}.".format(type(z))
    assert isinstance(L, types.FunctionType), \
        "L is not a function in shooting_ivp. It is: {}.".format(type(L))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in shooting_ivp. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in shooting_ivp. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in shooting_ivp. Current type is: {}.".format(type(beta))
    assert type(time) == numpy.ndarray or type(time) == list, \
        "time is not the supported type in shooting_ivp. Current type is: {}.".format(type(time))

    # create the initial conditions
    q_init = numpy.array([y_A, z])
    # integrate
    q = odeint(dq_dt, q_init, time, args=(L, h, alpha, beta))

    # get the last value
    y_boundary = q[-1, 0]
    phi = y_boundary - y_B
    assert type(phi) == float or type(phi) == numpy.float64 or type(phi) == numpy.float, \
        "phi is not the supported type in shooting_ivp. Current type is: {}.".format(type(phi))
    return phi


def shooting(L, alpha, beta, h, dt):
    """
    This function implements the algorithm for the shooting method of solving
    Boundary Value Problems (BVP).
    It starts by considering that at initial time y(a) = A, and y'(a) = z,
    with z being an initial guess. With this intial guess, the system is
    evolved over the time period required. When the last value is obtained,
    it should be y(b) = B. However, initially, this is not the case.
    As such, a function phi(t, z) is considered, such that:
    phi(t, z) = y(t, z) - B. In order to obtain the actual value of the initial
    guess z, for which y(b)=B, we must find the root of phi at position b, to
    a given tolerance.

    Paramters
    L - function
        the lagrangian function that describes the profit loss required to be
        minimesed, based on the machine output y

    h -float
        the step required in central differencing.

    alpha - float
            penalty function factor

    beta - float
            penalty function factor

    dt - float
        the time step required for integration

    Return
    time - numpy array (N, )
            the time domain distributed between 0 & 1 in equal spaces, defined
            by dt
    q[:, 0] - numpy array(N, )
            array containg the value of the function y(t), for the interval
            (0, 1)
    """
    assert isinstance(L, types.FunctionType), \
        "L is not a function in shooting. It is: {}.".format(type(L))
    assert type(h) == float or type(h) == int or type(h) == numpy.float64 or type(h) == numpy.float, \
        "h is not the supported type in shooting. Current type is: {}.".format(type(h))
    assert type(alpha) == float or type(alpha) == int or type(alpha) == numpy.float64 or type(alpha) == numpy.float, \
        "alpha is not the supported type in shooting. Current type is: {}.".format(type(alpha))
    assert type(beta) == float or type(beta) == int or type(beta) == numpy.float64 or type(beta) == numpy.float, \
        "beta is not the supported type in shooting. Current type is: {}.".format(type(beta))
    assert type(dt) == float or type(dt) == int or type(dt) == numpy.float64 or type(dt) == numpy.float, \
        "dt is not the supported type in shooting. Current type is: {}.".format(type(dt))

    # initial guess, chose as y_A - y_B, as the slope of the line between A & B
    z_guess = y_A - y_B
    time = numpy.linspace(0, 1, int(1/dt)+1)
    # get the proper z values for our function, which will give  0 to shooting_ivp
    z_proper = newton(shooting_ivp, z_guess, args=(L, h, alpha, beta, time), tol=1e-12, maxiter=200)
    # integrate and calculate the value of y(t)
    q = odeint(dq_dt, [y_A, z_proper], time, args=(L, h, alpha, beta))
    assert q.shape == (time.shape[0], 2), \
        "q does not have the porper shape in shooting. It has {}.".format(q.shape)

    return time, q[:, 0]


def get_convergence(L, alpha, beta, h_init, dt, N, order=2):
    errors = numpy.zeros(N)
    convergence = numpy.zeros([N, int(1/dt)+1])
    h = numpy.zeros(N)
    for i in range(N):
        h[i] = h_init / order**i
        time, convergence[i] = shooting(L, alpha, beta, h[i], dt)
        errors[i] = numpy.linalg.norm((convergence[i] - convergence[i-1]), 2)

    return h[1:], errors[1:]


def plot_graph(L, alpha, beta, h, dt):
    t, q = shooting(L, alpha, beta, h, dt)

    pyplot.figure(figsize=(12,6))
    pyplot.plot(t, q, label="y(t)")
    pyplot.xlabel("t")
    pyplot.ylabel("y(t)")
    pyplot.title("The efficiency of the plant required for alpha={} & beta={}. h={} and dt={}".format(alpha, beta, h, dt))
    pyplot.legend()
    pyplot.show()


def plot_convergence(L, alpha, beta, h_init, dt, N):
    h, errors = get_convergence(L, alpha, beta, h_init, dt, N)

    grad_conv, e_pow_conv = numpy.polyfit(numpy.log(h), numpy.log(errors), 1)
    convergence_y = numpy.exp(e_pow_conv) * h**grad_conv
    print(grad_conv)
    pyplot.figure(figsize=(12,6))
    pyplot.loglog(h, errors)
    pyplot.loglog(h, errors, "bx")
    pyplot.xlabel(r'$\Delta$h')
    pyplot.ylabel("Error")
    pyplot.grid(True, which="Both")
    pyplot.title("Convergence of the method. For initial h={0:.2f} & dt={1:.2f}, convergence factor={2:.3f}.".format(h_init, dt, grad_conv))
    pyplot.show()


if __name__=="__main__":
    h = 0.05
    dt = 0.01
    plot_graph(L, 5, 5, h, dt)
    plot_graph(L, 7/4, 5, h, dt)
    plot_convergence(L, 7/4, 5, h, dt, 8)