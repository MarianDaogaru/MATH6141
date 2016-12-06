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
        "dL is not the supported type in dL_dtdydot. Current type is: {}.".format(type(dL))

    return dL


def d2L_dydot2(L, t, q, h, alpha, beta):
    k1 = L(t, q[0], q[1]+h, alpha, beta)
    k2 = L(t, q[0], q[1], alpha, beta)
    k3 = L(t, q[0], q[1]-h, alpha, beta)
    return (k1 - 2 * k2 + k3) / h**2


def f(L, t, q, h, alpha, beta):
    return (dL_dy(L, t, q, h, alpha, beta) - \
             d2L_dtdydot(L, t, q, h, alpha, beta) - \
             d2L_dydydot(L, t, q, h, alpha, beta)) / \
             d2L_dydot2(L, t, q, h, alpha, beta)

def dq_dt(q, t, h, alpha, beta):
    #print(h)
    dqdt = numpy.zeros_like(q)
    dqdt[0] = q[1] # y dot
    dqdt[1] = f(L, t, q, h, alpha, beta) # y ddot
    return dqdt


def shooting_ivp(z, h, alpha, beta, time):
    q_init = numpy.array([y_A, z])
    q = odeint(dq_dt, q_init, time, args=(h, alpha, beta)) # nu-i bine
    y_boundary = q[-1, 0]
    return y_boundary - y_B


def shooting(alpha, beta, h, dt):
    z_guess = 0.1
    time = numpy.linspace(0, 1, int(1/dt)+1)
    z_proper = newton(shooting_ivp, z_guess, args=(h, alpha, beta, time), tol=1e-12, maxiter=200)
    q = odeint(dq_dt, [y_A, z_proper], time, args=(h, alpha, beta))
    return time, q[:, 0]


def get_convergence(alpha, beta, h_init, dt, N, order=2):
    errors = numpy.zeros(N)
    convergence = numpy.zeros([N, int(1/dt)+1])
    h = numpy.zeros(N)
    for i in range(N):
        h[i] = h_init / order**i
        time, convergence[i] = shooting(alpha, beta, h[i], dt)
        errors[i] = numpy.linalg.norm((convergence[i] - convergence[i-1]), 2)

    return h[1:], errors[1:]


def plot_graph(alpha, beta, h, dt):
    t, q = shooting(alpha, beta, h, dt)

    pyplot.figure(figsize=(12,6))
    pyplot.plot(t, q, label="y(t)")
    pyplot.xlabel("t")
    pyplot.ylabel("y(t)")
    pyplot.title("The efficiency of the plant required for alpha={} & beta={}. h={} and dt={}".format(alpha, beta, h, dt))
    pyplot.legend()
    pyplot.show()

def plot_convergence(alpha, beta, h_init, dt, N):
    h, errors = get_convergence(alpha, beta, h_init, dt, N)

    grad_conv, e_pow_conv = numpy.polyfit(numpy.log(h), numpy.log(errors), 1)
    convergence_y = numpy.exp(e_pow_conv) * h**grad_conv
    print(grad_conv)
    pyplot.figure(figsize=(12,6))
    pyplot.loglog(h, errors)
    pyplot.loglog(h, errors, "bx")
    pyplot.xlabel(r'$\Delta$h')
    pyplot.ylabel("Error")
    pyplot.grid(True, which="Both")
    pyplot.title("Convergence of the method. For initial h={} & dt={}, convergence factor={.2f}.".format(h_init, dt, grad_conv))
    pyplot.show()

if __name__=="__main__":
    h = 0.05
    dt = 0.01
    plot_graph(5, 5, h, dt)
    plot_graph(7/4, 5, h, dt)
    plot_convergence(7/4, 5, h, dt, 8)