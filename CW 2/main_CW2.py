import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
from scipy.optimize import newton


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
    """central differencing on dt"""
    return (L(t, q[0]+h, q[1], alpha, beta) - L(t, q[0]-h, q[1], alpha, beta)) / (2*h)


def d2L_dtdydot(L, t, q, h, alpha, beta):
    return (L(t+h, q[0], q[1]+h, alpha, beta) - \
            L(t+h, q[0], q[1]-h, alpha, beta) - \
            L(t-h, q[0], q[1]+h, alpha, beta) + \
            L(t-h, q[0], q[1]-h, alpha, beta)) / (4 * h**2)

def d2L_dydydot(L, t, q, h, alpha, beta):
    k1 = L(t, q[0]+h, q[1]+h, alpha, beta)
    k2 = L(t, q[0]-h, q[1]+h, alpha, beta)
    k3 = L(t, q[0]-h, q[1]-h, alpha, beta)
    k4 = L(t, q[0]+h, q[1]-h, alpha, beta)
    return q[1] * (k1 - k2 + k3 - k4) / (4*h**2)


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


def phi_z(t, q, z):
    return q[0] - y_B


def shooting_ivp(z, h, alpha, beta, time):
    q = odeint(dq_dt, [y_A, z], time, args=(h, alpha, beta)) # nu-i bine
    y_boundary = q[-1, 0]
    #print(y_boundary)
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
    pyplot.title("Convergence of the method. For initial h={} & dt={}, convergence factor={}.".format(h_init, dt, grad_conv))
    pyplot.show()

if __name__=="__main__":
    h = 0.05
    dt = 0.01
    plot_graph(5, 5, h, dt)
    plot_graph(7/4, 5, h, dt)
    plot_convergence(7/4, 5, h, dt, 8)