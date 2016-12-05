import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
from scipy.optimize import newton

def P(t, dy, alpha, beta):
    return alpha * dy**2 + beta * (t**2 - 1) * dy**3


def L(t, y, dy, alpha, beta):
    return P(t, dy, alpha, beta) - y


def dLdy(L, t, q, h, alpha, beta):
    """central differencing on dt"""
    return (L(t, q[0]+h, q[1], alpha, beta) - L(t, q[0-h, q[1], alpha, beta)) / (2*h)


def d2L_dtdydot(L, t, q, h, alpha, beta):
    return (L(t+h, q[0], q[1]+h, alpha, beta) - \
            L(t+h, q[0], q[1]-h, alpha, beta) - \
            L(t-h, q[0], q[1]+h, alpha, beta) + \
            L(t-h, q[0], q[1]-h, alpha, beta)) / (4 * h**2)

def d2L_dydydot(L, t, q, h, alpha, beta):
    return q[1] * (L(t, q[0]+h, q[1]+h, alpha, beta) - \
                   L(t, q[0]+h. q[1]-h, alpha, beta) - \
                   L(t, q[0]-h, q[1]+h, alpha, beta) + \
                   L(t, q[0]-h, q[1]-h, alpha, beta)) / (4*h**2)


def d2L_dydot2(l, t, q, h, alpha, beta):
        return (L(t, q[0], q[1]+h, alpha, beta) - \
                2 * L(t, q[0], [1], alpha, beta) + \
                L(t, q[0], q[1]-h, alpha, beta)) / h**2



 def f(q, x):
    dqdx = numpy.zeros_like(q)
    dqdx[0] = q[1]
    dqdx[1] = -1 - q[1]
    return dqdx

def shooting_ivp(z):
    soln = odeint(f, [y0b, z], [0, 1])
    y_boundary = soln[-1, 0]
    return y_boundary - 0.9

def shooting():
    z_guess = 0.1
    z_proper = newton(shooting_ivp, z_guess)
    x = numpy.linspace(0, 1)
    soln = odeint(f, [0, z_proper], x)
    return x, soln[:, 0]




if __name__=="__main__":

    x, y = shooting()
    pyplot.figure(figsize=(12,6))
    pyplot.plot(x, y)
    pyplot.plot(x, 2*numpy.exp(1)/(numpy.exp(1)-1)*(1-numpy.exp(-x))-x)
    pyplot.xlabel(r"$x$")
    pyplot.show()
    pyplot.figure(figsize=(12,6))
    pyplot.plot(x, y-(2*numpy.exp(1)/(numpy.exp(1)-1)*(1-numpy.exp(-x))-x))
    pyplot.xlabel(r"$x$")
    pyplot.show()