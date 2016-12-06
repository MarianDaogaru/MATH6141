import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
from scipy.optimize import newton


y_A = 1.
y_B = 0.9



def P(t, dy, alpha, beta):
    return alpha * dy**2 + beta * (t**2 - 1) * dy**3


def L(t, y, dy, alpha, beta):
    print(y, dy, t)
    return P(t, dy, alpha, beta) - y


def dL_dy(L, t, q, h, alpha, beta):
    """central differencing on dt"""
    return (L(t, q[0]+h, q[1], alpha, beta) - L(t, q[0]-h, q[1], alpha, beta)) / (2*h)


def d2L_dtdydot(L, t, q, h, alpha, beta):
    return (L(t+h, q[0], q[1]+h, alpha, beta) - \
            L(t+h, q[0], q[1]-h, alpha, beta) - \
            L(t-h, q[0], q[1]+h, alpha, beta) + \
            L(t-h, q[0], q[1]-h, alpha, beta)) / (4 * h**2)

def d2L_dydydot(L, t, q, h, alpha, beta):
    print(type(q))
    k1 = L(t, q[0]+h, q[1]+h, alpha, beta)
    k2 = L(t, q[0]-h, q[1]+h, alpha, beta)
    k3 = L(t, q[0]-h, q[1]-h, alpha, beta)
    k4 = L(t, q[0]+h, q[1]-h, alpha, beta)
    return q[1] * (k1 - k2 + k3 - k4) / (4*h**2)


def d2L_dydot2(L, t, q, h, alpha, beta):
    print("d2l_dydot2", q)
    k1 = L(t, q[0], q[1]+h, alpha, beta)
    k2 = L(t, q[0], q[1], alpha, beta)
    k3 = L(t, q[0], q[1]-h, alpha, beta)
    return (k1 - 2 * k2 + k3) / h**2

def f(L, t, q, h, alpha, beta):
    """dqdx = numpy.zeros_like(q)
    dqdx[0] = q[1]
    dqdx[1] = -1 - q[1]
    return dqdx"""
    return (dL_dy(L, t, q, h, alpha, beta) - \
             d2L_dtdydot(L, t, q, h, alpha, beta) - \
             d2L_dydydot(L, t, q, h, alpha, beta)) / \
             d2L_dydot2(L, t, q, h, alpha, beta)

def dq_dt(q, t, h, alpha, beta):
    dqdt = numpy.zeros_like(q)
    dqdt[0] = q[1] # y dot
    dqdt[1] = f(L, t, q, h, alpha, beta) # y ddot
    return dqdt


def phi_z(t, q, z):
    return q[0] - y_B


def shooting_ivp(z, h, alpha, beta):
    q = odeint(dq_dt, [y_A, z], [0, 1], args=(h, alpha, beta)) # nu-i bine
    y_boundary = q[-1, 0]
    return y_boundary - y_B


def shooting():
    z_guess = 0.1
    h = 0.01
    alpha = 7/4
    beta = 5
    z_proper = newton(shooting_ivp, z_guess, args=(h, alpha, beta))
    print("Z ", z_proper)
    t = numpy.linspace(0, 1, 501)
    soln = odeint(dq_dt, [y_A, z_proper], t, args=(h, alpha, beta))
    return t, soln[:, 0]




if __name__=="__main__":

#    x, y = shooting()
#    pyplot.figure(figsize=(12,6))
#    pyplot.plot(x, y)
#    pyplot.plot(x, 2*numpy.exp(1)/(numpy.exp(1)-1)*(1-numpy.exp(-x))-x)
#    pyplot.xlabel(r"$x$")
#    pyplot.show()
#    pyplot.figure(figsize=(12,6))
#    pyplot.plot(x, y-(2*numpy.exp(1)/(numpy.exp(1)-1)*(1-numpy.exp(-x))-x))
#    pyplot.xlabel(r"$x$")
#    pyplot.show()

    #q=shooting_ivp(0.1, 0.001, 5, 5)
    t, a = shooting()