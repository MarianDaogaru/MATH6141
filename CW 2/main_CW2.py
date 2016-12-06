import numpy
from matplotlib import pyplot
from scipy.integrate import odeint
from scipy.optimize import newton


y_A = 1.
y_B = 0.9



def P(t, dy, alpha, beta):
    return alpha * dy**2 + beta * (t**2 - 1) * dy**3


def L(t, y, dy, alpha, beta):
    #print(y, dy, t)
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
    #print(type(q))
    k1 = L(t, q[0]+h, q[1]+h, alpha, beta)
    k2 = L(t, q[0]-h, q[1]+h, alpha, beta)
    k3 = L(t, q[0]-h, q[1]-h, alpha, beta)
    k4 = L(t, q[0]+h, q[1]-h, alpha, beta)
    return q[1] * (k1 - k2 + k3 - k4) / (4*h**2)


def d2L_dydot2(L, t, q, h, alpha, beta):
    #print("d2l_dydot2", q)
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