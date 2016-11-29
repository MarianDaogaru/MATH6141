"""

Main script to be used for the first Coursework in MATH6141 - Numerical
Methods.

The software will implement a Runge-Kutta 3 and Gauss-Radau Runge-Kutta 3
method on the Modified Prothero-Robinson Problem described in:
Constantinescu, E.M. & Sandu, A. J Sci Comput (2013) 56: 28.
doi:10.1007/s10915-012-9662-z

To RUN this code and obtain the required plots / graphs, just run:

     !!!   CW_implementation() !!!


Author: Marian Daogaru
Contact: md2g12@soton.ac.uk
Created: 12/11/2016
Last Update: 21/11/2016
"""


import numpy as np
from scipy import polyfit
import scipy.optimize as sp_opt
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rcParams
import pytest

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.figsize'] = (16, 8)


# numpy method which ignores warnings, as they appear sometimes
np.seterr(all='ignore')


# -----------Question 1 algorithm implementation-------------------------------
# create the f function, the RHS in equation 2
def f(t, q, Gamma, epsilon, omega):
    """
    Function which deals with the Initial Value Problem (IVP) from:
    dq/dt = f(t, q). This function creates the RHS part of Equation 2
    described in the Coursework description, but also in the paper mentioned
    above, in the description of the script. In returns the result of the RHS.

    The equation constructed is:

    dq    d (x(t))
    --- = --(    ) = f(t, q) =                      (Equation 1)
    dt    dt(y(t))

      (Gamma       epsilon)  (    (-1 + x^2 - cos(t)) / (2x)    )       (        sin(t) / (2x)        )
    = (                   )  (                                  )   -   (                             )   (Equation 2)
      (epsilon        -1  ) .((-2 + y^2 - cost(omega * t) / (2y))       (omega * sin(omega * t) / (2y))


    Starting with initial conditions q(0) = (sqrt(2), sqrt(3)), the exact
    solution evolves to:

    q(t) = (sqrt(1 + cos(t)), sqrt(2 + cos(omega * t)))


    @Parameters:
    t - int / float
        time tn at which the integration of taking place, giving the t_(n+1)
        step

    q - np array (2)
        array containing the data for qn, the values of the function after
        differentiation with or without the addition parameters
        used for derivation. q contains x & y at time tn

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.


    @Returns
    f_val - np array (2)
            np array containing the computed x & y, for given parameters,
            according to the RHS of Equation 2.

    """
    # check if the inputs are properly given.
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(q) == np.ndarray, \
        "q must be a numpy array"
    assert q.shape == (2, ), \
        "q must be of shape (x, y) or (2,)"
    assert type(q[0]) == np.float64 and type(q[1]) == np.float64, \
        "X and Y in q must be float64 type"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # get x & y from q
    x, y = q

    # initialise first matrix in equation
    mtx1 = np.array([[Gamma, epsilon], [epsilon, -1]])

    # initialise second matrix in equation
    mtx2 = np.array([[(-1 + x**2 - np.cos(t)) / (2 * x)],
                     [(-2 + y**2 - np.cos(omega * t)) / (2 * y)]])
    # initialise third matrix in equation
    mtx3 = np.array([[np.sin(t) / (2 * x)],
                     [omega * np.sin(omega * t) / (2 * y)]])

    # compute the result of the RHS
    f_val = (np.dot(mtx1, mtx2) - mtx3).flatten()
    # use flatten so it gets [x,y] rather than [[x], [y]]

    # test f_val
    assert f_val.shape == (2,), \
        "the right hand side was not computed or returned properly in f."
    assert type(f_val[0]) == np.float64 and type(f_val[1]) == np.float64, \
        "X and Y in f_val must be float64 type. Not created properly."

    # return the computed RHS
    return f_val


# create the Runge-Kutta method implementation.
def MyRK3_step(f, t, qn, dt, Gamma, omega, epsilon):
    """
    Function that creates the explicit third order Runge-Kutta method,
    known as RK3, used as time-step discret method for approximating the
    solutions of ordinary differential equations.
    However, it is worth mentioning that this function only applies the
    RK3 for just 1 time-step, not for the entire time period.

    @Parameters

    f - function
        function that defines the ODE explained in the problem definition

    t - int / float
        time at which the integration of taking place, giving the tn step

    qn - np array (2)
        array containing the data from qn, the values of the function before
        differentiation

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns

    qn_new - np array (2)
        np array containing the values of the Equation after differention
        at time tn+1


    Description of algorithm:
    Given initial data qn, with q(0) = (sqrt(2), sqrt(3)), for the ODE problem
    described by Equation 1 (line 47) at location tn, using evenly spaced grid
    with spacing dt (such that t_(n+1) = t_n + dt), the algorithm is:

    k1 = f(tn, qn)
    k2 = f(tn + dt/2, qn + dt/2 * k1)
    k3 = f(tn + dt, qn + dt(-k1 + 2*k2))
    q_(n+1) = qn + dt/6 * (k1 + 4*k2 + k3)
    """
    # check if the inputs are properly given.
    assert callable(f), \
        "f must be a function, the function f defined previously typically."
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(qn) == np.ndarray, \
        "q must be a numpy array"
    assert qn.shape == (2, ), \
        "q must be of shape (x, y) or (2,)"
    assert type(qn[0]) == np.float64 and type(qn[1]) == np.float64, \
        "X and Y in q must be float64 type"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # start implementing the algorithm
    k1 = f(t, qn, Gamma, epsilon, omega)
    k2 = f(t + dt / 2, qn + dt * k1 / 2, Gamma, epsilon, omega)
    k3 = f(t + dt, qn + dt * (-k1 + 2*k2), Gamma, epsilon, omega)

    # compute the value at the new step.
    qn_new = qn + dt * (k1 + 4 * k2 + k3) / 6

    # test qn_new
    assert qn_new.shape == (2,), \
        "the right hand side was not computed or returned properly in f."
    assert type(qn_new[0]) == np.float64 and type(qn_new[1]) == np.float64, \
        "X and Y in f_val must be float64 type. Not created properly."

    return qn_new


# -----------Question 2 algorithm implementation-------------------------------
def F(K, f, t, qn, dt, Gamma, omega, epsilon):
    """
    Secondary function used to solve the implicit GRRK3 algorithm. it is
    explained in more detail in MyGRRK3_step function.

    @Parameters

    K - np array (2, )
        the matrix K containing k1 & k2 as [k1, k2]. k1 & k2 are also
        (2, ) np arrays. It is given this way by fsolve. As such, it must
        be reshaped before propper use.

    f - function
        function that defines the ODE explained in the problem definition

    t - int / float
        time at which the integration of taking place, giving the tn step

    qn - np array (2)
        array containing the data from qn, the values of the function before
        differentiation

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns
    K_new - np array (4, )
            because initial K containing k1 & k2, which are (2, ) np arrays
            K_new will containg 4 terms. In addition, it must be as an array,
            rather than matrix shaped so it can be used by fsolve.
            K_new contains the new values of k1 & k2, used by fsolve to
            converge.
    """
    # check if the inputs are properly given.
    assert K.shape == (4, ), \
        "K is not the proper shape"
    for i in range(4):
        assert type(K[i]) == np.float64, \
            "K[{}] is not a number".format(i)
    assert callable(f), \
        "f must be a function, the function f defined previously typically."
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(qn) == np.ndarray, \
        "q must be a numpy array"
    assert qn.shape == (2, ), \
        "q must be of shape (x, y) or (2,)"
    assert type(qn[0]) == np.float64 and type(qn[1]) == np.float64, \
        "X and Y in q must be float64 type"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # reshape K to make it useful & easier to implement
    K = K.reshape([2, 2])
    K_new = K - np.array([f(t + dt / 3, qn + dt / 12 * (5 * K[0] - K[1]), Gamma, epsilon, omega),
                         f(t + dt, qn+ dt / 4 * (3 * K[0] + K[1]), Gamma, epsilon, omega)])

    # flatten K_new
    K_new = K_new.flatten()
    assert K_new.shape == (4, ), \
        "K_new is not the proper shape. Was not created well."
    for i in range(4):
        assert type(K_new[i]) == np.float64, \
            "K_new[{}] is not a number".format(i)

    return K_new


def MyGRRK3_step(f, t, qn, dt, Gamma, omega, epsilon):
    """

    Function that creates the implicit third order accurate Gauss-Radau
    Runge-Kutta method, known as GRRK3, used as a time-step discret method
    for approximating the solution of ordinary differential equations (ODEs).

    This function only applies the GRRK3 algorithm for just 1 times-tep,
    not the entire period.

    @Parameters

    f - function
        function that defines the ODE explained in the problem definition

    t - int / float
        time at which the integration of taking place, giving the tn step

    qn - np array (2)
        array containing the data from qn, the values of the function before
        differentiation

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns

    qn_new - np array (2)
        np array containing the values of the Equation after differention
        at time tn+1


    Description of algorithm:
    Given initial data qn, with q(0) = (sqrt(2), sqrt(3)), for the ODE problem
    described by Equation 1 (line 47) at location tn, using evenly spaced grid
    with spacing dt (such that t_(n+1) = t_n + dt), the algorithm is:

    k1 = f(tn + dt / 3, qn + (dt / 12) * (5 * k1 - k2))
    k2 = f(tn + dt, qn + (dt / 4) * (3 * k1 + k2))
    q_n+1 = qn + (dt / 4) * (3 * k1 + k2)

    Note that both k1 and k2 depend on themselves, thus the implicit
    method.

    The funtion uses the scipy.optimize.fsolve to solve the implicity of the
    algorithm. A lower tolerance has been applied to fsolve, from Initially
    1.4902e-8 to 5e-7. This was implementd in order to get rid of warnings
    during stiff implementation of code, when the code was not making good
    process. As the stiff implementation works on sections where the difference
    between 2 points can be quite small, a smaller tolerance becomes
    inneficient, K is defines as:

        (k1)
    K = (  )
        (k2)

    Next, the problem / function to be solved is written as F(K) = 0, where:

            (f(tn + dt / 3, qn + (dt / 12) * (5 * k1 - k2)))
    F = K - (                                              )
            (   f(tn + dt, qn + (dt / 4) * (3 * k1 + k2))  )

    Next fsolve is applied to F, with initial conditions as:
    k1 = f(tn + dt / 3, qn)
    k2 = f(tn + dt, qn)

    After k1  & k2 are found by fsolve
    """
    # check if the inputs are properly given.
    assert callable(f), \
        "f must be a function, the function f defined previously typically."
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(qn) == np.ndarray, \
        "q must be a numpy array"
    assert qn.shape == (2, ), \
        "q must be of shape (x, y) or (2,)"
    assert type(qn[0]) == np.float64 and type(qn[1]) == np.float64, \
        "X and Y in q must be float64 type"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # initialise K, with initial conditions
    K_init = np.array([f(t + dt/3, qn, Gamma, epsilon, omega),
                       f(t + dt, qn, Gamma, epsilon, omega)])

    # start solving the implicit method
    Kn = sp_opt.fsolve(F, K_init, args=(f, t, qn, dt, Gamma, omega, epsilon),
                       xtol=5e-7, full_output=False)

    # reshape K such that it can be used easier in calculating qn_new
    Kn = Kn.reshape([2, 2])

    # calculate qn_new
    qn_new = qn + dt * (3 * Kn[0] + Kn[1]) / 4

    # test qn_new
    assert qn_new.shape == (2,), \
        "the right hand side was not computed or returned properly in f."
    assert type(qn_new[0]) == np.float64 and type(qn_new[1]) == np.float64, \
        "X and Y in f_val must be float64 type. Not created properly."

    return qn_new


# ------------Question 3 algorithm implementation------------------------------
def non_stiff_implementation(t=1, dt=0.05, Gamma=-2, omega=5, epsilon=0.05):
    """
    Function responsible for the non-stiff implementation of both the
    RK3 and GRRK3 methods. It is done by appliying MyGRRK3_step and MyRK3_step
    for each step tn (t_(n+1) = t_n + dt) from 0 to t.

    @Parameters

    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns

    qn_RK3 - np array (N, 2)
            np array containing the x & y values of the ODE over the time
            period t, computed using the RK3 method

    qn_GRRK3 - np array (N, 2)
            np array containing the x & y values of the ODE over the time
            period t computed using the GRRK3 method

    qn_exact - np array (N, 2)
            np array containg the x & y values of the ODe over the time
            period t computed using the exact solution given in the exercise.
    """
    # check if the inputs are properly given.
    assert callable(f), \
        "f must be a function, the function f defined previously typically."
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # get the number of time steps. +1 steps because t_1 is at 0.
    N = int(t/dt) + 1

    # create the empty qn_RK3 & populate initial conditions
    qn_RK3 = np.zeros([N, 2])
    qn_RK3[0] = [np.sqrt(2), np.sqrt(3)]

    # create the empty qn_GRRK3 & populate initial conditions
    qn_GRRK3 = np.zeros([N, 2])
    qn_GRRK3[0] = [np.sqrt(2), np.sqrt(3)]

    # create the exact solution for the problem
    times = np.linspace(0, t, N)
    qn_exact = np.array([np.sqrt(1 + np.cos(times)),  # X values
                         np.sqrt(2 + np.cos(omega * times))])  # Y values

    # create the qn for RK3 & GRRK3 for the timeperiod t defined
    for i in range(1, N):
        qn_RK3[i] = MyRK3_step(f, times[i-1], qn_RK3[i-1], dt, Gamma, omega, epsilon)
        qn_GRRK3[i] = MyGRRK3_step(f, times[i-1], qn_GRRK3[i-1], dt, Gamma, omega, epsilon)

    # test proper creating of qn_RK3, qn_GRRK3 & qn_exact
    assert qn_RK3.shape == (N, 2), \
        "qn_RK3 was not created properly, not the good shape"
    assert (qn_RK3[1:] != 0).any(), \
        "qn_RK3 contains only 0 values, not created properly."
    assert qn_GRRK3.shape == (N, 2), \
        "qn_GRRK3 was not created properly, not the good shape"
    assert (qn_GRRK3[1:] != 0).any(), \
        "qn_GRRK3 contains only 0 values, not created properly."
    assert qn_exact.shape == (2, N), \
        "qn_exact was not created properly, not the good shape"
    assert (qn_exact[1:] != 0).any(), \
        "qn_exact contains only 0 values, not created properly."

    return qn_RK3, qn_GRRK3, qn_exact


def question_3_plotting():
    """
    Function used to complete the requirements for Task 3 in the Coursework.

    The function calls the 'non_stiff_implementation' function in order
    to obtain the values produced by the RK3 & GRRK3 method when the
    defined system is not considered stiff.

    This function is autonomous, calling all necessary function & passing all
    the appropriate parameters. It then proceeds to use the generated solutions
    to create the x vs t & y vs t plots of the RK3, GRRk3 approximation & the
    exact solution. The plots are the displayed autonomously.

    @Parameters

    NONE - function takes no parameters


    @Returns
    fig - matplotlib figure
        Although the figure will be displayed automously, the function will
        return the figure so it can be used & investigated later on.
    """
    # get the values of the non stiff solutions
    qn_rk3, qn_grrk3, qn_exact = non_stiff_implementation()

    # get time spacing
    t = 1
    dt = 0.05
    times = np.linspace(0, t, int(t/dt) + 1)

    # initialise the figure for plots
    fig = plt.figure()
    # initialise first plot
    ax1 = fig.add_subplot(121)
    ax1.plot(times, qn_rk3[:, 0], 'b*', label="RK3")  # plot RK3 sol
    ax1.plot(times, qn_grrk3[:, 0], 'rx', label="GRRK3")  # plot GRRK3 sol
    ax1.plot(times, qn_exact[0], 'g', label="Exact")  # plot exact solution
    ax1.set_xlabel("Time (s)")  # set X axis title
    ax1.set_ylabel("X")  # set y axis label
    ax1.set_title("Q3: x(t) plot from RK3, GRRK3 & exact sol") # set overall titple
    ax1.legend()  # show overall legend

    # initialise second plot
    ax2 = fig.add_subplot(122)
    ax2.plot(times, qn_rk3[:, 1], 'b*', label="RK3")
    ax2.plot(times, qn_grrk3[:, 1], 'rx', label="GRRK3")
    ax2.plot(times, qn_exact[1], 'g', label="Exact")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y")
    ax2.set_title("Q3: y(t) plot from RK3, GRRK3 & exact sol")
    ax2.legend()

    print("Question 3 plots: The RK3 & GRRK3 plots.")
    fig.show  # show figure

    return fig  # return the figure


# ------------Question 4 algorithm implementation------------------------------
def error_step_nonstiff(i, t=1, dt=0.1):
    """
    Function that culculates the relative error between the RK3 & GRRk3
    methods & the exact solution for a given timestep (accuracy).

    @ Parameters

    i - int / float
        the power of 2 (2^i) which gives the number of steps in the time-period
        investigated. As i increaste, the timesped decrease, as the time-step
        is equal to 0.1/2^i.

    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    @ Returns

    err - np array (2)
            np. array contain the total error accumulated during one
            simulation, using the dimestep provided.
    """
    # calculate the time step
    assert type(i) == int or type(i) == float or type(i) == np.float64, \
        "the given i is not a valid / known / expected number format"
    N = 2**i * int(t / dt) + 1
    dt = dt / 2**i
    # get the values of approximations & exact solution, based on the time-step
    qn_RK3, qn_GRRK3, qn_exact = non_stiff_implementation(t=t, dt=dt)


    assert qn_RK3.shape == (N, 2), \
        "qn_RK3 was not created properly. Shape = {}, instead of {}".format(qn_RK3.shape, 2**i * 10 + 1)
    assert qn_GRRK3.shape == (N, 2), \
        "qn_GRRK3 was not created properly"
    assert qn_exact.shape == (2, N), \
        "qn_exact was not created properly"
    # create the np error array
    err = np.zeros(2)

    # calculate the total error for the given time step
    err[0] = dt * np.sum(np.abs(qn_RK3[:, 1] - qn_exact[1]))  # for RK3
    err[1] = dt * np.abs(qn_GRRK3[:, 1] - qn_exact[1]).sum()  # for GRRK3

    assert (err != 0).any(), \
        "One of the errors was not generated properly."
    return err


def question_4_plotting(N=8):
    """
    Function that creates executes the "non_stiff_implementation" several
    times, using different time-steps in order to obtain the absolute error
    for finer simulations.
    The error is obtained by comparing the values obtained using the RK3 &
    GRRK3 with the exact solution.Please see "error_step_nonstiff" function
    for exact detail of error calculation.

    Following the error calculation, log-log plots are created in order
    to desplay them easier. Also, convergence at 3rd order cam be observed
    for coarse time-steps.

    @ Parameters
    N - int / float
        THe number of simulations required to obtain the errors. An N number
        of non stiff solutions will be created, each with a timestep equal to
        0.1 / 2**i, i from 0 to N-1 inclusive.
        predefined to 8.

    @ Return
    fig2 - matplotlib figure
            the figure of the plos to show the relative absolute error of both
            thr RK3 and GRRK3 methods. Used to test the proper creation of
            the figure.
    """
    assert type(N) == int or type(N) == float or type(N) == np.float64, \
        "N is not the expecte / known types."

    errs = np.zeros([N, 2])

    # get the errors
    for i in range(N):
        errs[i] = error_step_nonstiff(i)

    times = 0.1 / 2 ** np.linspace(0, N-1, N)

    # start generating the loglog plots by creating a polyfit function
    n = N
    A_rk3 = np.zeros(n)  # array to store the arguments for RK3s polynomial
    A_grrk3 = np.zeros(n)  # same for GRRK3s polynomial
    # create the polyfit with the same power as the number of terms in the RK3
    A_rk3 = polyfit(times, errs[:, 0], n-1)  # for RK3
    A_grrk3 = polyfit(times, errs[:, 1], n-1)  # for GRRK3
    # storing the actual values of the polynomial equation
    B_rk3 = 0
    B_grrk3 = 0
    for i in range(n):
        B_rk3 += A_rk3[n - 1 - i] * times**i
        B_grrk3 += A_grrk3[n - 1 - i] * times**i

    # create the exponential function tto fit the polynomial for both methods
    grad_rk3, e_pow_rk3 = np.polyfit(np.log(times), np.log(errs[:, 0]), 1)
    convergence_y_rk3 = np.exp(e_pow_rk3) * times**grad_rk3

    grad_grrk3, e_pow_grrk3 = np.polyfit(np.log(times), np.log(errs[:, 1]), 1)
    convergence_y_grrk3 = np.exp(e_pow_grrk3) * times**grad_grrk3

    # plot the graphs, but loglog this time
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(121)
    ax1.loglog(times, errs[:, 0], 'kx', label="RK3 error")
    ax1.loglog(times, B_rk3, 'b', label="Polyfit")
    ax1.loglog(times, convergence_y_rk3, 'r', label="Converg @ {}".format(grad_rk3))
    ax1.set_xlabel("dt(s)")
    ax1.set_ylabel("Error")
    ax1.set_title("Q4: RK3 convergence from error 1-norm ")
    ax1.legend(loc=2)

    ax2 = fig2.add_subplot(122)
    ax2.loglog(times, errs[:, 1], 'kx', label="GRRK3")
    ax2.loglog(times, B_grrk3, 'b', label="Polyfit")
    ax2.loglog(times, convergence_y_grrk3, 'r', label="Converg @ {}".format(grad_grrk3))
    ax2.set_xlabel("dt(s)")
    ax2.set_ylabel("Error")
    ax2.set_title("Q4: GRRK3 convergence from error 1-norm ")
    ax2.legend(loc=2)

    print("Question 4 plots showing convergence.")
    fig2.show

    return fig2


# ------------Question 5 algorithm implementation------------------------------
def stiff_implementation_RK3(t=1, dt=0.001, Gamma=-2*10**5, omega=20, epsilon=0.5):
    """
    Function responsible for the stiff implementation of the
    RK3 method. It is done by appliying MyRK3_step
    for each step tn (t_(n+1) = t_n + dt) from 0 to t.

    @Parameters

    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns

    qn_RK3 - np array (N, 2)
            np array containing the x & y values of the ODE over the time
            period t, computed using the RK3 method

    qn_exact - np array (N, 2)
            np array containg the x & y values of the ODe over the time
            period t computed using the exact solution given in the exercise.
    """
    # check if the inputs are properly given.
    assert callable(f), \
        "f must be a function, the function f defined previously typically."
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # start creating the arrays for RK3 & exact solution, and populate them accordingly
    N = int(t/dt) + 1
    qn_RK3 = np.zeros([N, 2])
    qn_RK3[0] = [np.sqrt(2), np.sqrt(3)]

    times = np.linspace(0, t, N)
    qn_exact = np.array([np.sqrt(1 + np.cos(times)),
                         np.sqrt(2 + np.cos(omega * times))])

    for i in range(1, N):
        qn_RK3[i] = MyRK3_step(f, times[i-1], qn_RK3[i-1], dt, Gamma, omega, epsilon)

    # test proper creating of qn_RK3 & qn_exact
    assert qn_RK3.shape == (N, 2), \
        "qn_RK3 was not created properly, not the good shape"
    assert (qn_RK3[1:] != 0).any(), \
        "qn_RK3 contains only 0 values, not created properly."
    assert qn_exact.shape == (2, N), \
        "qn_exact was not created properly, not the good shape"
    assert (qn_exact[1:] != 0).any(), \
        "qn_exact contains only 0 values, not created properly."

    return qn_RK3, qn_exact


def question_5_plotting(t=1, dt=0.001, Gamma=-2*10**5, omega=20, epsilon=0.5):
    """
    Function that generates automatically the plots required in Exercise 5.
    The plots show how the RK3 becomes unstable when a so called "stiff"
    implementation is used.

    The graphs are expected to show how the solution tends to infinity during
    simulation.

    The function, once called will automatically call "stiff_implementation_RK3"
    to obtain the results of the approximation. Next, it used the values to
    create plots displaying the solutions.

    @Parameters
    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns
    fig3 - matplotlib figure
            figure returned to assert if a proper graph has been created.
            it can be later called to obtain the same graph.
    """
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."
    # get the stiff implementation RK3 results
    qn_rk3, qn_exact = stiff_implementation_RK3(t, dt, Gamma, omega, epsilon)

    t = 1
    dt = 0.001
    times = np.linspace(0, t, int(t/dt) + 1)

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(121)
    ax1.plot(times, qn_rk3[:, 0], 'b', label="RK3")
    ax1.plot(times, qn_exact[0], 'g', label="Exact")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("X")
    ax1.set_ylim(1, 1.5)
    ax1.set_title("Q5: x(t) stiff plot from RK3 & exact sol @ dt={}".format(dt))
    ax1.legend()

    ax2 = fig3.add_subplot(122)
    ax2.plot(times, qn_rk3[:, 1], 'b', label="RK3")
    ax2.plot(times, qn_exact[1], 'g', label="Exact")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y")
    ax2.set_ylim(1, 2)
    ax2.set_title("Q5: y(t) stiff plot from RK3 & exact sol @ dt={}".format(dt))
    ax2.legend()

    print("Question 5 plots: The RK3 for the stiff implementation @ dt={}".format(dt))
    fig3.show

    return fig3


# ------------Question 6 algorithm implementation------------------------------
def stiff_implementation_GRRK3(t=1, dt=0.005, Gamma=-2*10**5, omega=20, epsilon=0.5):
    """
    Function responsible for the stiff implementation of the
    GRRK3 method. It is done by appliying MyGRRK3_step
    for each step tn (t_(n+1) = t_n + dt) from 0 to t.

    @Parameters

    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @Returns

    qn_GRRK3 - np array (N, 2)
            np array containing the x & y values of the ODE over the time
            period t, computed using the GRRK3 method

    qn_exact - np array (N, 2)
            np array containg the x & y values of the ODe over the time
            period t computed using the exact solution given in the exercise.
    """
    # check if the inputs are properly given.
    assert callable(f), \
        "f must be a function, the function f defined previously typically."
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    N = int(t/dt) + 1
    qn_GRRK3 = np.zeros([N, 2])
    qn_GRRK3[0] = [np.sqrt(2), np.sqrt(3)]

    times = np.linspace(0, t, N)
    qn_exact = np.array([np.sqrt(1 + np.cos(times)),
                         np.sqrt(2 + np.cos(omega * times))])

    for i in range(1, N):
        qn_GRRK3[i] = MyGRRK3_step(f, times[i-1], qn_GRRK3[i-1], dt, Gamma, omega, epsilon)

    # test proper creating of qn_RK3, qn_GRRK3 & qn_exact
    assert qn_GRRK3.shape == (N, 2), \
        "qn_GRRK3 was not created properly, not the good shape"
    assert (qn_GRRK3[1:] != 0).any(), \
        "qn_GRRK3 contains only 0 values, not created properly."
    assert qn_exact.shape == (2, N), \
        "qn_exact was not created properly, not the good shape"
    assert (qn_exact[1:] != 0).any(), \
        "qn_exact contains only 0 values, not created properly."

    return qn_GRRK3, qn_exact


def question_6_plotting(t=1, dt=0.005, Gamma=-2*10**5, omega=20, epsilon=0.5):
    """
    Funtion that autonomously creates the plots required by Task 6.
    The function calls "stiff_implementation_GRRK3" to obtain the
    approximation values for a stiff implementation.

    Next, it creates 2 plots in order to illustrate the stability
    of GRRk3 method for approximating the derivative of the function.

    It can be seen that GRRk3 is mostly stable.

    @ parameters
    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @ Return
    fig 4 - matplotlib Figure
            matplotlib figure used to test the correct implementation of the
            code & to stoare the plots & display them later without calling
            the code again.
    """
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    qn_grrk3, qn_exact = stiff_implementation_GRRK3(t, dt, Gamma, omega, epsilon)

    t = 1
    dt = 0.005
    times = np.linspace(0, t, int(t/dt) + 1)

    fig4 = plt.figure()
    ax1 = fig4.add_subplot(121)
    ax1.plot(times, qn_grrk3[:, 0], 'b*', label="GRRK3")
    ax1.plot(times, qn_exact[0], 'g', label="Exact")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("X")
    ax1.set_title("Q6: x(t) stiff plot from RK3 & exact sol @ dt = {}".format(dt) )
    ax1.legend()

    ax2 = fig4.add_subplot(122)
    ax2.plot(times, qn_grrk3[:, 1], 'b*', label="GRRK3")
    ax2.plot(times, qn_exact[1], 'g', label="Exact")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Y")
    ax2.set_title("Q6: y(t) stiff plot from GRRK3 & exact sol @ dt={}".format(dt))
    ax2.legend()

    print("Question 6 plots: The GRRK3 for the stiff implementation @ dt={}".format(dt))
    fig4.show

    return fig4


# ------------Question 7 algorithm implementation------------------------------
def error_step_stiff(i, t=1, dt=0.05, Gamma=-2*10**5, omega=20, epsilon=0.5):
    """
    Function that culculates the relative error between the GRRk3
    methods & the exact solution for a given timestep (accuracy) when
    the stiff method is applied.

    @ Parameters

    i - int / float
        the power of 2 (2^i) which gives the number of steps in the time-period
        investigated. As i increaste, the timesped decrease, as the time-step
        is equal to 0.05/2^i.

    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @ Returns

    err - float / np.float64
            float contain the total error accumulated during one
            simulation, using the dimestep provided.
    """
    # calculate the time step
    assert type(i) == int or type(i) == float or type(i) == np.float64, \
        "the given i is not a valid / known / expected number format"
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # get the values of approximations & exact solution, based on the time-step
    N = 2**i * int(t / dt) + 1
    dt = dt / 2**i
    qn_GRRK3, qn_exact = stiff_implementation_GRRK3(t=t, dt=dt, Gamma=Gamma, omega=omega, epsilon=epsilon)
    assert qn_GRRK3.shape == (N, 2), \
        "qn_GRRK3 was not created properly"
    assert qn_exact.shape == (2, N), \
        "qn_exact was not created properly"

    err = dt * np.abs(qn_GRRK3[:, 1] - qn_exact[1]).sum()  # for GRRK3
    assert type(err) == float or type(err) == np.float64, \
        "err was not created properly, not a float in stiff-GRRK3"

    return err


def question_7_plotting(N=8, t=1, dt=0.05, Gamma=-2*10**5, omega=20, epsilon=0.5):
    """
    Funtion that autonomously creates the plots required by Task 6.
    The function calls "stiff_implementation_GRRK3" to obtain the
    approximation values for a stiff implementation.

    Next, it creates 2 plots in order to illustrate the stability
    of GRRk3 method for approximating the derivative of the function.

    It can be seen that GRRk3 is mostly stable.

    @ parameters

    N - float / int
        number of iterations for which the error function will run.
        Also defines the time-step size, as the step size is a function
        of 2^i, with i from 0 to N-1

    t - int / float
        time domain for which the derivated should be approximated

    dt - float
        time-step for derivation

    Gamma - float
            the capital Gamma term in the ODE.
            Used to stiffen or relax the derivation.

    omega - float
            the omega term in the ODE.
            Used to stiffen or relax the derivation.

    epsilon - float
                the epsilon term in the ODE.
                Used to stiffen or relax the derivation.

    @ Return
    fig 5 - matplotlib Figure
            matplotlib figure used to test the correct implementation of the
            code & to stoare the plots & display them later without calling
            the code again.
    """
    assert type(N) == int or type(N) == float or type(N) == np.float64, \
        "the given i is not a valid / known / expected number format"
    assert type(t) == int or type(t) == float or type(t) == np.float64, \
        "t (time) must be either an integer or float. Time is real!"
    assert type(dt) == int or type(dt) == float or type(dt) == np.float64, \
        "dt (timestep) must be either an integer or float. Time is real!"
    assert type(Gamma) == int or type(Gamma) == float, \
        "Gamma must be a number."
    assert type(omega) == int or type(omega) == float, \
        "omega must be a number."
    assert type(epsilon) == int or type(epsilon) == float, \
        "epsilon must be a number."

    # create the empty array for storring the errors
    errs = np.zeros(N)
    for i in range(N):
        errs[i] = error_step_stiff(i, t, dt, Gamma, omega, epsilon)

    times = dt / 2 ** np.linspace(0, N-1, N)

    # create the polyfit polynomial equation for GRRk3
    n = N
    A_grrk3 = np.zeros(n)
    A_grrk3 = polyfit(times, errs[:], n-1)  #for GRRK3
    B_grrk3 = 0
    for i in range(n):
        B_grrk3 += A_grrk3[n - 1 - i] * times**i

    # create the polyfit for the exponential equation & line in loglog graph
    grad_grrk3, e_pow_grrk3 = np.polyfit(np.log(times), np.log(errs), 1)
    convergence_y_grrk3 = np.exp(e_pow_grrk3) * times**grad_grrk3

    #plot the graph
    fig5 = plt.figure()
    ax1 = fig5.add_subplot(111)
    ax1.loglog(times, errs[:], 'kx', label="GRRK3 error")
    ax1.loglog(times, B_grrk3, 'b', label="Polyfit")
    ax1.loglog(times, convergence_y_grrk3, 'r', label="Converg @ {}".format(grad_grrk3))
    ax1.set_xlabel("dt(s)")
    ax1.set_ylabel("Error")
    ax1.set_title("Q7: GRRK3 convergence from error 1-norm @ dt={}".format(dt))
    ax1.legend(loc=2)

    print("Question 7 plots showing convergence.")
    fig5.show

    return fig5


# ------------Coursework run script--------------------------------------------
def CW_implementation():
    """
    Function that calls all the relevant functions, such that the plots
    required for MATH6141 - Numerical Methods, Coursework 1 are generated.

    Five functions are called: question_3_plotting - question_7_plotting
    are called.

    @Parameters
    NONE
    function does not require any parameters.

    @Returns
    NONE
    However, 9 plots are generated in 5 figures.
    """

    print("""Implementation of Question 1 & 2 can be seen in the code, and in
the solution of the remaining problems.
\n
\n""")
    # plots for Question 3
    plot_3 = question_3_plotting()
    assert type(plot_3) == matplotlib.figure.Figure, \
        "question_3_plotting should return a figure"

    # plots for Question 4
    plot_4 = question_4_plotting()
    assert type(plot_4) == matplotlib.figure.Figure, \
        "question_4_plotting should return a figure"

    # plots for Question 5
    plot_5 = question_5_plotting()
    assert type(plot_5) == matplotlib.figure.Figure, \
        "question_5_plotting should return a figure"

    #   plots for Question 6
    plot_6 = question_6_plotting()
    assert type(plot_6) == matplotlib.figure.Figure, \
        "question_6_plotting should return a figure"

    # plots for Question 7
    plot_7 = question_7_plotting()
    assert type(plot_7) == matplotlib.figure.Figure, \
        "question_7_plotting should return a figure"

    return None



# code to run automatically the CW implementation upon script run if and only if this script is the main.
if __name__ == "__main__":
    # CW_implementation()
    pass

CW_implementation()
