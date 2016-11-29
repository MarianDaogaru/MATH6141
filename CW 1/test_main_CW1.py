import pytest
import main_CW1 as CW1
import numpy as np


def test_non_stiff_implementation():
    for i in range(10):
        print("Test {} non stiff".format(i))
        t = i + 1
        dt = 0.005
        qn_rk, qn_grrk, qn_exact = CW1.non_stiff_implementation(t=t, dt=dt)
        assert (qn_rk.shape == (int(t/dt) + 1, 2))
        assert (qn_grrk.shape == (int(t/dt) + 1, 2))
        assert (qn_exact.shape == (2, int(t/dt) + 1))
        n = np.random.randint(int(t/dt)+1)
        assert (qn_exact[0, n] == np.sqrt(1 + np.cos(n * dt)))
        assert (qn_exact[1, n] == np.sqrt(2 + np.cos(n * dt * 5)))
        assert ((qn_rk[n] - qn_exact[:, n] < 1e-4).all())
        assert ((qn_grrk[n] - qn_exact[:, n] < 1e-4).all())


def test_stiff_implementation():
    for i in range(10):
        print("Test {} stiff".format(i))
        t = i + 1
        dt = 0.005
        qn_rk, qn_exact1 = CW1.stiff_implementation_RK3(t=t, dt=dt)
        qn_grrk, qn_exact2 = CW1.stiff_implementation_GRRK3(t=t, dt=dt)
        assert (qn_rk.shape == (int(t/dt) + 1, 2))
        assert (qn_grrk.shape == (int(t/dt) + 1, 2))
        assert (qn_exact1.shape == (2, int(t/dt) + 1))
        assert (qn_exact2.shape == (2, int(t/dt) + 1))
        n = np.random.randint(int(t/dt)+1)
        assert (qn_exact1[0, n] == np.sqrt(1 + np.cos(n * dt)))
        assert (qn_exact1[1, n] == np.sqrt(2 + np.cos(n * dt * 20)))
        assert (qn_exact2[0, n] == np.sqrt(1 + np.cos(n * dt)))
        assert (qn_exact2[1, n] == np.sqrt(2 + np.cos(n * dt * 20)))
        # assert ((qn_rk[n] - qn_exact[:, n] < 1e-3).all())
        assert ((qn_grrk[n] - qn_exact1[:, n] < 1e-4).all())


def test_error_non_stiff():
    for i in range(10):
        print("test {} error non stiff.".format(i))
        t = 2
        dt = 0.05
        j = np.random.randint(10)

        err = CW1.error_step_nonstiff(j, t=t, dt=dt)
        assert (err.shape == (2,))
        assert ((err < 1e-3).any())


def test_error_stiff():
    for i in range(3):
        print("test {} error stiff.".format(i))
        t = 2
        dt = 0.01
        j = np.random.randint(10)

        err = CW1.error_step_stiff(j, t=t, dt=dt)
        assert (err < 1e-3)


if __name__ == "__main__":
    pytest.main()
