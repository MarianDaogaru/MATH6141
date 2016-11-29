import pytest
from main_CW1 import *

def pytest_non_stiff_implementation():
    for i in range(10):
        t = np.random.randint(1, 10)
        dt = 0.005
        qn_rk, qn_grrk, qn_exact = non_stiff_implementation(t=t, dt=dt)
        assert (qn_rk.shape == (int(t/dt) + 1, 2))
        assert (qn_grrk.shape == (int(t/dt) + 1, 2))
        assert (qn_exact.shape == (2, int(t/dt) + 1))
        n = np.random.randint(int(t/dt)+1)
        assert (qn_exact[0, n] == np.sqrt(1 + np.cos(n * dt)))
        assert (qn_exact[1, n] == np.sqrt(2 + np.cos(n * dt * 5)))
        assert ((qn_rk[n] - qn_exact[:, n] < 1e-3).all())
        assert ((qn_grrk[n] - qn_exact[:, n] < 1e-3).all())

def pytest_stiff_implementation():
    for i in range(10):
        t = np.random.randint(1, 10)
        dt = 0.005
        qn_rk, qn_exact1 = stiff_implementation_RK3(t=t, dt=dt)
        qn_grrk, qn_exact2 = stiff_implementation_GRRK3(t=t, dt=dt)
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
        assert ((qn_grrk[n] - qn_exact1[:, n] < 1e-3).all())

if __name__ == "__main__":
    pytest.main()

