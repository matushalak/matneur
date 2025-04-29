# @matushalak Mathematical neuroscience 2025
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Helper functions
# steady‑state gates
def m_inf(V, par):
    return 0.5 * (1 + np.tanh((V - par['V1']) / par['V2']))

def n_inf(V, par):
    return 0.5 * (1 + np.tanh((V - par['V3']) / par['V4']))

def I_ca(V, par):
    return par['gCa'] * m_inf(V, par) * (V - par['ECa'])

def I_kca (V, Ca, par):
    return par['gKCa'] * (Ca / (Ca + 1)) * (V - par['EK'])

# time constant
def tau_n(V, par):
    return 1 / np.cosh((V - par['V3']) / (2 * par['V4']))

def MLburst(t: float,
            y: list[float, float, float],
            I=lambda t: 45,
            params: dict | None = None,
            epsilon=1e-3, mu=0.02) -> np.ndarray:
    """
    Morris–Lecar–KCa square‑wave burster (Ermentrout & Terman, Eq 5.2).
    y = [V, n, Ca]
    """
    if params is None:     # default parameter set
        params = dict(gL=2,  EL=-60,
                      gK=8,  EK=-84,
                      gCa=4, ECa=120,
                      gKCa=0.25,
                      V1=-1.2, V2=18,
                      V3=12,  V4=17.4,
                      phi=0.23,
                      Cm=20, kCa=1.0,
                      epsilon = 1e-3, mu = 0.02)

    V, n, Ca = y

    # ionic currents
    I_Ca  = I_ca(V, params)
    I_K   = params['gK'] * n * (V - params['EK'])
    I_KCa = I_kca(V, Ca, params)
    I_L   = params['gL'] * (V - params['EL'])
    # fixed applied current
    I_app = I(t) if callable(I) else params.get('Iapp', 0.0) 

    # differential equations
    dVdt  = (I_app - I_L - I_K - I_Ca - I_KCa) / params['Cm']
    dndt  = params['phi'] * (n_inf(V, params) - n) / tau_n(V, params)
    dCadt = epsilon * (-mu * I_Ca - params['kCa'] * Ca)

    return np.array([dVdt, dndt, dCadt])

def MLfast(t: float,
           y: list[float, float],
           Ca: float,
           I=lambda t: 45,
           params: dict | None = None) -> np.ndarray:
    '''
    Fast subsystem where Ca parameter
    '''
    if params is None:     # default parameter set
        params = dict(gL=2,  EL=-60,
                      gK=8,  EK=-84,
                      gCa=4, ECa=120,
                      gKCa=0.25,
                      V1=-1.2, V2=18,
                      V3=12,  V4=17.4,
                      phi=0.23,
                      Cm=20, kCa=1.0)

    V, n = y

    # ionic currents
    I_Ca  = I_ca(V, params)
    I_K   = params['gK'] * n * (V - params['EK'])
    I_KCa = I_kca(V, Ca, params)
    I_L   = params['gL'] * (V - params['EL'])
    # fixed applied current
    I_app = I(t) if callable(I) else params.get('Iapp', 0.0) 

    # differential equations
    dVdt  = (I_app - I_L - I_K - I_Ca - I_KCa) / params['Cm']
    dndt  = params['phi'] * (n_inf(V, params) - n) / tau_n(V, params)

    return np.array([dVdt, dndt])

def H_of_Ca(Ca, t_trans=1500, t_end=8000, dt_max=0.1,
            epsilon=0.001, mu=0.02):
    """
    Returns   H(Ca) = (1/T) ∫_0^T  dCa/dt dt
    where dCa/dt = ε (-μ I_Ca(V) - kCa Ca)
    along the last cycle of the fast subsystem with [Ca] frozen.
    """
    # params
    p = dict(gL=2,  EL=-60,
            gK=8,  EK=-84,
            gCa=4, ECa=120,
            gKCa=0.25,
            V1=-1.2, V2=18,
            V3=12,  V4=17.4,
            phi=0.23,
            Cm=20, kCa=1.0)
    # ---- integrate fast subsystem long enough to reach the LC ----
    sol = solve_ivp(MLfast, (0, t_end), [-60, 0],
                    max_step=dt_max, args=(Ca,))
    V = sol.y[0]; t = sol.t
    # breakpoint()
    # ---- find spike peaks in V (prominence filters out noise) ----
    peaks, _ = find_peaks(V)
    # keep only peaks after transients
    peaks = peaks[t[peaks] > t_trans]
    # breakpoint()
    if len(peaks) < 2:
        raise RuntimeError(f'No full cycle detected for Ca={Ca:.3f}')
    # use the two last peaks → one full period
    k1, k2 = peaks[-3], peaks[-2]
    idx = slice(k1, k2+1)                  # slice of that cycle
    T  = t[k2] - t[k1]                     # period

    # ---- evaluate dCa/dt along that cycle ----
    # only 2nd term depends on Ca; with higher Ca, always more subtracted, 
    # so monotonically decreasing function
    dCadt = epsilon * (-mu*I_ca(V[idx], p) - p['kCa']*Ca)
    H = np.trapezoid(dCadt, t[idx]) / T        # trapezoidal rule

    return H

def q2_2 ():
    tspan = (0, 10000)
    y0 = [-60, 0.0, 0.01]

    sol = solve_ivp(MLburst, tspan, y0, max_step=0.1)
    
    ax = plt.axes(projection = '3d')
    ax.plot(sol.y[2], sol.y[1], sol.y[0])
    ax.set(ylabel = 'n', xlabel = 'Ca', zlabel = 'V')
    plt.show()

    fg, tsax = plt.subplots(nrows=3, sharex='col')
    tsax[0].plot(sol.t, sol.y[0])
    tsax[0].set_ylabel('V (mV)')
    tsax[1].plot(sol.t, sol.y[1])
    tsax[1].set_ylabel('n')
    tsax[2].plot(sol.t, sol.y[2])
    tsax[2].set_ylabel('[Ca]')
    tsax[2].set_xlabel('time (ms)')
    plt.tight_layout()
    plt.show()
    plt.close()

def q2_3():
    Ca_grid = np.linspace(0, 1, 10)
    H_vals  = []

    for Ca in Ca_grid:
        try:
            H_vals.append(H_of_Ca(Ca, epsilon=1e-3))
            print(Ca, 'LC found')
        except RuntimeError:
            H_vals.append(np.nan)
            print(Ca, 'LC not found')
    
    plt.plot(Ca_grid, H_vals, 'k-')
    plt.axhline(0, ls='--', lw=0.7)
    plt.xlabel('[Ca] (fixed parameter)')
    plt.ylabel('H(Ca)')
    plt.title('Average d[Ca]/dt along limit cycle')
    plt.show()


def q2_5():
    for eps in [1e-7, 1e-6, 1e-5, 3e-4, 1e-3, 1e-2, 1e-1, 0.5]:
        tspan = (0, 10000)
        y0 = [-60, 0.0, 0.01]
        eps_try = lambda t, vars: MLburst(t=t, y = vars, epsilon=eps)
        sol = solve_ivp(eps_try, tspan, y0, max_step=0.1)
        
        ax = plt.axes(projection = '3d')
        ax.plot(sol.y[2], sol.y[1], sol.y[0])
        ax.set(ylabel = 'n', xlabel = 'Ca', zlabel = 'V')
        plt.savefig(f'Eps_{eps}3D.png', dpi = 300)
        plt.show()

        fg, tsax = plt.subplots(nrows=3, sharex='col')
        tsax[0].plot(sol.t, sol.y[0])
        tsax[0].set_ylabel('V (mV)')
        tsax[1].plot(sol.t, sol.y[1])
        tsax[1].set_ylabel('n')
        tsax[2].plot(sol.t, sol.y[2])
        tsax[2].set_ylabel('[Ca]')
        tsax[2].set_xlabel('time (ms)')
        plt.tight_layout()
        plt.savefig(f'Eps_{eps}traces.png', dpi = 300)
        plt.show()
        plt.close()

def q2_6():
    for mu in [2e-2, 6e-2, 1e-1, 2e-1]:
        tspan = (0, 10000)
        y0 = [-60, 0.0, 0.01]
        mu_try = lambda t, vars: MLburst(t=t, y = vars, mu=mu)
        sol = solve_ivp(mu_try, tspan, y0, max_step=0.1)
        
        ax = plt.axes(projection = '3d')
        ax.plot(sol.y[2], sol.y[1], sol.y[0])
        ax.set(ylabel = 'n', xlabel = 'Ca', zlabel = 'V')
        plt.savefig(f'MU_{mu}3D.png', dpi = 300)
        plt.show()

        fg, tsax = plt.subplots(nrows=3, sharex='col')
        tsax[0].plot(sol.t, sol.y[0])
        tsax[0].set_ylabel('V (mV)')
        tsax[1].plot(sol.t, sol.y[1])
        tsax[1].set_ylabel('n')
        tsax[2].plot(sol.t, sol.y[2])
        tsax[2].set_ylabel('[Ca]')
        tsax[2].set_xlabel('time (ms)')
        plt.tight_layout()
        plt.savefig(f'MU_{mu}traces.png', dpi = 300)
        plt.show()
        plt.close()
if __name__ == '__main__':
    # q2_2()

    # q2_3()

    # Q2.5
    # epsilon around 3e-4 leads to spiking eventually; around 1e-3 - 1e-2 bursting, 1e-1 is spiking again
    # q2_5()

    # Q2.6
    q2_6()