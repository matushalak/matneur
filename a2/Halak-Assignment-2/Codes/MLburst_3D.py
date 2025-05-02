#@matushalak Mathematical neuroscience 2025
# This is a separate script for the 3D plotting in MLburst
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
from scipy.integrate import solve_ivp


def Q2_5_6(eps:float = 1e-3, mu:float = 2e-2,
           iniVALS:list[float, float, float] = [-60, 0.0, 0.01]):
    # ------------------------------------------------------------
    # Parameter set (square‑wave burster)
    # ------------------------------------------------------------
    par = dict(
        gL=2.0,  EL=-60.0,
        gK=8.0,  EK=-84.0,
        gCa=4.0, ECa=120.0,
        gKCa=0.25,
        V1=-1.2, V2=18.0,
        V3=12.0, V4=17.4,
        phi=0.23,
        Cm=20.0,
        epsilon=eps, mu=mu, kCa=1.0,
        Iapp=45.0
    )

    # ------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------
    def m_inf(V):
        return 0.5 * (1 + np.tanh((V - par['V1']) / par['V2']))

    def n_inf(V):
        return 0.5 * (1 + np.tanh((V - par['V3']) / par['V4']))

    def I_Ca(V):
        return par['gCa'] * m_inf(V) * (V - par['ECa'])

    def dV_fast(V, Ca):   # with n = n_inf(V)
        n = n_inf(V)
        I_L   = par['gL'] * (V - par['EL'])
        I_K   = par['gK'] * n * (V - par['EK'])
        I_KCa = par['gKCa'] * (Ca / (Ca + 1)) * (V - par['EK'])
        return par['Iapp'] - (I_L + I_K + I_Ca(V) + I_KCa)

    def dV_full(V):       # full‑system eq. using dCa/dt=0 & dn/dt=0
        Ca = -(par['mu'] / par['kCa']) * I_Ca(V)
        n  = n_inf(V)
        I_L   = par['gL'] * (V - par['EL'])
        I_K   = par['gK'] * n * (V - par['EK'])
        I_KCa = par['gKCa'] * (Ca / (Ca + 1)) * (V - par['EK'])
        return par['Iapp'] - (I_L + I_K + I_Ca(V) + I_KCa)

    # ------------------------------------------------------------
    # 1) Fast‑subsystem equilibrium branch (curve)
    # ------------------------------------------------------------
    Ca_vals = np.linspace(0.0, 2, 501)     # slow‑variable parameter
    V_branch, n_branch, Ca_branch = [], [], []

    V_scan = np.linspace(-80, 30, 4001)        # coarse voltage grid for root bracketing
    for Ca in Ca_vals:
        f = dV_fast(V_scan, Ca)
        s = np.sign(f)
        idxs = np.where(np.diff(s))[0]
        for idx in idxs:
            V0, V1 = V_scan[idx], V_scan[idx+1]
            try:
                V_root = brentq(dV_fast, V0, V1, args=(Ca,))
                V_branch.append(V_root)
                n_branch.append(n_inf(V_root))
                Ca_branch.append(Ca)
            except ValueError:
                pass

    # ------------------------------------------------------------
    # 3) dCa/dt = 0 surface
    # ------------------------------------------------------------
    V_surf = np.linspace(-80, 30, 121)
    n_surf = np.linspace(0, 1, 51)
    V_mesh, n_mesh = np.meshgrid(V_surf, n_surf)
    Ca_mesh = -(par['mu'] / par['kCa']) * I_Ca(V_mesh)

    def MLburst(t: float,
                y: list[float, float, float],
                I=lambda t: 45,
                params: dict | None = None,
                epsilon=eps, mu=mu) -> np.ndarray:
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
                        Cm=20, kCa=1.0)

        V, n, Ca = y

        # steady‑state gates and time constant
        m_inf = 0.5 * (1 + np.tanh((V - params['V1']) / params['V2']))
        n_inf = 0.5 * (1 + np.tanh((V - params['V3']) / params['V4']))
        tau_n = 1 / np.cosh((V - params['V3']) / (2 * params['V4']))

        # ionic currents
        I_Ca  = params['gCa']  * m_inf * (V - params['ECa'])
        I_K   = params['gK']   * n      * (V - params['EK'])
        I_KCa = params['gKCa'] * (Ca / (Ca + 1)) * (V - params['EK'])
        I_L   = params['gL']   * (V - params['EL'])
        # fixed applied current
        I_app = I(t) if callable(I) else params.get('Iapp', 0.0) 

        # differential equations
        dVdt  = (I_app - I_L - I_K - I_Ca - I_KCa) / params['Cm']
        dndt  = params['phi'] * (n_inf - n) / tau_n
        dCadt = epsilon * (-mu * I_Ca - params['kCa'] * Ca)

        return np.array([dVdt, dndt, dCadt])

    # ------------------------------------------------------------
    # 4) Plot
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    tspan = (0, 10000)
    y0 = iniVALS

    sol = solve_ivp(MLburst, tspan, y0, max_step=0.1)

    # surface: semi‑transparent
    mask = Ca_mesh > np.max(sol.y[2]) + .2
    Ca_mesh[mask] = np.nan
    V_mesh[mask] = np.nan
    
    ax.plot_surface(Ca_mesh, n_mesh, V_mesh, alpha=0.3, rstride=1, cstride=1,
                    linewidth=0, antialiased=False, label = 'dCa/dt = 0')

    # fast‑equilibrium branch
    ax.scatter(Ca_branch, n_branch, V_branch, lw=1.0, color='gold',
            label='Fast‑subsystem equilibria')

    # full‑system equilibria
    def intersections(vars, par):
        V, Ca = vars
        return [
        dV_fast(V, Ca),
        Ca + (par['mu']/par['kCa'])*I_Ca(V)
        ]

    x0 = [-30, 0.5]
    # same initial guess
    Vstar, Castar = fsolve(intersections, x0, args=(par,))
    # if V_eq:
    ax.scatter(Castar, n_inf(Vstar), Vstar, s=100, color='red', label='Full equilibria')

    # burster orbit
    ax.plot(sol.y[2], sol.y[1], sol.y[0], color = 'magenta', label = 'Burster orbit')

    ax.set_xlabel('[Ca]')
    ax.set_ylabel('n')
    ax.set_zlabel('V (mV)')         # convention: V on z‑axis
    ax.view_init(elev=25, azim=-60)
    ax.legend(loc='upper right')
    # ax.set_xlim(0, 2)
    plt.tight_layout()
    plt.savefig(f'Eps{eps}_Mu{mu}_3D_manifolds.png', dpi = 300)
    plt.show() 

if __name__ == '__main__':
    # # Q2.2
    Q2_5_6()

    # # Q2.5
    for eps in [3e-4, 5e-4, 6e-4, 7e-4, 1e-3, 2e-3, 5e-3, 1e-2]:
        Q2_5_6(eps=eps)

    # # Q2.6
    # # intersection points move
    for mu in [2e-2, 6e-2, 1e-1, 2e-1]:
        Q2_5_6(mu=mu, iniVALS=[-10, 0.2, 0.25])
    
    # lowering mu = tonic spiking (similar to lowering epsilon)
    Q2_5_6(mu=1.8e-2, iniVALS=[-10, 0.2, 0.25])
