# @matushalak Mathematical neuroscience 2025
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import uniform
from scipy.integrate import solve_ivp
import os

def m(v: float | np.ndarray) -> float | np.ndarray:
    alpha = lambda v: 0.055 * ((-27.01 - v) / (np.exp((-27.01 - v) / 3.8) - 1))
    beta = lambda v: 0.94 * np.exp((-63.01 - v) / 17)
    return alpha(v) / (alpha(v) + beta(v)) # activation only depends on voltage

def h(Ca: float | np.ndarray, ki = 0.001) -> float | np.ndarray:
    return ki / (ki + Ca)

# Define the constant-field (Goldman-Hodgkin-Katz) function phi(V, Ca)
def phi(V, Ca, params: dict = {'z':2, 'F':96520, 'Caout':2, 'R':8313.4, 'T':273.15 + 2}):
    xi = (params['z'] * params['F'] * V) / (params['R'] * params['T'])
    return xi * (Ca - params['Caout'] * np.exp(-xi)) / (1 - np.exp(-xi))


def Ltype(t:float, vars:list[float], I:callable, 
          verbose: bool = False,
          params : dict = {'gL':0.05,'EL':-70,
                           'Pmax': 0.002, 'ki':0.001, 
                           'z':2, 'F':96520, 'Caout':2, 'R':8313.4, 'T':273.15 + 25,
                           'Cainf':1e-4, 'tauCa':200, 'Beta':0.01})->np.ndarray:
    '''
    2D Model
    Minimal model for L-type currents
    '''
    # unpack variables
    V, Ca = vars

    # gating variables

    # currents
    xi = lambda v: (params['z'] * params['F'] * v )/ (params['R'] * params['T'])
    Idrive = lambda v, Ca: params['Pmax'] * params['z'] * params['F'] * xi(v) * ((Ca - params['Caout'] * np.exp(-xi(v))) / (1 - np.exp(-xi(v))))
    
    Ileak = params['gL'] * (V - params['EL'])
    ICaL = m(V) * h(Ca) * Idrive(V, Ca)

    # changes in variables
    dvdt = I(t) - Ileak - ICaL
    dCadt = (-params['Beta'] * ICaL) - ((Ca - params['Cainf']) / params['tauCa'])

    if verbose:
        return np.array([m(V), h(Ca), Ileak, ICaL])
    else:
        return np.array([dvdt, dCadt])

def voltage_trace(duration: int, 
                  v_start:float | np.ndarray, 
                  ca_start:float | np.ndarray,
                  applied_current:tuple | float,
                  everything:bool = False,
                  eq_voltages:dict[str:float] | None = None)->np.ndarray:
    '''
    Given a duration of simulation, set of initial conditions (v, ca), and set of applied current amplitudes,
    Calculates, and plots the voltage traces in reponse to applied current of specified strength.

    Returns solution trajectories and their derivatives
    '''
    t_interval = [0, duration]
    # prepare applied current function
    match applied_current:
        case Iamp, Istart, Istop, Period, Pulse_duration:
            Iapp = lambda t: Iamp if t >= Istart and t <= Istop and (t % Period) < Pulse_duration else 0
        case Iamp, Istart, Istop:
            Iapp = lambda t: Iamp if t >= Istart and t <= Istop else 0
        case Iamp:
            Iapp = lambda t: Iamp
    
    ltype = lambda t, vars: Ltype(t, vars, Iapp)
    solution = solve_ivp(fun = ltype, t_span=t_interval, t_eval=np.linspace(0, duration, 5000),
                         y0 = [v_start, ca_start], method = 'RK45')
    
    ts = solution.t
    voltage, calcium = solution.y
    if everything:
        all_res = [Ltype(t, (v, ca), Iapp, verbose = True) for t, v, ca in zip(ts, voltage, calcium)]
        m_act, h_act, Il, ICaL = np.array(all_res).reshape(4,-1)

    applied_I = [Iapp(t) for t in ts]

    # plotting traces
    nr = 4 if everything else 2
    fig, axes = plt.subplots(nrows=nr, ncols=1, figsize = (10,14), sharex = True)

    # Voltage plot
    axes[0].plot(ts, voltage, color = 'k')
    if eq_voltages is not None:
        for eq, vlt in eq_voltages.items():
            axes[0].plot(ts, np.full(ts.shape, vlt), label = eq, color = 'orange', alpha = 0.3)
        axes[0].legend(loc = 4)
    axes[0].set_ylabel('Membrane Voltage (mV)')
    
    # Calcium concentration
    axes[1].plot(ts, calcium, color = 'darkred')
    axes[1].set_ylabel(r'$[\text{Ca}^{2+}]_{in} (M)$')

    if everything:
        # Gating variables
        # Activation & Inactivation variables
        axes[2].plot(ts, m_act, label = 'L-type act (m)', color = 'b')
        axes[2].plot(ts, h_act, label = 'L-type inact (h)', color = 'g')
        axes[2].set_ylabel('Activation variables')
        axes[2].legend(loc = 4)

        # Applied Current & IL & ICaL
        axes[-1].plot(ts, applied_I, label = 'applied', color = 'k')
        axes[-1].plot(ts, Il, label = r'$I_{leak}$', color = 'orange')
        axes[-1].plot(ts, ICaL, label = r'$I_{Ca,L}$', color = 'r', alpha = 0.5)
        axes[-1].set_ylabel('Current (µA)')
        axes[-1].set_xlabel('Time [ms]')
        axes[-1].legend(loc=4)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    plt.close()



def window_current():
    v_range = np.linspace(-90, 25, 200)
    ms = m(v_range)
    Ca_range = np.linspace(0, .0175, 200)
    hs = h(Ca_range)

    fig, axs = plt.subplots(1, 2, sharey = True)
    # m(V) - activation variable
    axs[0].plot(v_range, ms, marker = 'o', label = r'$m_{\infty}$(V)', color = 'b')
    # 90 % activated
    m90 = np.searchsorted(ms, 0.9)
    axs[0].vlines(v_range[m90], -.05, ms[m90], color = 'r')
    axs[0].hlines(ms[m90], -100, v_range[m90], color = 'r')
    axs[0].plot(v_range[m90], ms[m90], marker = '*', label = f'{round(v_range[m90], 2)} mV', color = 'r')    
    axs[0].set_xlabel('Voltage (mV)')
    axs[0].set_ylabel('Activation state')
    axs[0].set_xlim(-90, 25)
    axs[0].set_ylim(-0.05, 1.05)
    axs[0].legend(loc = 6, fontsize = 9)
    # h(V) - inactivation variable
    axs[1].plot(Ca_range, hs, marker = 'o', label = 'h(Ca)', color = 'g')
    # 90 % inactivated
    h90 = -np.searchsorted(hs[::-1], 0.1)
    axs[1].vlines(Ca_range[h90], -.05, hs[h90], color = 'orange')
    axs[1].hlines(hs[h90], -.0005, Ca_range[h90], color = 'orange')    
    axs[1].plot(Ca_range[h90], hs[h90], marker = 'h', label = f'{round(Ca_range[h90], 3)} M', color = 'orange')    
    axs[1].set_xlabel(r'$[\text{Ca}^{2+}]_{in} (M)$')
    axs[1].set_xlim(-0.0005, 0.0175)
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].legend(loc = 5, fontsize = 9)
    fig.tight_layout()
    plt.savefig('Ltype_window.png', dpi = 300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # run the model
    # trajectories = voltage_trace(20000, -70, 1e-4, applied_current = (5, 0, 20000, 200, 20), 
    #                              everything=True, eq_voltages={'El':-70})
    window_current()


''''
1.3
ACTIVATION is instantaneous m = minf(V), no differential equation for m
vs INACTIVATION slow, depends on calcium buildup (which is SLOW - ∆Ca equation is slow)
'''

''''
1.4
window current plot
l=type long lasting inactivate only at high ca (high ca is slow)
'''

''''
1.5 phase portrait and nullclines with constant applied current I(all_t) = 1
'''

''''
1.6
Find bifurcation parameters and make bifurcation diagram
'''