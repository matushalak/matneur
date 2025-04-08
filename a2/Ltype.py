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
def phi(V, Ca, params: dict = {'z':2, 'F':96520, 'Caout':2, 'R':8313.4, 'T':273.15 + 25}):
    xi = (params['z'] * params['F'] * V) / (params['R'] * params['T'])
    return xi * (Ca - params['Caout'] * np.exp(-xi)) / (1 - np.exp(-xi))

def Ileak(V, params : dict = {'gL':0.05,'EL':-70}):
    return params['gL'] * (V - params['EL'])

def ICaL(V, Ca, params: dict = {'Pmax': 0.002, 'z':2, 'F':96520, 'Caout':2}):
    Idrive = lambda v, ca: params['Pmax'] * params['z'] * params['F'] * phi(v, ca)
    return m(V) * h(Ca) * Idrive(V, Ca)

def Ltype(t:float, vars:list[float], I:callable,
          params : dict = {'Cainf':1e-4, 'tauCa':200, 'Beta':0.01})->np.ndarray:
    '''
    2D Model
    Minimal model for L-type currents
    '''
    # unpack variables
    V, Ca = vars

    # currents
    IL = Ileak(V)
    ICaLt = ICaL(V, Ca)

    # changes in variables
    dvdt = I(t) - IL - ICaLt
    dCadt = (-params['Beta'] * ICaLt) - ((Ca - params['Cainf']) / params['tauCa'])

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
        # Pulsed
        case Iamp, Istart, Istop, Period, Pulse_duration:
            Iapp = lambda t: Iamp if t >= Istart and t <= Istop and (t % Period) < Pulse_duration else 0
        # Stepped
        case Iamp, Istart, Istop:
            Iapp = lambda t: Iamp if t >= Istart and t <= Istop else 0
        # Constant
        case Iamp:
            Iapp = lambda t: Iamp
    
    ltype = lambda t, vars: Ltype(t, vars, Iapp)
    solution = solve_ivp(fun = ltype, t_span=t_interval, t_eval=np.linspace(0, duration, 5000),
                         y0 = [v_start, ca_start], method = 'RK45')
    
    ts = solution.t
    voltage, calcium = solution.y
    if everything:
        m_act = m(voltage)
        h_act = h(calcium)
        Il = Ileak(voltage)
        ICaLt = ICaL(voltage, calcium)

    applied_I = [Iapp(t) for t in ts]

    # plotting traces
    nr = 4 if everything else 2
    fig, axes = plt.subplots(nrows=nr, ncols=1, figsize = (5,7), sharex = True)

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
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend(loc = 1)

        # Applied Current & IL & ICaL
        axes[-1].plot(ts, applied_I, label = 'applied', color = 'k')
        axes[-1].plot(ts, Il, label = r'$I_{leak}$', color = 'orange')
        axes[-1].plot(ts, ICaLt, label = r'$I_{Ca,L}$', color = 'r', alpha = 0.5)
        axes[-1].set_ylabel('Current')
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
    ''''
    full params:
    params: dict = {'gL':0.05,'EL':-70,
                    'Pmax': 0.002, 'ki':0.001, 
                    'z':2, 'F':96520, 'Caout':2, 'R':8313.4, 'T':273.15 + 25,
                    'Cainf':1e-4, 'tauCa':200, 'Beta':0.01}
    '''
    # run the model
    trajectories = voltage_trace(5000, -70, 1e-4, applied_current = (
                                                                    # -2 # constant
                                                                    2, 500, 1000 # stepped
                                                                    # 3, 500, 5000, 2000, 50 # pulsed
                                                                     ), 
                                 everything=True, eq_voltages={'El':-70})
    # window_current()


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