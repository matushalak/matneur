# @matushalak Mathematical neuroscience 2025
import numpy as np
import scipy.integrate as sp_integrate
import scipy.optimize as sp_optim
import scipy.signal as sp_sig
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from numpy.random import uniform
import os

# Basic helper functions 
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

def ICaL(V, Ca, 
         params: dict = {'Pmax': 0,#.002, 
                        'z':2, 'F':96520, 'Caout':2}
                                ):
    Idrive = lambda v, ca: params['Pmax'] * params['z'] * params['F'] * phi(v, ca)
    return m(V) * h(Ca) * Idrive(V, Ca)

# Model
def Ltype(t:float, vars:list[float], I:callable, ical: bool = True,
          params : dict = {'Cainf':1e-4, 'tauCa':200, 'Beta':0.01})->np.ndarray:
    '''
    2D Model
    Minimal model for L-type currents
    '''
    # unpack variables
    V, Ca = vars

    # currents
    IL = Ileak(V)
    ICaLt = ICaL(V, Ca) if ical else 0

    # changes in variables
    dvdt = I(t) - IL - ICaLt
    dCadt = (-params['Beta'] * ICaLt) - ((Ca - params['Cainf']) / params['tauCa'])

    return np.array([dvdt, dCadt])

# To plot voltage, calcium and related variables
def voltage_trace(duration: int, 
                  v_start:float | np.ndarray, 
                  ca_start:float | np.ndarray,
                  applied_current:tuple | float,
                  everything:bool = False,
                  eq_voltages:dict[str:float] | None = None, 
                  ical : bool = True,
                  name:str = '',
                  plot:bool = True,
                  show:bool = True,
                  n_samples:int = 1000,
                  return_fig: bool = False)->np.ndarray:
    '''
    Given a duration of simulation, set of initial conditions (v, ca), and set of applied current amplitudes,
    Calculates, and plots the voltage traces in reponse to applied current of specified strength.

    Returns solution trajectories and their derivatives
    '''
    if type(v_start) != np.ndarray:
        v_start = [v_start]
        ca_start = [ca_start]
    
    t_interval = [0, duration]
    TRAJECTORIES = []

    # prepare applied current function
    match applied_current:
        # Pulsed
        case Iamp, Istart, Istop, Period, Pulse_duration:
            Iapp = lambda t: Iamp if t >= Istart and t <= Istop and (t % Period) < Pulse_duration else 0
        # Stepped
        case Iamp, Istart, Istop:
            Iapp = lambda t: Iamp if t >= Istart and t <= Istop else 0
        # Stepped
        case Iamp, Istart:
            Iapp = lambda t: Iamp if t >= Istart else 0
        # Constant
        case Iamp:
            Iapp = lambda t: Iamp
    
    # calculate solutions separately for different starting conditions 
    for vs, cs in zip(v_start, ca_start):
        ltype = lambda t, vars: Ltype(t, vars, Iapp, ical = ical)
        solution = sp_integrate.solve_ivp(fun = ltype, t_span=t_interval, t_eval=np.linspace(0, duration, n_samples),
                            y0 = [vs, cs], method = 'RK45')
        
        TRAJECTORIES.append(solution.y)

    ts = solution.t
    TRAJECTORIES = np.array(TRAJECTORIES)
    retTraj = TRAJECTORIES.swapaxes(0,1)
    voltage, calcium  = TRAJECTORIES[:,0,:], TRAJECTORIES[:,1,:]

    if everything:
        voltage, calcium, ts= voltage.squeeze(), calcium.squeeze(), ts.squeeze()
        m_act = m(voltage)
        h_act = h(calcium)
        Il = Ileak(voltage)
        ICaLt = ICaL(voltage, calcium) if ical else np.zeros_like(ts)

    if plot:
        applied_I = [Iapp(t) for t in ts]

        # plotting traces
        nr = 4 if everything else 2
        fig, axes = plt.subplots(nrows=nr, ncols=1, figsize = (5,7), sharex = True)

        # Voltage plot
        match voltage.shape:
            case ntraces, nsampl:
                for itr in range(ntraces):
                    axes[0].plot(ts, voltage[itr,:], alpha = 0.5)    
            case nsampl:
                axes[0].plot(ts, voltage, color = 'k')

        if eq_voltages is not None:
            for eq, vlt in eq_voltages.items():
                axes[0].plot(ts, np.full(ts.shape, vlt), label = eq, color = 'orange', alpha = 0.3)
            axes[0].legend(loc = 4)
        axes[0].set_ylabel('Membrane Voltage (mV)')
        
        # Calcium concentration
        match calcium.shape:
            case ntraces, nsampl:
                for itr in range(ntraces):
                    axes[1].plot(ts, calcium[itr,:], alpha = 0.5)
            case nsampl:
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
        if not return_fig:
            plt.savefig(f'voltage_trace_{name}.png', dpi = 300)
            if show:
                plt.show()
            plt.close()
    if return_fig:
        return retTraj, fig
    else:
        return retTraj


# --------------- Q1.4 -------------------
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


# q1.5
def fV(v, ca, 
       Iapp = 1):
    # The voltage nullcline: f(V,Ca) = I_app - Ileak(V) - ICaL(V,Ca)
    return Iapp - Ileak(v) - ICaL(v, ca)

def fCa(v, ca, 
        params : dict = {'Cainf':1e-4, 'tauCa':200, 'Beta':0.01}):
    # The calcium nullcline: g(V,Ca) = -Beta * ICaL(V,Ca) - (Ca - Cainf)/tauCa
    return (-params['Beta'] * ICaL(v, ca)) - ((ca - params['Cainf']) / params['tauCa'])

def equilibrium_guess(args, I = 1):
    v, ca = args
    return [fV(v, ca, I), fCa(v, ca)]


# Workhorse, does a lot, quite proud of this function
def phase_portait(V_range: tuple, Ca_range: tuple, density: int = 1000, I: float = 1,
                  eq_ini: tuple | None = None, save:bool = False, plot:bool = True,
                  trajectories: None | np.ndarray = None,
                  plot0nullcline:bool = False, show:bool = False, return_fig: bool = False):
    '''
    Gives phase portrait with equilibria and nullclines (numerically determined) at one level of applied current
    '''
    # nullclines
    vrange = np.linspace(*V_range, num=density)
    carange = np.linspace(*Ca_range, num=density)

    # points where we will explore the values of fV and fCa
    VV, CC = np.meshgrid(vrange, carange)
    DVDT = np.array([[fV(v = x, ca = y, Iapp=I) for x, y in zip(rows, cols)] for rows, cols in zip(VV, CC)])
    if plot0nullcline:
        dvdt0 = np.array([[fV(v = x, ca = y, Iapp=0) for x, y in zip(rows, cols)] for rows, cols in zip(VV, CC)])
    DCaDT = np.array([[fCa(v = x, ca = y) for x, y in zip(rows, cols)] for rows, cols in zip(VV, CC)])
    
    # equilibria (roots)
    if eq_ini is None:
        # finding equilibria from the drawn grid with min Sum of Squares
        SS = DVDT**2 + DCaDT**2 # points where both closest to 0
        min_index = np.argmin(SS)
        i, j = np.unravel_index(min_index, SS.shape)
        V_guess = VV[i, j]
        Ca_guess = CC[i, j]
        x0 = (V_guess, Ca_guess) if I != 0 else (-70, 0.05)# guess based on voltage plot
    else:
        x0 = eq_ini
    equil_sol = sp_optim.fsolve(equilibrium_guess, x0, args = I, xtol=1e-9)
    print(equil_sol)
    Stable, Type = stability(equil_sol, Iapp=I)
    stabilities = {True:'Stable', False:'Unstable'}
    types = {0: 'saddle', 1: 'node (real)', 2: 'focus (complex)'}
    print(f'{stabilities[Stable]} {types[Type]}')
    
    # PLOTTING!
    if plot:
        # Use contour to plot the implicit curves DVDT=0 and DCaDT=0
        # levels = [a, b, c] plots contour lines where Z == a / b / c, 
        # we choose only 0, so just the nullclines are plotted
        fig, ax = plt.subplots(figsize=(8,6))
        if plot0nullcline:
            # Black contour for V-nullcline at Iapp = 0
            vnull0 = ax.contour(VV, CC, dvdt0, levels=[0], colors='black')    
        # Blue contour for V-nullcline
        vnull = ax.contour(VV, CC, DVDT, levels=[0], colors='teal')
        # Red contour for Ca-nullcline
        canull = ax.contour(VV, CC, DCaDT, levels=[0], colors='fuchsia')
        # equilibria
        eq = ax.scatter(equil_sol[0], equil_sol[1], color = 'g' if Stable else 'r', label = f'{stabilities[Stable]} {types[Type]} Equilibrium \n{np.round(equil_sol, 3)}', 
                        zorder = 2, s = 50)
        
        # vector field
        # discrete colors give interpretability to different areas of phase plane
        quadrant = (np.sign(DVDT) > 0).astype(int) + 2 * (np.sign(DCaDT) > 0).astype(int) 
        custom_cmap = ListedColormap(['lightcoral', 'blue', 'orange', 'limegreen'])
        ax.quiver(VV, CC, DVDT, DCaDT,
                quadrant, cmap = custom_cmap, 
                pivot = 'tip',angles='xy',width=0.002, zorder = 0, alpha = 0.5)
        
        if trajectories is not None:
            for tr in range(trajectories.shape[1]):
                print('Trajectory', tr)
                ax.plot(trajectories[0,tr,:], trajectories[1,tr,:], alpha = 0.9, linestyle = '-')
                # startpoints
                ax.scatter(trajectories[0,tr,0], trajectories[1,tr,0], marker = '*')

        ax.set_xlabel('Voltage (mV)')
        ax.set_ylabel('Calcium (M)')
        ax.set_title(f'Phase portrait for I_app = {round(I,4)}')
        
        # fancy legend nonsense (ignore)
        # quadrant labels and arrows
        # Create the 4 arrows for your quadrants (dx,dy) 
        #   Q0: down-left,   Q1: down-right, 
        #   Q2: up-left,     Q3: up-right
        arrow_Q0 = make_arrow(-0.2, -0.2, 'lightcoral')
        arrow_Q1 = make_arrow( 0.2, -0.2, 'blue')
        arrow_Q2 = make_arrow(-0.2,  0.2, 'orange')
        arrow_Q3 = make_arrow( 0.2,  0.2, 'limegreen')

        legend_arrows = [arrow_Q0,
                        arrow_Q1,
                        arrow_Q2,
                        arrow_Q3]
        
        legend_labels = [r"$\frac{dv}{dt} < 0$, $\frac{dCa}{dt} < 0$",
                        r"$\frac{dv}{dt} > 0$, $\frac{dCa}{dt} < 0$",
                        r"$\frac{dv}{dt} < 0$, $\frac{dCa}{dt} > 0$",
                        r"$\frac{dv}{dt} > 0$, $\frac{dCa}{dt} > 0$"]

        # nullcline labels
        vlabeldummy = mlines.Line2D([], [], color='teal', label='V-nullcline')
        calabeldummy = mlines.Line2D([], [], color='fuchsia', label='Ca-nullcline')
        
        ax.legend(handles=[vlabeldummy, calabeldummy, eq, *legend_arrows],
                labels=["V-nullcline", "Ca-nullcline", f'{stabilities[Stable]} {types[Type]} \n{np.round(equil_sol, 3)}'] + legend_labels,
                loc='upper right',
                handler_map={mpatches.FancyArrowPatch: HandlerArrow()},
                handletextpad=1.2, labelspacing=1.2
                )
        
        fig.tight_layout()
        if not return_fig:
            if save:
                if trajectories is None:
                    if not os.path.exists(savedir := 'phase_portraits'):
                        os.makedirs('phase_portraits')
                    savename = f'Phase_portrait_(I={round(I, 4)}).png'
                    plt.savefig(os.path.join(savedir, savename), dpi = 300)
                else:
                    savename = f'Phase_portrait_(I={round(I, 4)})_trajectories.png'
                    plt.savefig(savename, dpi = 300)

            if show:
                plt.show()
            plt.close()
    if return_fig:
        return fig
    else:
        return x0


# -------------- Q1.6 S3 --------------------
def transients(V_range=(-80, 25), Ca_range=(0, 1.65), show:bool=False):
    # define a range of initial conditions
    # random
    # v_vals = np.random.uniform(*V_range, size = 10)
    # ca_vals = np.random.uniform(*Ca_range, size = 10)
    # hand-picked

    v_vals = np.asarray([-75, -45, -30, -50])
    ca_vals = np.asarray([1.4, 0.1, 0.02, 1.62])
    currents = [(iapp, 1600) for iapp in np.linspace(0,3,40)]
    n_ts = 1000

    # all_signal = np.zeros((2, len(currents) * len(v_vals), n_ts))
    current_ranges = [(i*len(v_vals), (i+1)*len(v_vals)) for i in range(len(currents))]
    
    for ic, i_config in enumerate(currents):
        # returns a (nsignals, n_ini_conditions, n_time_samples)
        #   nsignals are [voltage_trace, calcium_trace]
        res = voltage_trace(duration = 10000, v_start=v_vals, ca_start=ca_vals, applied_current=i_config,
                            plot=True, n_samples=n_ts, name=f'Istep_{ic}', show=show)
        
        phase_portait(V_range=(-80, 35), Ca_range=(0, 1.7), density=200, I = i_config[0], trajectories=res,
                      plot0nullcline=True, 
                      plot=True,  save=True, show=show)

# semi-successful attempt at bifucations
# -------------- Bifurcations ----------
# works
def estimate_Jacobian(local_vars: np.ndarray, Iapp: float, perturb = 1e-6
                      ) -> np.ndarray:
    '''
    Estimates 2 x 2 Jacobian at [V, Ca] using finite differences
    '''
    f0 = Ltype(0, local_vars, I = lambda t: Iapp) # dv/dt, dCa/dt
    J = np.zeros((2,2)) # jacobian
    
    for i in range(2):
        perturbation = np.zeros(2)
        perturbation[i] = perturb
        f1 = Ltype(0, local_vars + perturbation, I = lambda t: Iapp) # dv/dt, dCa/dt
        J[:, i] = (f1 - f0) / perturb

    return J


# works
def stability(local_vars: np.ndarray, Iapp: float)-> float:
    '''
    Returns: (stability, type)
        Stability
        ---------
        True -> stable, attracting
        False -> unstable, repelling / saddle

        Type
        ----
        0 -> saddle
        1 -> node (real)
        2 -> focus (complex)
    '''
    # get Jacobian
    J = estimate_Jacobian(local_vars, Iapp)

    # eigenvalues
    eig_J = np.linalg.eigvals(J)
    real = np.real(eig_J)
    imag = np.imag(eig_J)

    # trace
    trace_J = np.trace(J)
    # determinant
    det_J = np.linalg.det(J)

    if det_J < 0:
        assert np.any(real > 0) and np.any(real < 0), 'Jacobian Eigenvalues should have opposite signs for saddle'
        return False, 0 # saddle
    
    elif trace_J > 0:
        assert np.all(real > 0), 'Real parts should be > 0 for unstable equilibrium'
        if np.allclose(imag, 0, atol=1e-8):
            return False, 1 # unstable, node (real)
        else:
            return False, 2 # unstable, focus (complex)
    
    elif trace_J < 0:
        assert np.all(real < 0), 'Real parts should be < 0 for stable equilibrium'
        if np.allclose(imag, 0, atol=1e-8):
            return True, 1 # stable, node (real)
        else:
            return True, 2 # stable, focus (complex)
    
    else:
        print(f'Borderline case at [V, Ca] = {local_vars} with eigenvalues = {eig_J}')
        return 10,10


# Doesn't work properly, MatCont worked better 
def find_limit_cycle(I: float, ini_v_ca: tuple) -> tuple | None:
    '''
    Returns Min and Max of limit cycle IF limit cycle was encountered, else None
    '''
    ltype = lambda t, vars: Ltype(t, vars, I = lambda t: I)
    v_start, ca_start = ini_v_ca
    t_span = (0, 40000)  # Run for a long time (adjust based on timescales)
    t_eval = np.linspace(t_span[0], t_span[1], 80000)
    solution = sp_integrate.solve_ivp(fun = ltype, t_span=t_span, t_eval=t_eval,
                         y0 = [v_start, ca_start], method = 'RK45')
    ts = solution.t
    voltage, calcium = solution.y

    # Cut away transients and only look at last 20% tail of signal
    # if in limit cycle, will still be in the cycle at the end
    tail = int(0.95 * len(ts))
    # peaks?
    peaks, _ = sp_sig.find_peaks(voltage[tail:])
    t_peaks = ts[tail:][peaks]
    v_peaks = voltage[tail:][peaks]
    if len(t_peaks) > 1:
        period_estimate = np.mean(np.diff(t_peaks))
        print(f"Estimated period from {len(peaks)} peaks:", period_estimate)
        return np.min(v_peaks), np.max(v_peaks)
    else:
        print("Not enough peaks found; adjust threshold or simulation time.")
        return None

# Doesn't work properly, MatCont worked better 
def bifurcations(Irange: tuple = (0, 3), stepsize: float = 0.01, 
                 Ivals: np.ndarray | None = None, x0vals: np.ndarray | None = None):
    # bifurcation parameter
    I_values = np.arange(*Irange, stepsize) if Ivals is None else Ivals
    # to store results
    V_stable = []
    V_unstable = []
    V_cycle_min = []
    V_cycle_max = []

    ini_guess = (-70, 0.05)# guess based on voltage plot
    last_equilibrium = sp_optim.fsolve(equilibrium_guess, ini_guess, args=(Irange[0],)) # For I = 0

    for i, I_val in enumerate(I_values):
        # 1) Attempt to find equilibrium near last_equilibrium
        if x0vals is None:
            eq_sol = sp_optim.fsolve(equilibrium_guess, last_equilibrium, args=(I_val,))
        else:
            eq_sol = x0vals[i]

        # 2) Check stability (via estimated Jacobian):
        stable_eq, type = stability(eq_sol, I_val) 

        # 3) If stable, add eq_sol[0] to V_stable. If not stable, put eq_sol in V_unstable
        if stable_eq:
            V_stable.append((I_val, eq_sol[0]))  # store voltage eq
        else:
            V_unstable.append((I_val, eq_sol[0]))

            # Attempt time simulation to find limit cycle
            # e.g. run a 1000-ms simulation from a perturbation
            cycle_data = find_limit_cycle(I_val+stepsize, eq_sol)
            if cycle_data is not None:
                Vmin, Vmax = cycle_data
                V_cycle_min.append((I_val, Vmin))
                V_cycle_max.append((I_val, Vmax))

        # 4) Update last_equilibrium for next iteration
        last_equilibrium = eq_sol
    
    V_stable = np.array(V_stable)
    V_unstable = np.array(V_unstable)
    V_cycle_min = np.array(V_cycle_min)
    V_cycle_max = np.array(V_cycle_max)

    plot_bifurcation(V_stable, V_unstable, V_cycle_min, V_cycle_max)

# Doesn't work properly, MatCont worked better 
def plot_bifurcation(Vstable, V_unstable, V_cycle_min, V_cycle_max):
    fig, axs = plt.subplots()
    axs.scatter(Vstable[:,0], Vstable[:,1], color = 'blue')
    axs.plot(V_unstable[:,0],V_unstable[:,1], linestyle = '--', color = 'red')
    axs.plot(V_cycle_min[:,0],V_cycle_min[:,1], linestyle = '-', color = 'purple')
    axs.plot(V_cycle_max[:,0],V_cycle_max[:,1], linestyle = '-', color = 'purple')

    plt.tight_layout()
    plt.show()


#----------------- Q 1.5 -----------------
# Explore phase portrait across range of applied currents
def move_through_Iapp():
    # move through a range of possible applied currents
    x0 = None
    critical_region = np.arange(1.1, 1.4, 0.025) # -> bifurcation around 1.29
    full_range = np.concat([np.arange(0, critical_region[0], 0.1), critical_region, np.arange(critical_region[-1], 6, 0.1)])
    full_range = np.arange(5.775, 6.2, 0.1)
    # x0s = []
    for Iapp in full_range:
        # Iapp = round(Iapp, 2)
        # continuation - using the equilibrium from the previous bifurcation parameter
        x0 = phase_portait(V_range=(-80, 100), Ca_range=(0, 1.6), density=300, I = Iapp, plot=True, save=True)
        print(f'Phase portrait for Iapp = {round(Iapp, 4)} done!')
        # x0s.append(x0)


# ------------ FANCY PLOTTING STUFF (ignore) -------------
def make_arrow(dx, dy, color='black', arrowstyle='-|>', mutation_scale=15, linewidth=.5):
    arrow = mpatches.FancyArrowPatch((0, 0), (dx, dy),
                                    arrowstyle=arrowstyle,
                                    mutation_scale=mutation_scale,
                                    color=color,
                                    linewidth=linewidth)
    # Store the direction as attributes so the handler can use them later.
    arrow.dx = dx
    arrow.dy = dy
    return arrow

class HandlerArrow(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Retrieve arrow desired direction (stored in the dummy handle)
        dx = getattr(orig_handle, 'dx', 1.0)
        dy = getattr(orig_handle, 'dy', 0.0)
        
        # Compute scale factors so that the arrow fits within the legend handle box.
        if abs(dx) < 1e-12:
            scale_x = 1
        else:
            scale_x = width / abs(dx)
        if abs(dy) < 1e-12:
            scale_y = 1
        else:
            scale_y = height / abs(dy)
        scale = min(scale_x, scale_y)
        
        # Scale the arrow components:
        dx_scaled = dx * scale
        dy_scaled = dy * scale
        
        # Compute the center of the allocated handle box.
        x_center = xdescent + 0.5 * width
        y_center = ydescent + 0.5 * height
        
        # To center the arrow, shift its starting point so that the entire arrow is centered.
        # The arrow’s total horizontal length is dx_scaled.
        x_start = x_center - 0.5 * dx_scaled
        y_start = y_center - 0.5 * dy_scaled
        x_end = x_start + dx_scaled
        y_end = y_start + dy_scaled
        
        arrow = mpatches.FancyArrowPatch(
            (x_start, y_start), (x_end, y_end),
            arrowstyle=orig_handle.get_arrowstyle(),
            mutation_scale=fontsize,  # adjust arrow head size with font size
            color=orig_handle.get_edgecolor() if hasattr(orig_handle, 'get_edgecolor') 
                  else orig_handle.get_facecolor(),
            linewidth=orig_handle.get_linewidth() if hasattr(orig_handle, 'get_linewidth')
                  else 1.5,
        )
        arrow.set_transform(trans)
        return [arrow]

# -------- main block that runs the script ---------
if __name__ == '__main__':
    ''''
    full params:
    params: dict = {'gL':0.05,'EL':-70,
                    'Pmax': 0.002, 'ki':0.001, 
                    'z':2, 'F':96520, 'Caout':2, 'R':8313.4, 'T':273.15 + 25,
                    'Cainf':1e-4, 'tauCa':200, 'Beta':0.01}
    '''
    # # Question 1.2 - decoupled system with no L-type calcium current
    decoupled = voltage_trace(4500, -70, .01, applied_current = (
                                                                # 0 # constant
                                                                # 2, 500, 1000 # stepped
                                                                2, 1000, 3400, 500, 50 # pulsed
                                                                    ), 
                                ical=False, # decoupling
                                everything=True, eq_voltages={'El':-70})
    
    # # Question 1.4 
    window_current()

    # Question 1.5 phase portrait and nullclines with constant applied current I(all_t) = 1
    phase_portait(V_range=(-80, 100), Ca_range=(0, 1.6), density=300, I = 1, plot=True)
    # # phase portrait across iapp
    move_through_Iapp()
    
    # Question 1.6
    # S3
    transients()
    # S4
    # couldn't get bifurcations to work 100%, used MatCont in the end
    # bifurcations(Irange=(1.0, 1.7), stepsize=1e-3)#, Ivals=full_range, x0vals=x0s)

