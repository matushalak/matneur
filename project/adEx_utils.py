#@matushalak
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Literal

# optimizations
import numba as nb
from adex_cython import adExcython_wrapper # conversion to C too expensive because solve_ivp in python

# Default Experimental parameters from original definition of adEx model!
# Parameters (from Brette & Gerstner 2005, Table 1, "regular spiking")
def default_experiment()->tuple[int, float, dict, dict, callable]:
    Tmax = 1000
    dt = 0.1
    
    model_params = {
        'C' : 281.0,     # pF
        'gL' : 30.0,     # nS
        'EL' : -70.6,    # mV
        'VT' : -50.4,    # mV
        'DeltaT' : 2.0,   # mV
        'Vreset' : -70.6, # mV
        'Vpeak' : 0.0, # mV - can't be more than 5 due to numerical overflow in scipy
        'tauw' : 144.0,  # ms
        'a' : 4.0,       # nS
        'b' : 80.5    # pA !!!!
    }
    experiment_settings = {
        'figure' : None,
        'custom' : None,
        'Istart' : 0, 'Istop' : Tmax, 
        'Iperiod' : 100, 'Iduration' : 60, 
        'Iamplitude' : 1000
        }
    
    # default periodically pulsed current
    current:callable = lambda t: Iapp(t, **experiment_settings)
    
    return Tmax, dt, model_params, experiment_settings, current


def define_experiment(**kwargs)->tuple[int, float, dict, dict, callable]:
    '''
    Only need to define deviations from default adEx values taken from
    Brette & Gerstner 2005, (Table 1, "regular spiking")
    '''
    Tmax, dt, model_params, experiment_settings, _ = default_experiment()
    # Update default model with user specifications
    for arg, argval in kwargs.items():
        if arg == 'Tmax':
            Tmax = argval
            # to maintain default intended behavior throughout recorded session
            if 'Istop' not in kwargs:
                experiment_settings['Istop'] = Tmax
        elif arg == 'dt':
            dt = argval
        elif arg in model_params:
            model_params[arg] = argval
        elif arg in experiment_settings:
            experiment_settings[arg] = argval
        else:
            raise KeyError(f'''{arg} is not a valid parameter name, see valid parameter names: 
                           Tmax, dt, 
                           model_params:{list(model_params)} 
                           experiment_settings:{list(experiment_settings)}''')
    
    # user-defined input current
    current:callable = lambda t: Iapp(t, **experiment_settings)
    return Tmax, dt, model_params, experiment_settings, current


def Iapp(t:float, 
         figure:Literal['small_large', 'hyperpol'] | None,
         custom:callable, 
         Istart:int, Istop:int, Iperiod:int, Iduration:int, 
         Iamplitude:float):
    '''
    Figure argument is used to reproduce figures from Brette & Gerstner (2005)
    '''
    # custom function supplied
    if custom is not None:
        return custom(t)
    if figure is None:
        # specified by parameters
        return Iamplitude if t >= Istart and t <= Istop and (t % Iperiod) < Iduration else 0
    elif figure ==  'small_large':
        # Fig 2C / 3C - small and large pulse with Vreset = EL OR Vreset = Vt + 3
        return 500 if t < 200 else(0 if t < 500 else 800)
    elif figure ==  'hyperpol':
        # Fig 3D - rebound spike
        return -800 if 10 < t < 410 else 0

@nb.njit(fastmath = True)
def forward_euler(y0:tuple[float, float], dt:float, 
                  model_params:tuple[float,...], reset_params:tuple[float,...],
                  ts:np.ndarray, currents:np.ndarray):
    V, w = y0
    Vout = np.zeros(ts.size, dtype=np.float64)
    wout = np.zeros(ts.size, dtype=np.float64)
    spts = np.zeros(ts.size, dtype=np.bool)

    # unpack reset parameters
    Vpeak, Vreset, b = reset_params
    C, gL, EL, VT, DeltaT, tauw, a = model_params
    for i, t in enumerate(ts):
        '''
        Brette & Gerstner 2005 adEx neuron WITHOUT synapses
        responding to input current
        '''
        dV = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + currents[i]) / C
        dw = (a * (V - EL) - w) / tauw

        # dV, dw = ADEX(t=t, y=(V, w), I = currents[i], params = model_params)
        # need to get change PER dt
        V2, w2 = V + dV*dt, w+dw*dt
        # spike
        if V2 >= Vpeak:
            V2 = Vreset
            w2 = w + b
            # Count spike occured at this time
            spts[i] = True
            Vout[i-1] = Vpeak
        
        # save results
        V, w = V2, w2
        Vout[i] = V
        wout[i] = w
    
    return Vout, wout, spts, ts


def run_experiment(adExModel:callable, Tmax:int, dt:float, model_params:dict, Iapp:callable, 
                   plot:bool = False, simple:bool = True):
    '''
    Runs entire experiment for adEx model of choice with model and experimental parameters of choice
    '''        
    y0 = (model_params['EL'], 0.0)
    t0 = 0.0
    ts = np.arange(t0, Tmax, dt, dtype=float)
    all_curr = np.array([Iapp(t) for t in ts], dtype=float)

    # Clock-based simulator (constant step sizes), simple forward euler method
    if simple:
        model_tuple = tuple([model_params[par] for par in ['C', 'gL', 'EL', 'VT', 'DeltaT', 'tauw', 'a']])
        reset_tuple = tuple([model_params[par] for par in ['Vpeak', 'Vreset', 'b']])
        V_all, w_all, spikes, t_all = forward_euler(y0=y0, dt=dt,
                                                    model_params=model_tuple, 
                                                    reset_params=reset_tuple,
                                                    ts= ts, 
                                                    currents=all_curr)
        # spikes = np.array(spikes, dtype=bool)
        spike_times = t_all[spikes]
    
    # Event-based simulator (adaptive step sizes)
    else:
        def spike_event(t, y):
            return y[0] - model_params['Vpeak']
        spike_event.terminal = True
        spike_event.direction = 1

        t_all = []
        V_all = []
        w_all = []
        spike_times = []

        model = lambda t, y: adExModel(t, y, params=model_params, I = Iapp(t))
        # cython wrapper
        # model = lambda t, y: adExcython_wrapper(t, y, params=model_params, t_array = ts, i_array = all_curr)

        while t0 < Tmax:
            # this is the slow step, each lambda call + solve_ivp is implemented in Python!
            # simply cython does not solve the problem
            sol = solve_ivp(fun=model, t_span=(t0, Tmax), y0 = y0, 
                            events=spike_event, max_step=dt, t_eval=np.arange(t0, Tmax, dt),
                            )
            t_all.extend(sol.t)
            V_all.extend(sol.y[0])
            w_all.extend(sol.y[1])
            if sol.t_events[0].size > 0:
                V_all[-1] = model_params['Vpeak']
                t_spike = sol.t_events[0][0]
                spike_times.append(t_spike)
                # print(f"Spike at t = {t_spike:.2f} ms, V = {sol.y_events[0][0][0]:.2f}", flush=True)
                # Reset for next interval
                y0 = [model_params['Vreset'], sol.y[1,-1] + model_params['b']]
                t0 = t_spike
                # Manually add Vreset to trace for visualization
                t_all.append(t0)
                V_all.append(model_params['Vreset'])
                w_all.append(y0[1])
            else:
                break
    
    if plot:
        f, ax = plt.subplots(nrows=3, figsize = (9, 9), sharex='all')
        ax[0].plot(t_all, V_all, label='V')
        ax[0].set_ylabel('Membrane potential (mV)')

        ax[1].plot(t_all, w_all, label='w')
        ax[1].set_ylabel('Adaptation variable (w)')

        ax[2].plot(t_all, [Iapp(t) for t in t_all], label='I')
        ax[2].set_ylabel('Input current')

        ax[2].set_xlabel('Time (ms)')
        f.suptitle('aEIF neuron with event-based spiking')
        plt.tight_layout()
        plt.show()

    print(f"Total spikes: {len(spike_times)}", flush=True)