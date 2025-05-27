import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Literal
from time import time
import adEx_utils
import adEx_models as adEx

'''
Default parameters Brette & Gerstner (2005)
'C' : 281.0,     # pF
'gL' : 30.0,     # nS
'EL' : -70.6,    # mV
'VT' : -50.4,    # mV
'DeltaT' : 2.0,   # mV
'Vreset' : -70.6, # mV
'Vpeak' : 0, # mV - can't be more than 5 due to numerical overflow in scipy
Time constant of adaptation
    'tauw' : 144.0,  # ms
    How fast the is the evolution of the adaptation variable (w)
    Lower values make it faster, higher values make it slower
    at tauw = 1, there is no separation of timescales anymore
Subthreshold adaptation (regardless of spikes, purely to voltage changes)
    default 'a' : 4.0,       # nS 
    the lower this is, the more the adaptation variable (w) just decays to 0 after a spike
    the higher this is, the more the adaptation variable (w) responds to the input signal regardless of spikes as a "leak" current
Spike-triggered adaptation
    default 'b' : 80.5    # pA !!!
    how much adaptation variable (w) if offset after a spike
    higher values mean voltage must overcome that many more mV before next spike can be triggered
'''
### Basic model without synapses experiments
start = time()
# Custom input current
Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(Tmax = 720 * 47,
                                                                custom = lambda t: np.random.normal(loc = 625, scale = 500) 
                                                                if t % 500 < 250 else np.random.normal(loc = 550, scale = 500))
adEx_utils.run_experiment(adExModel=adEx.nosynapse, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp)

# # Default experiment - pulsed current (not in original paper)
Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(Tmax = 720*47) # returns the same as adEx.utils.default_experiment()
adEx_utils.run_experiment(adExModel=adEx.nosynapse, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp)

# Figure 1C - Voltage response to small and large current
Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(Vpeak = -20, figure = 'small_large')
adEx_utils.run_experiment(adExModel=adEx.nosynapse, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp)

# Figure 2C - Bursting Voltage response to small and large current when setting Vreset = -47
Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(Vreset = -47, figure = 'small_large')
adEx_utils.run_experiment(adExModel=adEx.nosynapse, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp)

# Figure 2D - Postinhibitory Rebound Voltage response to hyperpolarization when setting EL = -60, a = 80, tauw = 720
Tmax, dt, model_params, _, Iapp =  adEx_utils.define_experiment(EL = -60, Vreset = -60, a = 80, tau_w = 720, figure = 'hyperpol')
adEx_utils.run_experiment(adExModel=adEx.nosynapse, Tmax=Tmax, dt = dt, model_params=model_params, Iapp=Iapp)

print(time()-start)

# ### Model WITH synapses experiments