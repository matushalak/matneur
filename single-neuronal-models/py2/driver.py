import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from argparse import ArgumentParser

from FitzhughNagumo import FitzhughNagumo
from IFs import LIF, QIF
from MorrisLecar import MorrisLecar

def main(mod:str, duration:int,
         params:list[float]):
    
    modelmap = {'fitz':FitzhughNagumo,
                'ml':MorrisLecar,
                'lif':LIF,
                'qif':QIF}
    
    inicondmap = {'fitz': [-70,0],
                'ml': [-70,0],
                'lif':[-70],
                'qif':[-70]}

    Model = modelmap[mod]
    
    # Applied current function - start with no applied current
    # will be vectorized so get the applied current for every timestep straight away
    
    # Current Applied function I(t) =  ...
    # Current injection entirely determines response of the system
    IAppliedF = lambda t: 25 if round(t) % 8 == 0 else 0 # regular current injections
    # IAppliedF = lambda t: 2 if round(t) in (16,17,18) else 0 # subthreshold injection, graded response
    # IAppliedF = lambda t: 10 if round(t) in (16,17,18) else 0 # suprathreshold injection - AP
    # IAppliedF = lambda t: 100 if round(t) < 110 and round(t) > 25 else 0 # supratheshold sustained
    # IAppliedF = lambda t: 50 if round(t) < 110 and round(t) > 25 else 0 # sustained varying intensity

    # Revisiting spike mechanism
    # IAppliedF = lambda t: 25 if round(t) >= 9 and round(t) <= 11 else 0 # suprathreshold injection - AP
    if mod == 'fitz':
        model_f = lambda t, u: Model(t, variables = u, Ifunc = IAppliedF, params = params)
    else:
        model_f = lambda t, u: Model(t, variables = u, Ifunc = IAppliedF)

    # Timesteps ( interval of integration)
    t_interval = [0, duration]

    # Solution is an object with attributes
    # care mostly about solution.t (time points) & solution.y (+1D for each variable that was solved for)
    if mod == 'lif':
        # LIF logic - vthresh, vreset, vrest
        def lif_spike(t, v):
            return v[0] - 25
        
        lif_spike.terminal = True
        lif_spike.direction = - 55

        solution = solve_ivp(fun = model_f, t_span = t_interval, y0 = inicondmap[mod],
                             events=lif_spike)
    
    elif  mod == 'qif':
        # QIF logic - vpeak, vreset, vrest
        def qif_spike(t, v):
            return v[0] - 90
        
        qif_spike.terminal = True
        qif_spike.direction = 20
        solution = solve_ivp(fun = model_f, t_span = t_interval, y0 = inicondmap[mod],
                             events = qif_spike)
    else:
        solution = solve_ivp(fun = model_f, t_span = t_interval, y0 = inicondmap[mod])

    ts = solution.t
    applied_current = [IAppliedF(tm) for tm in ts]
    if mod in ('fitz', 'ml'):
        voltage, recovery = solution.y
    else:
        voltage = solution.y[0,:]

    ### PLOTS
    fig, axes = plt.subplots(nrows=2, ncols=1)
    # 1) Voltage plot
    axes[0].plot(ts, voltage, color = 'k')
    axes[0].legend(loc = 4)
    axes[0].set_ylabel('Membrane Voltage [mV]')
    
    plt.tight_layout()
    plt.show()

# Command-line arguments to modify behavior of simulation
def parse_args():
    parser = ArgumentParser()

    # Duration of simulation (number of iterations & time-points)
    parser.add_argument('-model', type=str, help= 'Which model to investigate fitz | ml | lif | qif', choices=['fitz', 'ml', 'lif', 'qif'])
    parser.add_argument('-time', type=int, help= 'Duration of simulation', default= 140)
    parser.add_argument('-fitzparams', type = list, nargs='+', help = 'a b c', default = [-0.1, 0.01, 0.02])

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    main(mod = args.model, duration = args.time, params = args.fitzparams)