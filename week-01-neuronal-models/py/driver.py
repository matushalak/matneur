from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from HodgkinHuxley import HodgkinHuxley

# solve_ivp (Solve Initial Value Problem) same as ode45 in MATLAB
def main(ini_cond:list[float],
         params: list[float],
         duration:int):
    
    # Applied current function - start with no applied current
    # will be vectorized so get the applied current for every timestep straight away
    
    # Current Applied function I(t) =  ...
    # Current injection entirely determines response of the system
    # IAppliedF = lambda t: 25 if round(t) % 8 == 0 else 0 # regular current injections
    # IAppliedF = lambda t: 2 if round(t) in (16,17,18) else 0 # subthreshold injection, graded response
    # IAppliedF = lambda t: 10 if round(t) in (16,17,18) else 0 # suprathreshold injection - AP
    # IAppliedF = lambda t: 10 if round(t) < 110 and round(t) > 25 else 0 # supratheshold sustained
    # IAppliedF = lambda t: 50 if round(t) < 110 and round(t) > 25 else 0 # sustained varying intensity

    # Revisiting spike mechanism
    IAppliedF = lambda t: 25 if round(t) >= 9 and round(t) <= 11 else 0 # suprathreshold injection - AP
    # TODO: finish revisiting spike mechanism plots
    # TODO: window current plot y = Nav activation(V) & Nav inactivation(V) x = V


    # HH results - create a function that will take the time and current state and compute HH
    # Right-hand side of du/dt equation. as in: du/dt = f(*u, t)
    # by using this, we just turned it into a function with just t and u variables!
    hh = lambda t, u: HodgkinHuxley(t, variables=u, params=params, Ifunc= IAppliedF)
    
    # Timesteps ( interval of integration)
    t_interval = [0, duration]

    # Solution is an object with attributes
    # care mostly about solution.t (time points) & solution.y (4D for each variable that was solved for)
    solution = solve_ivp(fun = hh, t_span = t_interval, y0 = ini_cond)
    
    # RESULTS
    ts = solution.t
    voltage, n_act, m_act, h_act  = solution.y
    applied_current = [IAppliedF(tm) for tm in ts]
    
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize = (12, 8))
    
    # Voltage plot
    axes[0].plot(ts, voltage, color = 'k')
    axes[0].plot(ts, np.full(ts.shape, params[-4]), color = 'b', label = 'Ena')
    axes[0].plot(ts, np.full(ts.shape, params[-3]), color = 'r', label = 'Ek')
    axes[0].legend(loc = 4)
    axes[0].set_ylabel('Membrane Voltage [mV]')
    
    # Activation & Inactivation variables
    axes[1].plot(ts, n_act, label= 'Kv act (n)')
    axes[1].plot(ts, m_act, label = 'Nav act (m)')
    axes[1].plot(ts, h_act, label = 'Nav inact (h)')
    axes[1].set_ylabel('Activation variables')
    axes[1].legend(loc = 4)

    # Conductances as function of time gKt, gNat
    axes[2].plot(ts, ...)

    # Applied Current & IK & INa
    axes[-1].plot(ts, applied_current)
    axes[-1].set_ylabel('I_Applied [µA / cm^2]')
    axes[-1].set_xlabel('Time [ms]')
    
    plt.tight_layout()
    plt.savefig('HodgkinHuxley-Neuron.png', dpi = 200)
    plt.show()

# Command-line arguments to modify behavior of simulation
def parse_args():
    parser = ArgumentParser()

    # Model Parameters
    parser.add_argument('-cm', type=float, help= 'membrane capacitance', default= 1)
    parser.add_argument('-gna', type=float, help= 'sodium conductance',default= 120)
    parser.add_argument('-gk', type=float, help= 'potassium conductance',default= 36)
    parser.add_argument('-gl', type=float, help= 'leak conductance',default= 0.3)
    parser.add_argument('-ena', type=float, help= 'potassium reversal potential',default= 50)
    parser.add_argument('-ek', type=float, help= 'potassium reversal potential',default= - 77)
    parser.add_argument('-el', type=float, help= 'leak Reversal Potential',default= - 54.4)
    parser.add_argument('-phi', type=float, help= 'temperature factor (influences transition between activation (conformation) states of channels)',
                        default= 3**((20 - 6.3) / 10))
    
    # Initial Conditions
    parser.add_argument('-vm', type=float, help= 'membrane voltage', default= -60)
    parser.add_argument('-act_n', type=float, help= 'Kv activation gate (n)', default= 0)
    parser.add_argument('-act_m', type=float, help= 'Nav activation gate (m)', default= 0)
    parser.add_argument('-inact_h', type=float, help= 'Nav inactivation gate (h)', default= 0)

    # Duration of simulation (number of iterations & time-points)
    parser.add_argument('-time', type=int, help= 'Duration of simulation', default= 50)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    state = [args.vm, args.act_n, args.act_m, args.inact_h]
    model_params = [args.cm, args.gna, args.gk, args.gl, args.ena, args.ek, args.el, args.phi]

    main(ini_cond=state, params= model_params, duration=args.time)