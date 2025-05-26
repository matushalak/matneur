import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters (from Brette & Gerstner 2005, Table 1, "regular spiking")
C = 281.0     # pF
gL = 30.0     # nS
EL = -70.6    # mV
VT = -50.4    # mV
DeltaT = 2.0  # mV
Vreset = EL # mV
Vpeak = -20.0 # mV
tauw = 144.0  # ms
a = 4.0       # nS
b = 0.0805    # nA

def I(t):
    return 2000*np.sin(0.005*t).__abs__()
    # return 500 if 50 < t < 250 else (0.0 if t < 500 else 700)

def aEIF(t, y):
    V, w = y
    dVdt = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + I(t)) / C
    dwdt = (a * (V - EL) - w) / tauw
    return [dVdt, dwdt]

def spike_event(t, y):
    return y[0] - Vpeak
spike_event.terminal = True
spike_event.direction = 1

Tmax = 1000
dt = 0.1
t_all = []
V_all = []
w_all = []
spike_times = []

t0 = 0.0
y0 = [EL, 0.0]
while t0 < Tmax:
    sol = solve_ivp(aEIF, (t0, Tmax), y0, events=spike_event, max_step=dt, t_eval=np.arange(t0, Tmax, dt))
    t_all.extend(sol.t)
    V_all.extend(sol.y[0])
    V_all[-1] = Vpeak
    w_all.extend(sol.y[1])
    if sol.t_events[0].size > 0:
        t_spike = sol.t_events[0][0]
        spike_times.append(t_spike)
        print(f"Spike at t = {t_spike:.2f} ms, V = {sol.y_events[0][0][0]:.2f}")
        # Reset for next interval
        y0 = [Vreset, sol.y[1,-1] + b]
        t0 = t_spike + dt
        # Manually add Vreset to trace for visualization
        t_all.append(t0)
        V_all.append(Vreset)
        w_all.append(sol.y[1,-1] + b)
    else:
        break

f, ax = plt.subplots(nrows=3, figsize = (9, 9))
ax[0].plot(t_all, V_all, label='V')
ax[0].set_ylabel('Membrane potential (mV)')

ax[1].plot(t_all, w_all, label='w')
ax[1].set_ylabel('Adaptation variable')

ax[2].plot(t_all, [I(t) for t in t_all], label='I')
ax[2].set_ylabel('Input current')

ax[2].set_xlabel('Time (ms)')
f.suptitle('aEIF neuron with event-based spiking')
plt.tight_layout()
plt.show()

print(f"Total spikes: {len(spike_times)}")