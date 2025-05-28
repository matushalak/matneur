# @matushalak
import numpy as np
import numba as nb

###---Adaptive exponential and fire model definitions---
def nosynapse(t:float, y:tuple[float, float], 
            I:float, params:dict[str:float]):
    '''
    Brette & Gerstner 2005 adEx neuron WITHOUT synapses
    responding to input current
    '''
    V, w = y
    C = params['C']     # pF
    gL = params['gL']     # nS
    EL = params['EL']    # mV
    VT = params['VT']    # mV
    DeltaT = params['DeltaT']  # mV SLOPE
    tauw = params['tauw']  # ms
    a = params['a']       # nS

    dVdt = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + I) / C
    dwdt = (a * (V - EL) - w) / tauw
    return dVdt, dwdt

@nb.njit
def nosynapse_fast(t:float, y:tuple[float, float], I:float, 
                   params:tuple[float,...]):
    '''
    Brette & Gerstner 2005 adEx neuron WITHOUT synapses
    responding to input current
    '''
    V, w = y
    C, gL, EL, VT, DeltaT, tauw, a = params
    dVdt = (-gL * (V - EL) + gL * DeltaT * np.exp((V - VT)/DeltaT) - w + I) / C
    dwdt = (a * (V - EL) - w) / tauw
    return dVdt, dwdt

def twosynapse():
    pass


def nsynapses():
    pass