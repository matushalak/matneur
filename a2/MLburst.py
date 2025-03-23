import numpy as np

# TODO: find out reasonable parameters for V, n & Ca
def MLburst(t:float, vars:list[float, float, float], I:callable, 
            params : dict = {'gL':2,'EL':-60,
                             'gK':8,'EK':-84, 'V3':12, 'V4':17.4, 'phi':0.23,
                             'gCa':4,'ECa':120, 'V1':-1.2, 'V2':18,
                             'gKCa':0.25, 'epsilon':0.001, 'mu' : 0.02,
                             'Cm':20, 'Iapp':45
                             })->np.ndarray:
    '''
    3D model
    Morris-Lecar model extended with [Ca2+]-dependent Potassium channel to enable bursting behavior

    params for square wave burster in book by Ermentrout on page 115
    '''
    # unpack variables
    V, n, Ca = vars

    # gating variables
    minf = lambda v: (1 + np.tanh((v - params['V1']) / params['V2'])) / 2
    ninf = lambda v: (1 + np.tanh((v - params['V3']) / params['V4'])) / 2
    # time constant
    tau = lambda v : 1 / np.cosh((v - params['V3']) / 2*params['V4']) 

    # the various currents 
    Iapp = I(t) if 'Iapp' not in params else params['Iapp'] # by default use the default param as specified
    Ileak = params['gL'] * (V - params['EL'])
    IK = params['gK'] * n * (V - params['EK'])
    ICa = params['gCa'] * minf(V) * (V - params['ECa'])
    IK_Ca = params['gKCa'] * (Ca / (Ca + 1)) * (V - params['EK']) # calcium - concentration deprendent potassium current

    # changes in variables
    dvdt = (Iapp - Ileak - IK - ICa - IK_Ca) / params['Cm'] 
    dndt = params['phi'] * (ninf(V) - n) / tau(V)
    # 1 here is kCa which represents calcium removal rate, but it is not mentioned anywhere so just took dafault param == 1 from book
    dCadt = params['epsilon'] * ((-params['mu'] * ICa) - (1 * Ca))

    return np.array([dvdt, dndt, dCadt])

if __name__ == '__main__':
    pass