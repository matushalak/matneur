import numpy as np

# TODO: find out reasonable parameters for V & Ca
def Ltype(t:float, vars:list[float], I:callable, 
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
    alpha = lambda v: 0.055 * ((-27.01 - v) / (np.exp((-27.01 - v) / 3.8) - 1))
    beta = lambda v: 0.94 * np.exp((-63.01 - v) / 17)
    m = lambda v: alpha(v) / (alpha(v) + beta(v)) # activation only depends on voltage
    h = lambda Ca: params['ki'] / (params['ki'] + Ca) # inactivation only depends on Ca2+ concentration 

    # currents
    xi = lambda v: params['z'] * params['F'] * v / (params['R'] * params['T'])
    Idrive = lambda v, Ca: params['Pmax'] * params['z'] * params['F'] * xi(V) * ((Ca - params['Caout'] * np.exp(-xi(V))) / (1 - np.exp(-xi(V))))
    
    Ileak = params['gL'] * (V - params['EL'])
    ICaL = m(V) * h(Ca) * Idrive(V, Ca)

    # changes in variables
    dvdt = I(t) - Ileak - ICaL
    dCadt = -params['Beta'] * ICaL - ((Ca - params['Cainf']) / params['tauCa'])

    return np.array([dvdt, dCadt])

if __name__ == '__main__':
    pass