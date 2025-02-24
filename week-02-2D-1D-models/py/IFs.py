from numpy import ndarray, array

# 1D, Quasi-dynamical System, Not really a spiking model
def LIF(
        t:float, # time
        # these are the (state determining) variables that are solved for
        variables:list[float],
        # function that determines the applied current I based on the timepoint (t)
        Ifunc:callable,
        # these params determine the behavior of the model
        params:list[float] = [20, 2, - 60]
        ) -> ndarray: # returns du/dt with u containing state variables u 
    
    v = variables[0]
    # eL "resting"
    Cm, gL, eL = params

    IApp = Ifunc(t)
    IL = - gL * (v - eL)
    
    # TODO: LIF logic
    dvdt = (IApp + IL) / Cm

    # 1D
    return array([dvdt])

# 1D Quasi-Dynamical System, Spiking model
def QIF(
        t:float, # time
        # these are the (state determining) variables that are solved for
        variables:list[float],
        # function that determines the applied current I based on the timepoint (t)
        Ifunc:callable,
        # these params determine the behavior of the model
        params:list[float] = [20, 0.1, - 70, 20]
        ) -> ndarray: # returns du/dt with u containing state variables u 
    
    v = variables[0]
    # eL "resting"
    Cm, a, Vr, Vp = params  # Vr: resting potential, Vc: peak voltage

    IApp = Ifunc(t)

    # QIF dynamics: quadratic nonlinearity
    dvdt = (a * (v - Vr) * (v - Vp) + IApp) / Cm

    # 1D
    return array([dvdt])
