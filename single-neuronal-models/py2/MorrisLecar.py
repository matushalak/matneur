from numpy import ndarray, array, tanh, cosh

# 2D Dynamical System
def MorrisLecar(
        t:float, # time
        # these are the (state determining) variables that are solved for
        variables:list[float],
        # function that determines the applied current I based on the timepoint (t)
        Ifunc:callable,
        # these params determine the behavior of the model
        params:list[float] = [20, 2, -60, 8, -84, 4, 120, -1.2, 18, 12, 17.4]
        ) -> ndarray: # returns du/dt with u containing state variables u 
    
    # TODO: MorrisLecar model logic
    # voltage, recovery variable
    v, n = variables

    Cm, gL, eL, gK, eK, gCa, eCa, v1, v2, v3, v4 = params

    # v1 & v3 control location of curve (left-right), v2 & v4 control steepness / width / shape
    minf = (1+tanh((v-v1)/v2))/2                         
    ninf = (1+tanh((v-v3)/v4))/2              
    tau = 1/cosh((v-v3)/(2*v4))                         

    # Currents:
    # Leak
    IL = - gL*(v - eL)
    # recovery (potassium)
    IK = - gK*n*(v - eK)
    # active (sodium / calcium in morris lecar)
    ICa = - gCa*minf*(v - eCa)
    # applied current
    IApp = Ifunc(t)

    dvdt = (IApp + IL + IK + ICa) / Cm
    dndt = (ninf - n) / tau

    # 2D Dynamical System
    return array([dvdt,
                  dndt])