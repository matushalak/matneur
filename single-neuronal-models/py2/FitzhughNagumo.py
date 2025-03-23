from numpy import ndarray, array

# 2D Dynamical System
def FitzhughNagumo(
        t:float, # time
        # these are the (state determining) variables that are solved for
        variables:list[float],
        # these params determine the behavior of the model
        params:list[float],
        # function that determines the applied current I based on the timepoint (t)
        Ifunc:callable
        ) -> ndarray: # returns du/dt with u containing state variables u 
    
    # TODO: Fitzhugh-Nagumo logic
    # w recovery (Kv)
    v, w = variables

    a, b, c = params

    IApp = Ifunc(t)

    dvdt = v * (a-v) * (v-1) - w + IApp
    dwdt = (b * v) - (c * w) # recovery variable
    
    # 2D Dynamical System
    return array([dvdt,
                  dwdt])