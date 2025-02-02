from numpy import exp, zeros, array, ndarray

def HodgkinHuxley(
        t:float, # time
        # these are the (state determining) variables that are solved for
        variables:list[float],
        # these params determine the behavior of the model
        params:list[float],
        # function that determines the applied current I based on the timepoint (t)
        Ifunc:callable
        # returns du/dt with u containing state variables u = [v, n, m, h]
        ) -> ndarray:
    
    # v (voltage), n (activation gate Kv), m (activation gate Nav), h (inactivation gate Nav)
    v, n, m, h = variables
    # Cm (membrane capacitance), gNa (Na conductance), gK (K conductance), gL (leak conductance), 
    # eNa (Na Nernst), eK (K Nernst), eL (leak reversal potential), 
    # phi (temperature factor, influences transitions to different activation / inactivation states)
    Cm, gNa, gK, gL, eNa, eK, eL, phi = params

    # Calculate what the currents are from the state variables & parameters
    # IK = -gK * n^4(V-Ek)
    IK = - gK * (n**4) * (v - eK)
    # INa = -gNa * m^3 h(V-Ena)
    INa = - gNa * (m**3) * h * (v - eNa)
    # IL = -gL(V - El)
    IL = - gL * (v - eL)
    # Applied current at time t
    I_applied = Ifunc(t)

    # Activation & Inactivation functions  (parameters set by Hodkin & Huxley)
    # for the next timestep update in activation states 
    # ! based on current activation state (n /m /h) & voltage (v) !
    # in general these parameters are described by:
    # m = alpha_m(V)*(1-m) - beta_m(V)*m , same for n and h
    # Kv activation (n)
    alpha_n = 0.01 * (v + 55)/(1 - exp(-(v + 55)/10))
    beta_n  = 0.125 * exp(-(v + 65)/80)
    # Nav activation (m)
    alpha_m = 0.1 * (v + 40)/(1 - exp(-(v + 40)/10))
    beta_m  = 4 * exp(-(v + 65)/18)
    # Nav INactivation (h)
    alpha_h = 0.07 * exp(-(v + 65)/20)
    beta_h  = 1/(1 + exp(-(v + 35)/10))
    
    # du/dt
    # step in membrane voltage / step in time = Currents (t) / Membrane Capacitance
    dvdt = (IK + INa + IL + I_applied) / Cm
    # Changes in activation / inactivation states of channels -> this makes the difference, phi: temperature dependent!
    # Kv activation (n)
    dndt = phi * ((alpha_n * (1-n)) - (beta_n * n))
    # Nav activation (m)
    dmdt = phi * ((alpha_m * (1-m)) - (beta_m * m))
    # Nav INactivation (h)
    dhdt = phi * ((alpha_h * (1-h)) - (beta_h * h))

    # breakpoint()

    return array([dvdt,
                  dndt,
                  dmdt,
                  dhdt])