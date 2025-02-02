function dudt = HodgkinHuxley(t,u,p,IAppFun)
  % function dudt = HodgkinHuxley(t,u,p,IAppFun)
  % Computes the vectorfield of the Hodgkin-Huxley model
  % The notation follows ET, Chapter 1.10.
  % Input varaibles: 
  %      t: time, expressed as a scalar
  %      u: the vector variables u = [v; n; m; h]
  %      p: the control parameters p = [cm gNa gK gL eNa eK eL phi]
  %      IAppFun: a function handle that specifies the applied input current as a function of time
  % Output varaibles: 
  %      dudt: the vectorfield, with components ordered as in u

  %% Unpack parameters
  Cm   = p(1); 
  gNa  = p(2); 
  gK   = p(3); 
  gL   = p(4); 
  eNa  = p(5); 
  eK   = p(6); 
  eL   = p(7); 
  phi  = p(8); 

  %% Unpack state variables
  v = u(1);
  n = u(2);
  m = u(3);
  h = u(4);

  %% Ativation and inactivation functions
  alphaN = 0.01*(v+55)/(1-exp(-(v+55)/10));
  betaN  = 0.125*exp(-(v+65)/80);
  alphaM = 0.1*(v+40)/(1-exp(-(v+40)/10));
  betaM  = 4*exp(-(v+65)/18);
  alphaH = 0.07*exp(-(v+65)/20);
  betaH  = 1/(1+exp(-(v+35)/10));

  %% Compute currents
  INa  = -gNa*m^3*h*(v-eNa);
  IK   = -gK*n^4*(v-eK);
  IL   = -gL*(v-eL);
  IApp =  IAppFun(t);

  %% Assemble the vectorfield 
  dudt = zeros(4,1);
  dudt(1) = (INa + IK + IL + IApp)/Cm;
  dudt(2) = phi*(alphaN*(1-n) - betaN*n);
  dudt(3) = phi*(alphaM*(1-m) - betaM*m);
  dudt(4) = phi*(alphaH*(1-h) - betaH*h);

end
