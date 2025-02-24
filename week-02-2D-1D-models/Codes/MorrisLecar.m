function dudt = MorrisLecar(t,u,p)
  % function dudt = MorrisLecar(t,u,p)
  % Computes the vectorfield of the Morris-Lecar model
  % The notation follows ET, Section 3.7.
  % Input varaibles: 
  %      t: time, expressed as a scalar
  %      u: the vector variables u = [v; w]
  %      p: the control parameters p = [a epsi gamma I]
  %      dudt: the vectorfield, with components ordered as in u

  %% Unpack parameters
  Cm   = p(1); 
  gCa  = p(2); 
  gK   = p(3); 
  gL   = p(4); 
  eCa  = p(5); 
  eK   = p(6); 
  eL   = p(7); 
  phi  = p(8); 
  V1   = p(9);
  V2   = p(10);
  V3   = p(11);
  V4   = p(12);
  IApp = p(13);

  %% Unpack state variables
  v = u(1);
  n = u(2);

  %% Activation and inactivation variables
  mInf = 0.5*( 1 + tanh((v-V1)/V2)); 
  nInf = 0.5*( 1 + tanh((v-V3)/V4)); 
  tau  = 1./( cosh((v-V3)/(2*V4)) );

  %% Currents
  ICa  = gCa*mInf*(v-eCa);
  IK   = gK*n*(v-eK);
  IL   = gL*(v-eL);

  %% Assemble the vectorfield 
  dudt = zeros(size(u));
  dudt(1) = (-ICa - IK - IL + IApp)/Cm;
  dudt(2) = phi/tau * (nInf - n);

end
