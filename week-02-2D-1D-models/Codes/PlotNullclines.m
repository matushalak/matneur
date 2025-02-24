function fig = PlotNullclines(v,p)

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

  fig = figure; hold on;

  mInf = 0.5*( 1 + tanh((v-V1)/V2)); 
  nInf = 0.5*( 1 + tanh((v-V3)/V4)); 

  %% Values of n on the N-nullcline
  n = nInf;
  plot(v,n);

  %% Values of n on the V-nullcines
  ICa  = gCa*mInf.*(v-eCa);
  IL   = gL*(v-eL);
  n    = (-ICa -IL + IApp)./(gK*(v-eK));
  plot(v,n);

  ylabel('n'); xlabel('v'); ylim([0 0.6]); grid on;

end
