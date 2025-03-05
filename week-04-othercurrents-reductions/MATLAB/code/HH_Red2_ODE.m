
close all
opt=odeset('MaxStep',0.1); % Needed for accuracy
X0=[-65,-65,-50,-50];  % Arbitrary initial condition
[t,X]=ode45(@HH_Red_ODE,[0 50],X0,opt,20);
figure;plot(t,X,'Linewidth',1);
legend('V','V_m','V_n','V_h');box on;
xlabel('Time [ms]','Fontsize',16);ylabel('V [mV]','Fontsize',16);
title('Timeseries Equivalent Potentials');
opt=odeset('MaxStep',0.1); % Needed for accuracy
X0=[-65,-50];  % Arbitrary initial condition
[t,X]=ode45(@HH_Red2D_ODE,[0 50],X0,opt,10,0);
figure;plot(t,X,'Linewidth',1);
xlabel('Time [ms]','Fontsize',16);ylabel('V [mV]','Fontsize',16);
legend('V','V_n'); box on;
title('Timeseries Equivalent Potentials -- Reduced & Slower');

%%
figure;hold on;plot(X(:,1),X(:,2),'Linewidth',1);
xlabel('V [mV]','Fontsize',16);ylabel('V_{n} [mV]','Fontsize',16);
%Nullcline Gate -- SIMPLE
VR=-76.1:50;Null_VN=-78+zeros(size(VR));Null_V1=-35+zeros(size(VR));Null_V2=-35+zeros(size(VR));
for i=2:length(VR)
  fun = @(x) HH_Red2D_ODE(0,[VR(i),x],0,2);
  [Null_VN(i),~,~]=fsolve(fun,Null_VN(i-1)+.01);
  fun = @(x) HH_Red2D_ODE(0,[VR(i),x],0,1);
  [Null_V1(i),~,~]=fsolve(fun,Null_V1(i-1)+.01);
  fun = @(x) HH_Red2D_ODE(0,[VR(i),x],10,1);
  [Null_V2(i),~,~]=fsolve(fun,Null_V2(i-1)+.01);
end
plot(VR,Null_V1,VR,Null_V2,VR,Null_VN,'Linewidth',1);
xlim([-80 50]);ylim([-70 -38]);
legend('orbit','V_{n}-nullcline','V-nullcline (I=0)','V-nullcline (I=10)')
title('Phase Plane Equivalent Potential');box on;
function dXdt=HH_Red_ODE(t,X,Iapp)
V=X(1);Vm=X(2);Vh=X(3);Vn=X(4);
E_Na=50;                                                                                                                                    
E_K=-77;                                                                                                                                    
E_L=-54.4;                                                                                                                                  
gNa=120;                                                                                                                                    
gK=36;                                                                                                                                      
gL=0.3;                                                                                                                                     
step=1e-5;                                                                                                                                  
am=@(VV) .1*(VV+40)/(1-exp(-(VV+40)/10));                                                                                                   
bm=@(VV) 4*exp(-(VV+65)/18);                                                                                                                
ah=@(VV) .07*exp(-(VV+65)/20);                                                                                                              
bh=@(VV) 1/(1+exp(-(VV+35)/10));                                                                                                            
an=@(VV) .01*(VV+55)/(1-exp(-(VV+55)/10));                                                                                                 
bn=@(VV) .125*exp(-(VV+65)/80);                                                                                                             
minf=@(VV) am(VV)/(am(VV)+bm(VV));                                                                                                          
hinf=@(VV) ah(VV)/(ah(VV)+bh(VV));                                                                                                          
ninf=@(VV) an(VV)/(an(VV)+bn(VV));                                                                                                          
derm_inf=(minf(V+step)-minf(V-step))/(2*step);
derh_inf=(hinf(V+step)-hinf(V-step))/(2*step);                                                                                              
dern_inf=(ninf(V+step)-ninf(V-step))/(2*step);
dXdt=[Iapp-gNa*minf(Vm)^3*hinf(Vh)*(V-E_Na)-gK*ninf(Vn)^4*(V-E_K)-gL*(V-E_L)                                                                  
(am(Vm)+bm(Vm))*(minf(V)-minf(Vm))/derm_inf;                                                                                           
(ah(Vh)+bh(Vh))*(hinf(V)-hinf(Vh))/derh_inf;
(an(Vn)+bn(Vn))*(ninf(V)-ninf(Vn))/dern_inf];
end

function dXdt=HH_Red2D_ODE(t,X,Iapp,jj)
% V=X(1);Vm=X(2);Vh=X(3);Vn=X(4);
V=X(1);Vm=V;Vn=X(2);Vh=Vn;
E_Na=50;                                                                                                                                    
E_K=-77;                                                                                                                                    
E_L=-54.4;                                                                                                                                  
gNa=120;                                                                                                                                    
gK=36;                                                                                                                                      
gL=0.3;                                                                                                                                     
step=1e-5;                                                                                                                                  
am=@(VV) .1*(VV+40)/(1-exp(-(VV+40)/10));                                                                                                   
bm=@(VV) 4*exp(-(VV+65)/18);                                                                                                                
ah=@(VV) .07*exp(-(VV+65)/20);                                                                                                              
bh=@(VV) 1/(1+exp(-(VV+35)/10));                                                                                                            
an=@(VV) .01*(VV+55)/(1-exp(-(VV+55)/10));                                                                                                 
bn=@(VV) .125*exp(-(VV+65)/80);                                                                                                             
minf=@(VV) am(VV)/(am(VV)+bm(VV));                                                                                                          
hinf=@(VV) ah(VV)/(ah(VV)+bh(VV));                                                                                                          
ninf=@(VV) an(VV)/(an(VV)+bn(VV));                                                                                                          
derm_inf=(minf(V+step)-minf(V-step))/(2*step);
derh_inf=(hinf(V+step)-hinf(V-step))/(2*step);                                                                                              
dern_inf=(ninf(V+step)-ninf(V-step))/(2*step);

dXdt=.6*[Iapp-gNa*minf(Vm)^3*hinf(Vh)*(V-E_Na)-gK*ninf(Vn)^4*(V-E_K)-gL*(V-E_L)                                                                  
% (am(Vm)+bm(Vm))*(minf(V)-minf(Vm))/derm_inf;                                                                                           
% (ah(Vh)+bh(Vh))*(hinf(V)-hinf(Vh))/derh_inf;
(an(Vn)+bn(Vn))*(ninf(V)-ninf(Vn))/dern_inf];
if jj>0 %For nullcline computation
   dXdt=dXdt(jj);
end
end

                                                    