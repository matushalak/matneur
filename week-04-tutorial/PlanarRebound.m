% Explore the effect of the T-type Calcium current
% on rebound action potentials
% Task A: Decrease I0 to de-inactivate T-current
% Task B: Vary EL to observe subthreshold oscillations

%Set main parameters
EL=-60;
I0=-.75;
Iapp=.5;dur=30;
% Preparing Nullclines
VV=-90:.1:20;
Vnull=Vcline(VV,EL,I0);
hnull=1./(1+exp((VV+83)/4));
% Simulating the T-current
X0=[-73,.04];
opt=odeset('MaxStep',0.1);
[t,X]=ode45(@ODE_PlanarRebound,[0 1200],X0,opt,EL,I0,Iapp,dur);

%Plot timeseries
figure(1000);plot(t,X);xlim([0 400]);
%Phase plane view
figure(1001);clf(1001);hold on;
plot(VV,Vnull,VV,hnull);%Nullclines
plot(X(:,1),X(:,2));%Timeseries
xlim([-90 20]);ylim([0 .5]);

function dXdt= ODE_PlanarRebound(t,X,EL,I0,Iapp,dur)
V=X(1);h=X(2);
F=96520;     %Faraday
R=8.3134*1e3;%Gas constant
T=273.15+25; %Body temparature, 25 Celsius in Ermentrout ODE-file
z=2; %valence of Ca^2+
xi=V*F*z/(R*T);
cao=2;cai=1e-4;
pcat=.15;
cfedrive=pcat*.002*F*xi*(cai-cao*exp(-xi))/(1-exp(-xi));
g_L=.05;
minf=1/(1+exp(-(V+59)/6.2));
hinf=1/(1+exp((V+83)/4));
tauh=22.7+.27/(exp((V+48)/4)+exp(-(V+407)/50));
I_CaT=minf^2*h*cfedrive;
% Modify Leak Reversal potential to -85,
% somewhere in between there are oscillations.
% EL=-60;
% Provide current pulse with some duration
Iapp= I0 + Iapp.*(t>20 && t<20+dur);
dXdt=[-g_L*(V-EL)-I_CaT+Iapp;
    (hinf-h)/tauh];
end

function h=Vcline(V,EL,I0)
F=96520;     %Faraday
R=8.3134*1e3;%Gas constant
T=273.15+25; %Body temparature, 25 Celsius in Ermentrout ODE-file
z=2; %valence of Ca^2+
xi=V*F*z/(R*T);
cao=2;cai=1e-4;
pcat=.15;
cfedrive=pcat*.002*F*xi.*(cai-cao*exp(-xi))./(1-exp(-xi));
g_L=.05;
minf=1./(1+exp(-(V+59)/6.2));
%Setting V'=0, we find the following expression for h:
h=(-g_L*(V-EL)+I0)./(minf.^2)./cfedrive;
end