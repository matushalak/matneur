%Delayed Spiking in the Connor-Stevens model due to inactivating Potassium
%current.
V0=[-60,.01,.95,.05,.01,.9];Iapp=15;ga=34;
opt=odeset('MaxStep',0.1,'RelTol',1e-8);
[t,X]=ode45(@ConnorStevens_ODE,[0 200],V0,opt,Iapp,ga);
figure(9);clf(9);plot(t,X(:,1),'Linewidth',1);
V0=[-60,.01,.95,.05,.01,.9];Iapp=15;ga=50;
opt=odeset('MaxStep',0.1,'RelTol',1e-8);
[t,X]=ode45(@ConnorStevens_ODE,[0 200],V0,opt,Iapp,ga);
figure(9);hold on;plot(t,X(:,1),'Linewidth',1);
xlim([00 100]);xlabel('time [ms]');ylabel('V [mV]');
title('Full Connor-Stevens Model (Blue g_{A}=34, Red g_{A}=40)');

function dXdt=ConnorStevens_ODE(t,X,Iapp,ga)
V=X(1);m=X(2);h=X(3);n=X(4);a=X(5);b=X(6);
% i(t)=i0+i1*heav(t-ton)
% par i0,ga=47.7
gtotal=67.7;
gk=gtotal-ga;
ek=-72;
ena=55;
ea=-75;
el=-17;   
gna=120;
gl=0.3;  
ms=-5.3;
hs=-12;
ns=-4.3;  
% par ap=2  ton=100  i1=0 
%Hodgkin-Huxley with shifts - 3.8 is temperature factor
am=-.1*(V+35+ms)/(exp(-(V+35+ms)/10)-1);
bm=4*exp(-(V+60+ms)/18);
minf=am/(am+bm);
taum=1/(3.8*(am+bm));
ah=.07*exp(-(V+60+hs)/20);
bh=1/(1+exp(-(V+30+hs)/10));
hinf=ah/(ah+bh);
tauh=1/(3.8*(ah+bh));
an=-.01*(V+50+ns)/(exp(-(V+50+ns)/10)-1);
bn=.125*exp(-(V+60+ns)/80);
ninf=an/(an+bn);
% # Taun is doubled
taun=2/(3.8*(an+bn));
% # now the A current
ainf=(.0761*exp((V+94.22)/31.84)/(1+exp((V+1.17)/28.93)))^(.3333);
taua=.3632+1.158/(1+exp((V+55.96)/20.12));
binf=1/(1+exp((V+53.3)/14.54))^4;
taub=1.24+2.678/(1+exp((V+50)/16.027));
% # Finally the equations...
dXdt=[-gl*(V-el)-gna*(V-ena)*h*m*m*m-gk*(V-ek)*n*n*n*n-ga*(V-ea)*b*a*a*a+Iapp.*(t>40)
(minf-m)/taum; 
(hinf-h)/tauh; 
(ninf-n)/taun;
(ainf-a)/taua; 
(binf-b)/taub];
end