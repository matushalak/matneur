V0=[-65,.8];Iapp=0;ga=40;
opt=odeset('MaxStep',0.1);
[t,X]=ode45(@ODE_ConnorStevens_Red,[0 1500],V0,opt,Iapp,ga);
figure(9);plot(t,X);xlim([20 3500]);
figure(10);hold on;plot(X(:,1),X(:,2));

function dXdt=ODE_ConnorStevens_Red(t,X,Iapp,ga)
%Reduced Connor-Stevens model as in Drion-O'Leary, Marder PNAS 2015
% 1. Sodium and A-type potassium activation are fast
%    Hence set to steady state
% 2. Keep potassium activation variable n, express
%    h and b using "equivalent potential".
V=X(1);n=X(2);
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
%Hodgkin-Huxley with shifts - 3.8 is temperature factor
am=-.1*(V+35+ms)/(exp(-(V+35+ms)/10)-1);
bm=4*exp(-(V+60+ms)/18);
minf=am/(am+bm);
% taum=1/(3.8*(am+bm));
% Compute equivalent potential by inverse
% Needed for reduction (V is argument in the paper, a bit confusing)
invn= @(V)-(200*log(1/V-21/20))/13-208/5;
Vn=invn(n);
ah=.07*exp(-(Vn+60+hs)/20);
bh=1/(1+exp(-(Vn+30+hs)/10));
h=ah/(ah+bh);
% hinf=ah/(ah+bh);
% tauh=1/(3.8*(ah+bh));
an=-.01*(V+50+ns)/(exp(-(V+50+ns)/10)-1);
bn=.125*exp(-(V+60+ns)/80);
% ninf=an/(an+bn);
ninf=1/(1.05+exp(-.065*(41.6+V)));
% # Taun is doubled
taun=2/(3.8*(an+bn));
% # now the A current
ainf=(.0761*exp((V+94.22)/31.84)/(1+exp((V+1.17)/28.93)))^(.3333);
% taua=.3632+1.158/(1+exp((V+55.96)/20.12));
b=1/(1+exp((Vn+53.3)/14.54))^4;
% binf=1/(1+exp((V+53.3)/14.54))^4;
% taub=1.24+2.678/(1+exp((V+50)/16.027));
% % Creating the reduction
m=minf;a=ainf;


% # Finally the equations...
dXdt=[-gl*(V-el)-gna*(V-ena)*h*m*m*m-gk*(V-ek)*n*n*n*n-ga*(V-ea)*b*a*a*a+Iapp.*(t>40)
% (minf-m)/taum; 
% (hinf-h)/tauh; 
(ninf-n)/taun];
% (ainf-a)/taua; 
% (binf-b)/taub];
end