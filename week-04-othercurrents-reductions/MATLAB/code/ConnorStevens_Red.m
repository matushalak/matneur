V0=[-60,.6];Iapp=10;ga=40;
opt=odeset('MaxStep',0.1);
[t,X]=ode45(@ODE_ConnorStevens_Red,[0 200],V0,opt,Iapp,ga,0);
figure(9);hold on;plot(t,X(:,1));xlim([00 100]);
xlabel('time(ms)');ylabel('V [mV]');

%Nullcline Gate -- SIMPLE
VR=-81:60;Null_N=zeros(size(VR));
for i=2:length(VR)
  fun = @(x) ODE_ConnorStevens_Red(0,[VR(i),x],0,40,2);
  [Null_N(i),~,~]=fsolve(fun,Null_N(i-1)+.01);
end
%Nullcline Potential -- Three branches
NR=0.00:.01:1;Null_V1=-72+zeros(size(NR));
for i=2:length(NR)
  fun = @(x) ODE_ConnorStevens_Red(0,[x,NR(i)],0,40,1);
  [Null_V1(i),~,~]=fsolve(fun,Null_V1(i-1));
end
NR2=[0.00:.01:.68 .685:.001:.696];
Null_V2=23+zeros(size(NR2));
Null_V3=-34+zeros(size(NR2));
for i=2:length(NR2)
  fun = @(x) ODE_ConnorStevens_Red(0,[x,NR2(i)],0,40,1);
  [Null_V2(i),~,~]=fsolve(fun,Null_V2(i-1));
  [Null_V3(i),~,~]=fsolve(fun,Null_V3(i-1));
end
%% Case Iapp\neq 0
NR4=0.00:.01:.67;Null_V4=23+zeros(size(NR4));
for i=2:length(NR4)
  fun = @(x) ODE_ConnorStevens_Red(50,[x,NR(i)],Iapp,40,1);
  [Null_V4(i),~,~]=fsolve(fun,Null_V4(i-1));
end
VR5=-70:30;Null_V5=1+zeros(size(VR5));
for i=2:length(VR5)
  fun = @(x) ODE_ConnorStevens_Red(50,[VR5(i),x],Iapp,40,1);
  [Null_V5(i),~,~]=fsolve(fun,Null_V5(i-1));
end
VR6=-70:-30;Null_V6=0.001+zeros(size(VR6));
for i=2:length(VR6)
  fun = @(x) ODE_ConnorStevens_Red(50,[VR6(i),x],Iapp,40,1);
  [Null_V6(i),~,~]=fsolve(fun,Null_V6(i-1));
end
%%
figure(10);clf(10);hold on;plot(X(:,1),X(:,2),'Color','b','Linewidth',1);
plot(X([1 1550],1),X([1 1550],2),'Color','b','LineStyle','none','Marker','o','MarkerFaceColor',[.95,.95,.2]);
plot(VR,Null_N,'Color','r','Linewidth',1);
plot(Null_V1,NR,Null_V2,NR2,Null_V3,NR2,'Color','k','Linewidth',1);
plot(Null_V4,NR4,VR5,Null_V5,VR6,Null_V6,'Color','g','Linewidth',1);
xlabel('V [mV]','Fontsize',16);ylabel('n','Rotation',0,'Fontsize',16);xlim([-80 60]); ylim([0 1]);
box on;title('Phase Plane Reduced Connor-Stevens Model (I=0 black, I=6 green)')

function dXdt=ODE_ConnorStevens_Red(t,X,Iapp,ga,jj)
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
if jj>0 %For nullcline computation
   dXdt=dXdt(jj);
end
end