%% Simulating the Sag current with inward rectifier;
X0=[-68,.24];
opt=odeset('MaxStep',.1);
[t,X]=ode45(@ODE_Sag,[0 2500],X0,opt,0);

figure(98);clf(98);hold on;
plot(t,X(:,1),'Linewidth',1);box on;
xlabel('time [ms]','Fontsize',16);
ylabel('V [mV]','Fontsize',16);
figure(99);clf(99);plot(X(:,1),X(:,2),'Linewidth',1);
hold on;V=-75:.5:-68;plot(V,1./(1+exp((V+75)/5.5)),'Linewidth',1,'Color','k');
% V_nullcline for I=0 and I=-1 (t>1000)
Null_V1=0.2+zeros(size(V));Null_V2=Null_V1;
for i=2:length(V)
  fun = @(x) ODE_Sag(0,[V(i),x],1);
  [Null_V1(i),~,~]=fsolve(fun,Null_V1(i-1));
  fun = @(x) ODE_Sag(1600,[V(i),x],1);
  [Null_V2(i),~,~]=fsolve(fun,Null_V2(i-1));
end
plot(V,Null_V1,V,Null_V2,'Linewidth',1);


xlabel('V [mV]','Fontsize',16);xlim([-74 -68]);
ylabel('y','Fontsize',16);ylim([0.2 0.4]);

% # sag + inward rectifier
% #
% par i=0
% par gl=.025,el=-70
% # sag 
% # migliore tau0=46,vm=-80,b=23
% # migliore vt=-81,k=8
% # mccormick tau0=1000,vm=-80,b=13.5
% #
function dXdt = ODE_Sag(t,X,jj)
v=X(1);y=X(2);
k=5.5;vt=-75;
ek=-85;gk=1;
va=-80;vb=5;
gl=0.25;el=-70;
gh=0.25;eh=-43;
tau0=1000;vm=-80;b=13.5;
%Sag
yinf=1/(1+exp((v-vt)/k));
ty=tau0/cosh((v-vm)/b);
ih=gh*(v-eh)*y;
%kir
minf=1/(1+exp((v-va)/vb));
ikir=gk*minf*(v-ek);
Iapp=0 -0.5*(t>1000);
dXdt=[Iapp-gl*(v-el)-ih-ikir; 
      (yinf-y)/ty];
if jj>0 %For nullcline computation
   dXdt=dXdt(jj);
end
end
% init v=-68
% init y=.24
% @ total=1000,meth=qualrk,dt=.25
% @ xp=v,yp=y,xlo=-90,xhi=-40,ylo=0,yhi=0.6
% @ nmesh=100
% done