% Simulate oscillations for the inward rectifier with
% passive uptake for 

% Simulating the Kir-current
I0=-0.3;tau=500;
X0=[-66,.1];
opt=odeset('MaxStep',0.1);
[t,X]=ode45(@ODE_Kir,[0 2000],X0,opt,I0,tau);

%Plot timeseries
figure(1005);hold on;plot(t,X(:,1));xlim([0 2000]);
xlabel('t');ylabel('V');
%Phase plane view
figure(1006);clf(1006);hold on;
plot(X(:,1),X(:,2));%Timeseries
xlabel('V');ylabel('K_{out}');
xlim([-80 -60]);ylim([0.05 .15]);

function dXdt=ODE_Kir(t,X,I0,tau)
V=X(1);Kout=X(2);
gL=0.1;gK=0.1;alpha=0.2;K0=0.1;EL=-65;
minf=1/(1+exp((V+71)/.8));
EK=85*log(Kout)/log(10);
IK=gK*minf*(V-EK);
dXdt=[I0-gL*(V-EL)-IK;
    (alpha*IK+K0-Kout)/tau];
end