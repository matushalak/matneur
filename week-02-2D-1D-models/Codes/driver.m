% Clear workspace and close windows
clear all, close all, clc;

% Display a comet
showcomet = true;
showpplane = true;

% Parameters
p(1)  =    20;  % Cm
p(2)  =   4.4;  % gCa
p(3)  =     8;  % gK
p(4)  =     2;  % gL
p(5)  =   120;  % eCa
p(6)  =   -84;  % eK
p(7)  =   -60;  % eL
p(8)  =  0.04;  % phi
p(9)  =  -1.2;  % V1   
p(10) =    18;  % V2    
p(11) =     2;  % V3    
p(12) =    30;  % V4    
p(13) =    60;  % IApp  

% Plot nullclines
if showpplane || showcomet
  vRange = linspace(-60,40,1000);
  pplane = PlotNullclines(vRange,p);
end

% Right-hand side function
ml = @(t,u) MorrisLecar(t,u,p);
 
%% Time step, initial condition for a spike
u0 = [-19; 0.07]; 
tspan = [0 200];
[t,U] = ode45(ml,tspan,u0);
 
% Plot voltage
vplot = figure();
plot(t,U(:,1)); 
grid on; xlabel('t'); ylabel('v(t)');
 
% Comet plot
if showcomet
  pplaneComet = PlotNullclines(vRange,p);
  pause
  figure(pplaneComet), hold on;
  comet(U(:,1), U(:,2),0.6);
  hold off;
end
pause

%% Time step, initial condition does not lead to a spike
u0 = [-20; 0.1]; 
[t,U] = ode45(ml,tspan,u0);

figure(vplot), hold on; plot(t,U(:,1));

% Comet plot
if showcomet
  pplaneComet = PlotNullclines(vRange,p);
  pause
  figure(pplaneComet), hold on;
  comet(U(:,1), U(:,2),0.6);
  hold off;
end
pause

%% Time step, a periodic orbit is obtained when I app increases
p(13) =    100; 
ml = @(t,u) MorrisLecar(t,u,p); % required, as p has changed
tspan = [0 400];
[t,U] = ode45(ml,tspan,u0);

% Plot
if showpplane
  pplane2 = PlotNullclines(vRange,p);
end
figure; plot(t,U(:,1)); grid on; xlabel('t'); ylabel('v(t)');

% Comet plot
if showcomet
  pplaneComet = PlotNullclines(vRange,p);
  pause
  figure(pplaneComet), hold on;
  comet(U(:,1), U(:,2),0.6);
  hold off;
end
pause

%% Plot v versus dvdt
v = U(:,1); n = U(:,2);
dudt = MorrisLecarVectorised(t,[v;n],p); % takes a vector of values (v_i, n_i) and evaluates the ML's right-hand side
dvdt = dudt(1:length(v));
figure;

figure;
plot(v,dvdt,'*');
xlabel('v'); ylabel('dvdt'); grid on;
