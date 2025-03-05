%  Destexe \& Pare model
%  J. Neurophys 1999 

X0=[-73.87,0,1,.002,.007];
[t,X]=ode45(@ODE_Mcur,[0 1500],X0);
V=X(:,1);dv=diff(V);
[ind,~]=find(dv(1:end-1).*dv(2:end)<0);
ind=ind(1:2:end);
figure(1);clf(1);plot(t,X(:,1));hold on;
plot(t(ind),X(ind,1), 'Marker','o','Linestyle','none');
dt=diff(t(ind));
figure(2);plot(1000./dt,'Linewidth',1);
xlabel('#Spike','Fontsize',16);ylabel('Instantaneous Frequency [Hz]','Fontsize',16);
figure(3);clf(3);plot(t,X(:,5));title('Adaptation building up');


function dXdt = ODE_Mcur(t,X)
v=X(1);m=X(2);h=X(3);n=X(4);mk=X(5);
gkm=2;cm=1;
gna=120;ena=55;
gk=100;ek=-85;Iapp=6;
gl=.019;el=-65;
% # shifted to acct for threshold
vt=-58;vs=-10;
%  sodium
am=-.32*(v-vt-13)/(exp(-(v-vt-13)/4)-1);
bm=.28*(v-vt-40)/(exp((v-vt-40)/5)-1);
ah=.128*exp(-(v-vt-vs-17)/18);
bh=4/(1+exp(-(v-vt-vs-40)/5));
ina=gna*m^3*h*(v-ena);
% delayed rectifier
an=-.032*(v-vt-15)/(exp(-(v-vt-15)/5)-1);
bn=.5*exp(-(v-vt-10)/40);
ikdr=gk*n^4*(v-ek);
% slow potassium M-current
akm=.0001*(v+30)/(1-exp(-(v+30)/9));
bkm=-.0001*(v+30)/(1-exp((v+30)/9));
ikm=gkm*mk*(v-ek);
dXdt=[(Iapp-gl*(v-el)-ikdr-ina-ikm)/cm;
   am*(1-m)-bm*m;
   ah*(1-h)-bh*h;
   an*(1-n)-bn*n;
   akm*(1-mk)-bkm*mk];
% # numerics stuff
% @ total=1000,dt=.25,meth=qualrk,xhi=1000,maxstor=10000
% @ bound=1000,ylo=-85,yhi=-50
% done

end