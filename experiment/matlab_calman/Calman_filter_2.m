N=100; 
a = 0.15; % ускорение 
T = -0.5; 
sigmaPsi=1; % знач. для расчета погрешности модели 
sigmaEta=4; % знач. для расчета погрешности датчика/сенсора 
k=1:N; 
x=k; 
x(1)=0; 
z(1)=x(1)+normrnd(0,sigmaEta); % нач. значение с датчика + его погрешность 
for t=1:(N-1) 
x(t+1)=x(t)+a*T*t+normrnd(0,sigmaPsi); % уравнение изменения координаты + погрешность модели 
z(t+1)=x(t+1)+normrnd(0,sigmaEta); 
end; 
%Фильтр Калмана 
xOpt(1)=z(1); 
eOpt(1)=sigmaEta; 
msum = eOpt(1); 
sum = (eOpt(1))^2; 
for t=1:(N-1) 
eOpt(t+1)=sqrt((sigmaEta^2)*(eOpt(t)^2+sigmaPsi^2)/(sigmaEta^2+eOpt(t)^2+sigmaPsi^2)); % среднее знач. квадрата ошибки 
msum = msum + eOpt(t+1); 
sum = sum + (eOpt(1))^2; 
K(t+1)=(eOpt(t+1))^2/sigmaEta^2; % усиление Калмана 
xOpt(t+1)=(xOpt(t)+a*T*t)*(1-K(t+1))+K(t+1)*z(t+1); % оптимальное отфильтрованное значение 
end; 
figure;%%%%%%%%%%%%%
sr = msum/N; 
vdisp = sum/N - sr^2; 
hold all; 
plot(k,xOpt); 
plot(k,z); 
plot(k,x,'--','Color',[.1 .0 .0],'LineWidth',1); 
title('Результаты фильтрации'); 
xlabel('время (sec)'); 
ylabel('координата (m)'); 
legend('значения с фильтра Калмана','показания датчика','исходные данные');
%%%%%%%%%%%%%%%Вычисление погрешности (SKO)отфильтрованного сигнала
for i=1:(N-1)%%%%%%%%%%%
err(i)=(xOpt(i)-z(i)); 
end;
DZ=err*100./(max(z)-min(z));
SKO=std(DZ)
figure;
i=1:(N-1);
plot(i,DZ(i));%%%%%%%%%%%%%%%
title('Погрешность фильтрации,%'); 
 
pause;
close all;
clear;
clc;
