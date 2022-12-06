function res=Q_CV(im1,im2,fim)

alpha_c=1;
alpha_s=0.685;
f_c=97.3227;
f_s=12.1653;

windowSize=16;

alpha=5;


im1=double(im1);
im2=double(im2);
fim=double(fim);


im1=normalize1(im1);
im2=normalize1(im2);
fim=normalize1(fim);

flt1=[-1 0 1 ; -2 0 2 ; -1 0 1];
flt2=[-1 -2 -1; 0 0 0; 1 2 1];


fuseX=filter2(flt1,fim,'same');
fuseY=filter2(flt2,fim,'same');
fuseG=sqrt(fuseX.*fuseX+fuseY.*fuseY);

buffer=(fuseX==0);
buffer=buffer*0.00001;
fuseX=fuseX+buffer;
fuseA=atan(fuseY./fuseX);


img1X=filter2(flt1,im1,'same');
img1Y=filter2(flt2,im1,'same');
im1G=sqrt(img1X.*img1X+img1Y.*img1Y);

buffer=(img1X==0);
buffer=buffer*0.00001;
img1X=img1X+buffer;
im1A=atan(img1Y./img1X);

img2X=filter2(flt1,im2,'same');
img2Y=filter2(flt2,im2,'same');
im2G=sqrt(img2X.*img2X+img2Y.*img2Y);

buffer=(img2X==0);
buffer=buffer*0.00001;
img2X=img2X+buffer;
im2A=atan(img2Y./img2X);

[hang,lie]=size(im1);
H=floor(hang/windowSize);
L=floor(lie/windowSize);

fun=@(x) sum(sum(x.^alpha)); 

ramda1=blkproc(im1G, [windowSize windowSize],[0 0],fun);
ramda2=blkproc(im2G, [windowSize windowSize],[0 0],fun);


f1=im1-fim;
f2=im2-fim;


[u,v]=freqspace([hang,lie],'meshgrid');

u=lie/8*u; v=hang/8*v;


r=sqrt(u.^2+v.^2);

theta_m=2.6*(0.0192+0.144*r).*exp(-(0.144*r).^1.1);

index=find(r==0);
r(index)=1;

buff=0.008./(r.^3)+1;

buff=buff.^(-0.2);
buff1=-0.3*r.*sqrt(1+0.06*exp(0.3*r));

theta_d=((buff).^(-0.2)).*(1.42*r.*exp(buff1));
theta_d(index)=0;
clear buff;
clear buff1;

theta_a=alpha_c*exp(-(r/f_c).^2)-alpha_s*exp(-(r/f_s).^2);

ff1=fft2(f1); 
ff2=fft2(f2);

Df1=ifft2(ifftshift(fftshift(ff1).*theta_m));
Df2=ifft2(ifftshift(fftshift(ff2).*theta_m));

fun2=@(x) mean2(x.^2);
D1=blkproc(Df1, [windowSize windowSize],[0 0],fun2);
D2=blkproc(Df2, [windowSize windowSize],[0 0],fun2);

Q=sum(sum(ramda1.*D1+ramda2.*D2))/sum(sum(ramda1+ramda2));

res=Q;
