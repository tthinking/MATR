function res=Q_NCIE(im1,im2,fim)

im1=normalize1(im1);
im2=normalize1(im2);
fim=normalize1(fim);

[hang,lie]=size(im1);
b=256;
K=3;



NCCxy=NCC(im1,im2);


NCCxf=NCC(im1,fim);

NCCyf=NCC(im2,fim);


R=[ 1 NCCxy NCCxf; NCCxy 1 NCCyf; NCCxf NCCyf 1];
r=eig(R);


HR=sum(r.*log2(r./K)/K);
HR=-HR/log2(b);


NCIE=1-HR;

res=NCIE;