function [qtmi,et]=LMI(I1,I2,F,alpha)

tic;
[decomp1]=lmi_qt1(I1);
[decomp2]=lmi_qt1(I2);
[decompf]=lmi_qt1(F);
[miafd,miaf]=lmi_mi(I1,F,decomp1,alpha);
[mifad,mifa]=lmi_mi(F,I1,decompf,alpha);
[mibfd,mibf]=lmi_mi(I2,F,decomp2,alpha);
[mifbd,mifb]=lmi_mi(F,I2,decompf,alpha);
qtmi=(miaf+mifa+mibf+mifb)/2;
et=toc;