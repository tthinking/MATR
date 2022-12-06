function res=Q_P(im1,im2,fused)


fea_threshold=0.1;

im1=double(im1);
im2=double(im2);

    
[pc1,or1,M1,m1]=myphasecong3(im1);
clear or1;

[pc2,or2,M2,m2]=myphasecong3(im2);
clear or2;

[pcf,orf,Mf,mf]=myphasecong3(fused);
clear orf;

[hang,lie]=size(fused);

mask=(pc1>pc2);
pc_max=mask.*pc1+(~mask).*pc2;
M_max=mask.*M1+(~mask).*M2;
m_max=mask.*m1+(~mask).*m2;

mask1=(pc1>fea_threshold);
mask2=(pc2>fea_threshold);
mask3=(pc_max>fea_threshold);


resultPC=correlation_coeffcient(pc1,pc2,pc_max,pcf,mask1,mask2,mask3);
clear pc1;
clear pc2;
clear pc_max;
clear pcf;

resultM=correlation_coeffcient(M1,M2,M_max,Mf,mask1,mask2,mask3);
clear M1;
clear M2;
clear M_max;
clear Mf;

resultm=correlation_coeffcient(m1,m2,m_max,mf,mask1,mask2,mask3);
clear m1;
clear m2;
clear m_max;
clear mf;

[resultPC resultM resultm]';

res=resultPC*resultM*resultm;



function res=correlation_coeffcient(im1,im2,im_max,imf, mask1,mask2,mask3)

window=fspecial('gaussian',11,1.5);
window=window./(sum(window(:)));

C1=0.0001;
C2=0.0001;
C3=0.0001;


im1=mask1.*im1;
im2=mask2.*im2;
im_max=mask3.*im_max;

mu1=filter2(window,im1,'same');
mu2=filter2(window,im2,'same');
muf=filter2(window,imf,'same');
mu_max=filter2(window,im_max,'same');

mu1_sq=mu1.*mu1;
mu2_sq=mu2.*mu2;
muf_sq=muf.*muf;
mu_max_sq=mu_max.*mu_max;

mu1_muf=mu1.*muf;
mu2_muf=mu2.*muf;
mu_max_muf=mu_max.*muf;

sigma1_sq=filter2(window,im1.*im1,'same')-mu1_sq;
sigma2_sq=filter2(window,im2.*im2,'same')-mu2_sq;
sigmaMax_sq=filter2(window,im_max.*im_max,'same')-mu_max_sq;
sigmaf_sq=filter2(window,imf.*imf,'same')-muf_sq;

sigma1f=filter2(window,im1.*imf,'same')-mu1_muf;
sigma2f=filter2(window,im2.*imf,'same')-mu2_muf;
sigmaMaxf=filter2(window,im_max.*imf,'same')-mu_max_muf;

index1=find(mask1==1);
index2=find(mask2==1);
index3=find(mask3==1);

res1=mu1.*0;
res2=res1;
res3=res1;

res1(index1)=(sigma1f(index1)+C1)./(sqrt(abs(sigma1_sq(index1).*sigmaf_sq(index1)))+C1);
res2(index2)=(sigma2f(index2)+C2)./(sqrt(abs(sigma2_sq(index2).*sigmaf_sq(index2)))+C2);
res3(index3)=(sigmaMaxf(index3)+C3)./(sqrt(abs(sigmaMax_sq(index3).*sigmaf_sq(index3)))+C3);

buffer(:,:,1)=res1;
buffer(:,:,2)=res2;
buffer(:,:,3)=res3;

result=max(buffer,[],3);

A1=sum(mask1(:));
A2=sum(mask2(:));
A3=sum(mask3(:));

res=sum(result(:))/A3;






