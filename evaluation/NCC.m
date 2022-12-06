function res=NCC(im1,im2)


im1=double(im1);
im2=double(im2);

[hang,lie]=size(im1);
count=hang*lie;
N=256;
b=256;

h=zeros(N,N);

for i=1:hang
    for j=1:lie
       
        h(im1(i,j)+1,im2(i,j)+1)=h(im1(i,j)+1,im2(i,j)+1)+1;
    end
end

h=h./sum(h(:));

im1_marg=sum(h);  
im2_marg=sum(h'); 
H_x=-sum(im1_marg.*log2(im1_marg+(im1_marg==0)));
H_y=-sum(im2_marg.*log2(im2_marg+(im2_marg==0)));



H_xy=-sum(sum(h.*log2(h+(h==0))));
H_xy=H_xy/log2(b);


H_x=H_x/log2(b);
H_y=H_y/log2(b);

res=H_x+H_y-H_xy;
