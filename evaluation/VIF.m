function VIF = VIF(img1, img2,imgf)
VIF1 =vifvec(img1,imgf);
VIF2 =vifvec(img2,imgf);
VIF=(VIF1+VIF2)./2;
end