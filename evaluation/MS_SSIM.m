function MS_SSIM = MSSSIM(img1, img2,imgf)
MS_SSIM1 =msssim(img1,imgf);
MS_SSIM2 =msssim(img2,imgf);
MS_SSIM=(MS_SSIM1+MS_SSIM2)./2;
end