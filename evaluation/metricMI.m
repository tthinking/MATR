function res=metricMI(im1,im2,fim,sw)



im1=normalize1(im1);
im2=normalize1(im2);
fim=normalize1(fim);

if nargin==3
    sw=1;
end

switch sw
    case 1
        % Q_MI
        [I_fx,H_xf,H_x,H_f1]=mutual_info(im1,fim);
        [I_fy,H_yf,H_y,H_f2]=mutual_info(im2,fim);
        
        MI=2*(I_fx/(H_f1+H_x)+I_fy/(H_f2+H_y));
        res=MI;
    case 2
        q=1.85;    
        I_fx=tsallis(im1,fim,q);
        I_fy=tsallis(im2,fim,q);
        res=I_fx+I_fy;
        
    case 3
        %  Q_TE
        q=0.43137;
        
        I_fx=tsallis(im1,fim,q);
        I_fy=tsallis(im2,fim,q);
        I_xy=tsallis(im1,im2,q);        
       
        [M_xy,H_xy,H_x,H_y]=mutual_info(im1,im2);

        MI=(I_fx+I_fy)/(H_x.^q+H_y.^q+I_xy);
        res=MI;
    otherwise
        error('Your input is wrong! Please check help file.');
end

