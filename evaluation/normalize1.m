function RES=normalize1(data)


data=double(data);
da=max(data(:));
xiao=min(data(:));
if (da==0 & xiao==0)
    RES=data;
else
    newdata=(data-xiao)/(da-xiao);
    RES=round(newdata*255);
end


