function vif=vifvec(imorg,imdist);

addpath(genpath('steerable_pyramid-master'));
M=3;
subbands=[4 7 10 13 16 19 22 25];
sigma_nsq=0.4;

[pyr,pind] = buildSpyr(imorg, 4, 'sp5Filters', 'reflect1'); 
org=ind2wtree(pyr,pind); 
[pyr,pind] = buildSpyr(imdist, 4, 'sp5Filters', 'reflect1');
dist=ind2wtree(pyr,pind);

[g_all,vv_all]=vifsub_est_M(org,dist,subbands,M);

[ssarr, larr, cuarr]=refparams_vecgsm(org,subbands,M);

vvtemp=cell(1,max(subbands));
ggtemp=vvtemp;
for(kk=1:length(subbands))
    vvtemp{subbands(kk)}=vv_all{kk};
    ggtemp{subbands(kk)}=g_all{kk};
end


for i=1:length(subbands)
    sub=subbands(i);
    g=ggtemp{sub};
    vv=vvtemp{sub};
    ss=ssarr{sub};
    lambda = larr(sub,:);, 
    cu=cuarr{sub};

    neigvals=length(lambda);
    
    lev=ceil((sub-1)/6);
    winsize=2^lev+1; offset=(winsize-1)/2;
    offset=ceil(offset/M);
    
    g=g(offset+1:end-offset,offset+1:end-offset);
    vv=vv(offset+1:end-offset,offset+1:end-offset);
    ss=ss(offset+1:end-offset,offset+1:end-offset);
    
    
    temp1=0; temp2=0;
    for j=1:length(lambda)
        temp1=temp1+sum(sum((log2(1+g.*g.*ss.*lambda(j)./(vv+sigma_nsq))))); 
        temp2=temp2+sum(sum((log2(1+ss.*lambda(j)./(sigma_nsq))))); 
    end
    num(i)=temp1;
    den(i)=temp2;
    
end

vif=sum(num)./sum(den);
