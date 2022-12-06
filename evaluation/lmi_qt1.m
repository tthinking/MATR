function [decomp]=lmi_qt1(I)

[row,column]=size(I);
clength=max(floor(log2(row)),floor(log2(column))); 
mindim=4;
l=clength-mindim/2; 
decomp=cell(l,1);   
r=row;  
c=column;
x11=1;
y11=1;

dblock1=lmi_dblocks(I,x11,y11,r,c);
decomp{1}=dblock1;  
for level=1:l    

    sumen=zeros(1,length(decomp{level})/4);
    stopen=zeros(1,length(decomp{level})/4);
    
    for j=1:length(decomp{level})/4 
        for k=1:4            
            sumen(j)=sumen(j)+decomp{level}{(j-1)*4+k}(5); 
        end
        stopen(j)=sumen(j)/4;
    end

  
    dblock=cell(1,length(decomp{level}));
   indexdecomp =zeros(1,length(decomp{level}));  
    for k=1:length(decomp{level})/4
        for kk=1:4
            if decomp{level}{(k-1)*4+kk}(5)>=stopen(k)
                r=decomp{level}{(k-1)*4+kk}(3);   
                c=decomp{level}{(k-1)*4+kk}(4);   
                x11=decomp{level}{(k-1)*4+kk}(1);
                y11=decomp{level}{(k-1)*4+kk}(2);
                indexdecomp((k-1)*4+kk)=1;  
                dblock{(k-1)*4+kk}=lmi_dblocks(I,x11,y11,r,c);
            end
        end
    end

 
    ltemp=length(decomp{level})-sum(indexdecomp);
    tempd=cell(1,ltemp);
    mm=1;
    for m=1:length(decomp{level})
        if indexdecomp(m)==0
            tempd{level}(mm)=decomp{level}(m);
            mm=mm+1;
        end
    end
    decomp{level}=tempd{level};
  
    index=0;
    for ll=1:length(dblock)
        if ~isempty(dblock{ll})
            index=index+1;
        end
    end
    ddd=cell(1,index);
    jj=1;
    for ii=1:length(dblock)
        if ~isempty(dblock{ii})
            ddd{jj}=dblock{ii};
            jj=jj+1;
        end
    end
   
    for ii=1:length(ddd)
        for jj=1:4           
            decomp{level+1}{(ii-1)*4+jj}=ddd{ii}{jj};
        end
    end

end
    