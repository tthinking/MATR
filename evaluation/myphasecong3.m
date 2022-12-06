

function [phaseCongruency, or, M, m]=myphasecong3(varargin)

       
[im, nscale, norient, minWaveLength, mult, sigmaOnf, ...
                  dThetaOnSigma,k, cutOff, g] = checkargs(varargin(:));     

v = version; Octave = v(1)<'5'; 
epsilon         = .0001;        

thetaSigma = pi/norient/dThetaOnSigma; 

[rows,cols] = size(im);
imagefft = fft2(im);           

zero = zeros(rows,cols);
totalEnergy = zero;             
totalSumAn  = zero;              
orientation = zero;              
                               
EO = cell(nscale, norient);                               
covx2 = zero;                   
covy2 = zero;
covxy = zero;

estMeanE2n = [];
ifftFilterArray = cell(1,nscale);

if mod(cols,2)
    xrange = [-(cols-1)/2:(cols-1)/2]/(cols-1);
else
    xrange = [-cols/2:(cols/2-1)]/cols;	
end

if mod(rows,2)
    yrange = [-(rows-1)/2:(rows-1)/2]/(rows-1);
else
    yrange = [-rows/2:(rows/2-1)]/rows;	
end

[x,y] = meshgrid(xrange, yrange);

radius = sqrt(x.^2 + y.^2);      
radius(floor(rows/2)+1,floor(cols/2)+1)=1;  
theta = atan2(-y,x);              
radius = ifftshift(radius);    
theta  = ifftshift(theta);      

sintheta = sin(theta);
costheta = cos(theta);
clear x; clear y; clear theta;    
lp = lowpassfilter([rows,cols],.45,15);  

logGabor = cell(1,nscale);

for s = 1:nscale
    wavelength = minWaveLength*mult^(s-1);
    fo = 1.0/wavelength;                  
    logGabor{s} = exp((-(log(radius/fo)).^2) / (2 * log(sigmaOnf)^2));  
    logGabor{s} = logGabor{s}.*lp;        
    logGabor{s}(1,1) = 0;                
end


spread = cell(1,norient);

for o = 1:norient
  angl = (o-1)*pi/norient;         
  ds = sintheta * cos(angl) - costheta * sin(angl);   
  dc = costheta * cos(angl) + sintheta * sin(angl);  
  dtheta = abs(atan2(ds,dc));                         
  spread{o} = exp((-dtheta.^2) / (2 * thetaSigma^2));  
                                                      
end


for o = 1:norient               

  if Octave fflush(1); end

  angl = (o-1)*pi/norient;           
  sumE_ThisOrient   = zero;         
  sumO_ThisOrient   = zero;       
  sumAn_ThisOrient  = zero;      
  Energy            = zero;      

  for s = 1:nscale,                 
    filter = logGabor{s} .* spread{o};   
                                        
        ifftFilt = real(ifft2(filter))*sqrt(rows*cols);  
        ifftFilterArray{s} = ifftFilt;                  
    EO{s,o} = ifft2(imagefft .* filter);      

    An = abs(EO{s,o});                        
    sumAn_ThisOrient = sumAn_ThisOrient + An;  
    sumE_ThisOrient = sumE_ThisOrient + real(EO{s,o}); 
    sumO_ThisOrient = sumO_ThisOrient + imag(EO{s,o});

    if s==1                                
      EM_n = sum(sum(filter.^2));          
      maxAn = An;                          
    else
      maxAn = max(maxAn, An);
    end

  end                                      
  XEnergy = sqrt(sumE_ThisOrient.^2 + sumO_ThisOrient.^2) + epsilon;   
  MeanE = sumE_ThisOrient ./ XEnergy; 
  MeanO = sumO_ThisOrient ./ XEnergy; 

  for s = 1:nscale,       
      E = real(EO{s,o}); O = imag(EO{s,o}); 
      Energy = Energy + E.*MeanE + O.*MeanO - abs(E.*MeanO - O.*MeanE);
  end

  medianE2n = median(reshape(abs(EO{1,o}).^2,1,rows*cols));
  meanE2n = -medianE2n/log(0.5);
  estMeanE2n(o) = meanE2n;

  noisePower = meanE2n/EM_n;                      

  EstSumAn2 = zero;
  for s = 1:nscale
    EstSumAn2 = EstSumAn2 + ifftFilterArray{s}.^2;
  end

  EstSumAiAj = zero;
  for si = 1:(nscale-1)
    for sj = (si+1):nscale
      EstSumAiAj = EstSumAiAj + ifftFilterArray{si}.*ifftFilterArray{sj};
    end
  end
  sumEstSumAn2 = sum(sum(EstSumAn2));
  sumEstSumAiAj = sum(sum(EstSumAiAj));


  EstNoiseEnergy2 = 2*noisePower*sumEstSumAn2 + 4*noisePower*sumEstSumAiAj;

  tau = sqrt(EstNoiseEnergy2/2);                    
  EstNoiseEnergy = tau*sqrt(pi/2);                  
  EstNoiseEnergySigma = sqrt( (2-pi/2)*tau^2 );

  T =  EstNoiseEnergy + k*EstNoiseEnergySigma;       

  T = T/1.7;       
  Energy = max(Energy - T, zero);        
 

  width = sumAn_ThisOrient ./ (maxAn + epsilon) / nscale;    


  weight = 1.0 ./ (1 + exp( (cutOff - width)*g)); 


  Energy_ThisOrient=weight.*Energy;
  totalSumAn=totalSumAn+sumAn_ThisOrient;
  totalEnergy=totalEnergy+Energy_ThisOrient;
  
  if (o==1),
      maxEnergy=Energy_ThisOrient;
  else
      change=Energy_ThisOrient>maxEnergy;
      orientation=(o-1).*change+orientation.*(~change);
      maxEnergy=max(maxEnergy, Energy_ThisOrient);
  end


  PC{o} = weight.*Energy./sumAn_ThisOrient; 
  featType{o} = E+i*O;

  covx = PC{o}*cos(angl);
  covy = PC{o}*sin(angl);
  covx2 = covx2 + covx.^2;
  covy2 = covy2 + covy.^2;
  covxy = covxy + covx.*covy;

end  
phaseCongruency=totalEnergy./(totalSumAn+epsilon);
orientation=orientation*(180/norient);



covx2 = covx2/(norient/2);
covy2 = covy2/(norient/2);
covxy = covxy/norient;  

denom = sqrt(covxy.^2 + (covx2-covy2).^2)+epsilon;
sin2theta = covxy./denom;
cos2theta = (covx2-covy2)./denom;
or = atan2(sin2theta,cos2theta)/2;    
or = round(or*180/pi);              
neg = or < 0;                                 
or = ~neg.*or + neg.*(or+180);        

M = (covy2+covx2 + denom)/2;         
m = (covy2+covx2 - denom)/2;        



    
function [im, nscale, norient, minWaveLength, mult, sigmaOnf, ...
          dThetaOnSigma,k, cutOff, g] = checkargs(arg); 

    nargs = length(arg);
    
    if nargs < 1
        error('No image supplied as an argument');
    end    
    
   
    im              = [];
    nscale          = 4;   
    norient         = 6;    
    minWaveLength   = 3;    
    mult            = 2.1;   
    sigmaOnf        = 0.55;  
                               
    dThetaOnSigma   = 1.2;  
    k               = 2.0;   
    cutOff          = 0.5;  
    g               = 10;                        
    
    allnumeric   = 1;      
    keywordvalue = 2;      
    readstate = allnumeric; 
    
    if readstate == allnumeric
        for n = 1:nargs
            if isa(arg{n},'char')
                readstate = keywordvalue;
                break;
            else
                if     n == 1, im            = arg{n}; 
                elseif n == 2, nscale        = arg{n};              
                elseif n == 3, norient       = arg{n};
                elseif n == 4, minWaveLength = arg{n};
                elseif n == 5, mult          = arg{n};
                elseif n == 6, sigmaOnf      = arg{n};
                elseif n == 7, dThetaOnSigma = arg{n};
                elseif n == 8, k             = arg{n};              
                elseif n == 9, cutOff        = arg{n}; 
                elseif n == 10,g             = arg{n};                                                    
                end
            end
        end
    end

    if readstate == keywordvalue
        while n < nargs
            
            if ~isa(arg{n},'char') | ~isa(arg{n+1}, 'double')
                error('There should be a parameter name - value pair');
            end
            
            if     strncmpi(arg{n},'im'      ,2), im =        arg{n+1};
            elseif strncmpi(arg{n},'nscale'  ,2), nscale =    arg{n+1};
            elseif strncmpi(arg{n},'norient' ,2), norient =   arg{n+1};
            elseif strncmpi(arg{n},'minWaveLength',2), minWavelength = arg{n+1};
            elseif strncmpi(arg{n},'mult'    ,2), mult =      arg{n+1};
            elseif strncmpi(arg{n},'sigmaOnf',2), sigmaOnf =  arg{n+1};
            elseif strncmpi(arg{n},'dthetaOnSigma',2), dThetaOnSigma =  arg{n+1};
            elseif strncmpi(arg{n},'k'       ,1), k =         arg{n+1};
            elseif strncmpi(arg{n},'cutOff'  ,2), cutOff   =  arg{n+1};
            elseif strncmpi(arg{n},'g'       ,1), g        =  arg{n+1};         
            else   error('Unrecognised parameter name');
            end

            n = n+2;
            if n == nargs
                error('Unmatched parameter name - value pair');
            end
            
        end
    end
    
    if isempty(im)
        error('No image argument supplied');
    end

    if ~isa(im, 'double')
        im = double(im);
    end
    
    if nscale < 1
        error('nscale must be an integer >= 1');
    end
    
    if norient < 1 
        error('norient must be an integer >= 1');
    end    

    if minWaveLength < 2
        error('It makes little sense to have a wavelength < 2');
    end          

    if cutOff < 0 | cutOff > 1
        error('Cut off value must be between 0 and 1');
    end
    

    



