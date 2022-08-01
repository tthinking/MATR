
import torch
import torch.nn as nn

from torch.nn import functional as F
import numpy as np  

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False            
            self.oc = out_channels
            self.ks = kernel_size

            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws,ws))

            self.num_lat = int((kernel_size * kernel_size) / 2 + 1)

            self.ce = nn.Linear(ws*ws, self.num_lat, False)            
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            self.act = nn.ReLU(inplace=True)
            

            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels

            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            self.gd = nn.Linear(self.num_lat, kernel_size * kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size * kernel_size, False)

            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            self.sig = nn.Sigmoid()
    def forward(self, x):

        if self.ind:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()
            weight = self.weight

            gl = self.avg_pool(x).view(b,c,-1)

            out = self.ce(gl)

            ce2 = out
            out = self.ce_bn(out)
            out = self.act(out)

            out = self.gd(out)

            if self.g >3:

                oc = self.ci(self.act(self.ci_bn2(ce2).\
                                      view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
            else:

                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2,1))).transpose(2,1).contiguous() 
            oc = oc.view(b,self.oc,-1) 
            oc = self.ci_bn(oc)
            oc = self.act(oc)

            oc = self.gd2(oc)   

            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))

            x_un = self.unfold(x)
            b, _, l = x_un.size()

            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)

            return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))
            
