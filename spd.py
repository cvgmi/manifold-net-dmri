import torch 
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import fm_ops as spd_ops
from batch_svd import batch_svd


def weightNormalize(weights):
    out = []
    for row in weights.view(weights.shape[0],-1):
         out.append(torch.clamp(row, min=0.001, max=0.999))
    return torch.stack(out).view(*weights.shape)



class SPDConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kern_size, stride):
        super(SPDConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kern_size = kern_size
        self.stride = stride
        self.weight_matrix = torch.nn.Parameter(torch.rand(out_channels, (kern_size**2)*in_channels),requires_grad=True)


    # x: [batches, channels, rows, cols, 3, 3]
    def forward(self, x):
        ####
        #out = x.view(-1,3,3)
        #for i in range(out.shape[0]):
        #    if torch.det(out[i]) <= 0:
        #        print(torch.eig(out[i]))
        ####

       # x: [batches, channels, rows, cols, 3, 3] -> 
        #    [batches, channels, 3, 3, rows, cols]
        x = x.permute(0,1,4,5,2,3).contiguous()

        # x_windows: [batches, channels, 3, 3, rows_reduced, cols_reduced, window_x, window_y]
        x_windows = x.unfold(4, self.kern_size, self.stride).contiguous()
        x_windows = x_windows.unfold(5, self.kern_size, self.stride).contiguous()

        x_s = x_windows.shape
        #x_windows: [batches, channels, 3, 3,  rows_reduced, cols_reduced, window]   
        x_windows = x_windows.view(x_s[0],x_s[1],x_s[2],x_s[3],x_s[4],x_s[5],-1)

        #x_windows: [batches, rows_reduced, cols_reduced, window, channels, 3,3]
        x_windows = x_windows.permute(0,4,5,6,1,2,3).contiguous()

        x_s = x_windows.shape
        x_windows = x_windows.view(x_s[0],x_s[1],x_s[2],-1,x_s[5],x_s[6]).contiguous()


        #Output format: [batches, sequence, out_channels, cov_x, cov_y]
        return spd_ops.recursiveFM2D(x_windows, weightNormalize(self.weight_matrix)), 0



class SPDLinear(nn.Module):
    def __init__(self):
        super(SPDLinear, self).__init__()
        self.A = torch.rand(2,288).cuda()

    #X: [-1, 3,3]
    #Y: [-1, 3,3]
    def GLmetric(self, X, Y):
        inner = torch.matmul(torch.inverse(X), Y)


        u,s,v = batch_svd(inner)
        s_log = torch.diag_embed(torch.log(s))
        log_term = torch.matmul(u,torch.matmul(s_log,v.permute(0,2,1)))
        dist = torch.sum(torch.diagonal(torch.matmul(log_term,log_term), dim1=-2, dim2=-1),1)
        return dist
    
    #x: [batch, channels, rows, cols, 3,3]
    def forward(self, x):
        x_s = x.shape

        #x: [batch*channels, rows*cols, 3,3]
        x = x.view(x.shape[0]*x.shape[1], -1, x.shape[4], x.shape[5])

        #x: [batch*channels, 1, 1, rows*cols, 3,3]
        x = x.unsqueeze(1).unsqueeze(2)


        #weights: [1,rows*cols-1]
        weights = (1.0/torch.arange(start=2.0,end=x.shape[3]+1)).unsqueeze(0).cuda()
        
        #unweightedFM: [batches*channels, 1,1,1, 3,3]
        unweighted_FM = spd_ops.recursiveFM2D(x,weights)


        #unweightedFM: [batches*channels,3,3]
        unweighted_FM = unweighted_FM.view(-1, x_s[4], x_s[5])
        
        #unweightedFM: [batches*channels,rows*cols,3,3]
        unweighted_FM = unweighted_FM.unsqueeze(1).repeat(1, x_s[2]*x_s[3], 1, 1)

        #unweightedFM: [batches*channels*rows*cols,3,3]
        unweighted_FM = unweighted_FM.view(-1, x_s[4], x_s[5])


        #x: [batches*channels,rows*cols,3,3]
        x = x.view(-1, x_s[2]*x_s[3], x_s[4], x_s[5])
        #x: [batches*channels*rows*cols,3,3]
        x = x.view(-1, x_s[4], x_s[5])

        out = self.GLmetric(x, unweighted_FM)

        #out: [batch, channels*rows*cols]
        out = out.view(x_s[0], x_s[1]*x_s[2]*x_s[3])


        return out

