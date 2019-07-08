import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from batch_svd import batch_svd

def weightnormalize(weights):
    out = []
    for row in weights:
         out.append(row**2/torch.sum(row**2))
    return torch.stack(out)


def mexp(x, exp):
    u,s,v = batch_svd(x)
    ep = torch.diag_embed(torch.pow(s,exp))
    v = torch.einsum('...ij->...ji', v)
    return torch.matmul(torch.matmul(u,ep),v)


#M: [-1,3,3] 
#N: [-1,3,3]
#w: [-1]
def batchGLMean(M,N,w):
    
    # w:[-1, 3, 3]
    w = w.unsqueeze(1).repeat(1,M.shape[-1])

    u,s,v = batch_svd(M)
    s_pow = torch.diag_embed(torch.pow(s,0.5))
    
    M_sqrt = torch.matmul(u, torch.matmul(s_pow, v.permute(0,2,1)))

    M_sqrt_inv = torch.inverse(M_sqrt)

    inner_term = torch.matmul(M_sqrt_inv, torch.matmul(N, M_sqrt_inv))

    
    u_i, s_i, v_i = batch_svd(inner_term)


    s_i_c = s_i.view(-1)
    s_i_c_pow = s_i_c**(w.view(-1))
    s_i_pow = s_i_c_pow.view(*s.shape)

    s_i_pow = torch.diag_embed(s_i_pow)

    inner_term_weighted = torch.matmul(u_i, torch.matmul(s_i_pow, v_i.permute(0,2,1)))


    return torch.matmul(M_sqrt, torch.matmul(inner_term_weighted, M_sqrt))


#windows: [batches, rows_reduced, cols_reduced, window, 3, 3]
#weights: [out_channels, in_channels*kern_size**2}
def recursiveFM2D(windows, weights):
    w_s = windows.shape

    # windows: [batches*rows_reduced*cols_reduced, window, 3, 3]
    windows = windows.view(-1, windows.shape[3], windows.shape[4], windows.shape[5])

    oc = weights.shape[0]

    # weights: [batches*rows_reduced*cols_reduced, out_channels, in_channels*kern_size**2]\
    weights = weights.unsqueeze(0).repeat(windows.shape[0],1,1)

    # weights: [batches*rows_reduced*cols_reduced*out_channels, in_channels*kern_size**2]\
    weights = weights.view(-1, weights.shape[2])

    # [batches*rows_reduced*cols_reduced*channels_out, 3,3]
    running_mean = windows[:,0,:,:].unsqueeze(1).repeat(1,oc,1,1)
    running_mean = running_mean.view(-1,running_mean.shape[2], running_mean.shape[3])


    for i in range(1,weights.shape[1]):
        current_fiber = windows[:,i,:,:]
        
        #[batches*rows_reduced*cols_reduced, channels_out, 3, 3]
        current_fiber = current_fiber.unsqueeze(1).repeat(1,oc,1,1)
        cf_s = current_fiber.shape
        
        # [batches*rows_reduced*cols_reduced*channels_out, 3, 3]
        current_fiber = current_fiber.view(-1, cf_s[2], cf_s[3])


        running_mean = batchGLMean(current_fiber, running_mean, weights[:,i])

    #out: [batches, rows_reduced, cols_reduced, channels_out, 3, 3]
    out = running_mean.view(w_s[0], w_s[1], w_s[2], oc, w_s[4], w_s[5])

    #out: [batches, channels_out, rows_reduced, cols_reduced, 3, 3]
    out = out.permute(0,3,1,2,4,5).contiguous()
    
    return out









