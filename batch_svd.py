import torch
import torch_batch_svd


class BatchSVDFunction(torch.autograd.Function):

    @staticmethod
    def forward(self, x):
        U0, S, V0 = torch_batch_svd.batch_svd_forward(x, True, 1e-7, 100)
        k = S.size(1)
        U = U0[:, :, :k]
        V = V0[:, :, :k]
        self.save_for_backward(x, U, S, V)

        return U, S, V

    @staticmethod
    def backward(self, grad_u, grad_s, grad_v):

        x, U, S, V = self.saved_variables

        grad_out = torch_batch_svd.batch_svd_backward(
            [grad_u, grad_s, grad_v],
            x, True, True, U, S, V
        )

        #if torch.isnan(grad_out).any():
        #    ind = torch.arange(grad_out.shape[0])[torch.sum(torch.sum(torch.isnan(grad_out),1),1)>0]
        #    for i in range(ind.shape[0]):
        #        print(S[ind[i]])
        #
        #    quit()

        return grad_out


def batch_svd(x):
    """
    input:
        x --- shape of [B, M, N]
    return:
        U, S, V = batch_svd(x) where x = USV^T
    """
    x_c = x.view(-1,x.shape[-1],x.shape[-1])
    u,s,v = BatchSVDFunction.apply(x_c)
    return u,s,v
