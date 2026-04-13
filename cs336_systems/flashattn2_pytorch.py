import torch

class FlassAttn2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K, Q, V, is_causal: bool = False):

        #...
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
    
# O = FlashAttn2.apply(Q, K, V
    
