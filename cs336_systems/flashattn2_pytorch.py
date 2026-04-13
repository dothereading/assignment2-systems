import torch


B_c = 16
B_r = 16

class FlassAttn2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        
        _, N, d = K.size() # N is seq_len, d is dimensions

        T_r = (N + B_r - 1) // B_r # formula for cieling division
        T_c  = (N + B_c - 1) // B_c

        Q_list = torch.split(Q, T_r, dim=2)
        K_list = torch.split(K, T_c, dim=2)
        V_list = torch.split(V, T_c, dim=2)

        for i in range(T_r):
            Q_i = Q_list[i]

            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(B_r)
            m_i = torch.full((1, B_r), -float('inf'))

            for j in range(T_c):
                K_j = K_list[j]
                V_j = V_list[j]

                S_ij = Q_i @ K_j.T
                m_ij = torch.max(m_i[j-1 if j>0 else 0], max(S_ij))
                P_i = #bla
                l_i = #bla
                O_i = #bla

                # end for...
            # end for...


        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return torch.cat(O_i, dim=2)

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
    
# O = FlashAttn2.apply(Q, K, V
    
