import torch
import math
from einops import einsum

B_k = 16
B_q = 16

class FlashAttn2(torch.autograd.Function):

    @staticmethod
    def forward(ctx, K: torch.Tensor, Q: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        
        batch, N, d = K.size() # _ is batch, N is seq_len, d_k is dimensions
        _, _, d_v = V.size() # _ is batch, N is seq_len, d_v is dimensions


        T_q = (N + B_q - 1) // B_q # formula for cieling division
        T_k  = (N + B_k - 1) // B_k

        Q_list = torch.split(Q, B_q, dim=1)
        K_list = torch.split(K, B_k, dim=1)
        V_list = torch.split(V, B_k, dim=1)
        O_list = []
        L_list = []

        for i in range(T_q):
            Q_i = Q_list[i]

            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(batch, B_q)
            m_i = torch.full((batch, B_q), -float('inf'))

            for j in range(T_k):
                K_j = K_list[j]
                V_j = V_list[j]

                S_ij = einsum(Q_i, K_j, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d)
                m_i_prev = m_i
                print("S_ij\n\n")
                print(S_ij.shape)
                print(f"Dimension of max {torch.max(S_ij, dim=-1).values.shape}")
                print(f"Dimension of m_i {m_i.shape}")

                m_i = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)
                P_i = torch.exp(S_ij - torch.unsqueeze(m_i, dim=-1)) # dimensions are sus
                l_i = torch.exp(m_i_prev - m_i) * l_i + torch.sum(P_i, dim=-1) # dim still sus
                O_i = torch.exp(m_i_prev - m_i).unsqueeze(dim=-1) * O_i + einsum(P_i, V_j, "... seq seq, ... seq d_v -> ... seq d_v")

            O_i = O_i / l_i.unsqueeze(-1)
            L_i = m_i + torch.log(l_i)

            O_list.append(O_i)
            L_list.append(L_i)

        O = torch.cat(O_list, dim=1)
        L = torch.cat(L_list, dim=1)

        print(f"L dimensions: {L.shape}")
        print(f"O dimensions: {O.shape}")

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
    
# O = FlashAttn2.apply(Q, K, V)
    
