import torch
import math
from einops import einsum

B_k = 16
B_q = 16


class FlashAttn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        *batch, N_q, d = Q.size()  # _ is batch, N is seq_len, d_k is dimensions
        *batch, N_k, d = K.size()  # _ is batch, N is seq_len, d_v is dimensions

        T_q = (N_q + B_q - 1) // B_q  # formula for ceiling division
        T_k = (N_k + B_k - 1) // B_k

        Q_list = torch.split(Q, B_q, dim=1)
        K_list = torch.split(K, B_k, dim=1)
        V_list = torch.split(V, B_k, dim=1)
        O_list = []
        L_list = []

        for i in range(T_q):
            Q_i = Q_list[i]

            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(*batch, B_q)
            m_i = torch.full((*batch, B_q), -float("inf"))

            for j in range(T_k):
                K_j = K_list[j]
                V_j = V_list[j]

                S_ij = einsum(Q_i, K_j, "... b_q d, ... b_k d -> ... b_q b_k") / math.sqrt(d)
                m_i_prev = m_i
                if i == 0 and j == 1:
                    print("S_ij\n\n")
                    print(S_ij.shape)
                    print(f"Dimension of max {torch.max(S_ij, dim=-1).values.shape}")
                    print(f"Dimension of m_i {m_i.shape}")

                m_i = torch.maximum(m_i, torch.max(S_ij, dim=-1).values)
                P_i = torch.exp(S_ij - torch.unsqueeze(m_i, dim=-1))  # dimensions are sus
                l_i = torch.exp(m_i_prev - m_i) * l_i + torch.sum(P_i, dim=-1)  # dim still sus
                if i == 0 and j == 1:
                    print(f"m_i : {m_i}")
                    print(f"m_i_prev: {m_i_prev}")
                    print(f"m_i diff: {m_i_prev - m_i}")
                    print(f"m_i diff diag?: {torch.exp(m_i_prev - m_i).unsqueeze(dim=-1)}")
                    print(f"diag_embed? {torch.exp(torch.diag_embed(m_i_prev - m_i))}")
                    print(f"m_i diff diag shape?: {torch.exp(m_i_prev - m_i).unsqueeze(dim=-1).shape}")
                    print(f"diag_embed shape? {torch.exp(torch.diag_embed(m_i_prev - m_i)).shape}")
                    print(f"{torch.exp(torch.diag_embed(m_i_prev - m_i)) @ O_i}")
                max_diag = torch.diag_embed(torch.exp(m_i_prev - m_i))
                O_i = einsum(max_diag, O_i, "... b_q b_k, ... b_q d -> ... b_k d")
                O_i += einsum(P_i, V_j, "... b_k b_q, ... b_q d -> ... b_k d")

            # O_i, m_i, l_i here should be values for j = T_k
            O_i = einsum(torch.linalg.inv(torch.diag_embed(l_i)), O_i, "... b_q b_k, ... b_q d -> ... b_k d")
            L_i = m_i + torch.log(l_i)

            O_list.append(O_i)
            L_list.append(L_i)

        print(f"O_list: {O_list}")
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
