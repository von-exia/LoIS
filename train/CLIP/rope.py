import torch


class RoPE:
    @staticmethod
    def apply_rotary_pos_emb(q, k, cos, sin):
        q_ = (q * cos) + (RoPE.rotate_half(q) * sin)
        k_ = (k * cos) + (RoPE.rotate_half(k) * sin)
        return q_, k_

    @staticmethod
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

def get_rotary_embeddings(head_dim, seq_len, device):
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim {head_dim} must be even to apply RoPE.")
    
    # Generate sin and cos tensors with the full head_dim dimension
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    position_ids = torch.arange(seq_len, dtype=torch.float, device=device)
    sinusoid_inp = torch.einsum("i,j->ij", position_ids, inv_freq)
    
    sin, cos = torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    
    # Expand dimensions to match (batch_size, n_head, seq_len, head_dim/2)
    sin = sin.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim/2)
    
    return sin, cos


# def apply_rotary_pos_emb(x, cos, sin):
#     # Split q and k into two halves along the last dimension
#     x1, x2 = x.split(x.shape[-1] // 2, dim=-1)  # q1 and q2 will each have shape [batch_size, n_head, seq_len, 32]
#     # Apply sin and cos to the respective halves
#     x_ = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

#     return x_


def apply_rotary_pos_emb(q, k, cos, sin):
    # Split q and k into two halves along the last dimension
    q1, q2 = q.split(q.shape[-1] // 2, dim=-1)  # q1 and q2 will each have shape [batch_size, n_head, seq_len, 32]
    k1, k2 = k.split(k.shape[-1] // 2, dim=-1)

    # Print shapes to verify correctness
    # print(f"q1 shape: {q1.shape}, q2 shape: {q2.shape}")  # Expecting [257, 16, 36, 32] for both
    # print(f"k1 shape: {k1.shape}, k2 shape: {k2.shape}")  # Expecting [257, 16, 36, 32] for both

    # Apply sin and cos to the respective halves
    q_ = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_ = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    #print(f"q_ shape after RoPE: {q_.shape}")  # Should match original q shape [257, 16, 36, 64]
    #print(f"k_ shape after RoPE: {k_.shape}")  # Should match original k shape [257, 16, 36, 64]

    return q_, k_


