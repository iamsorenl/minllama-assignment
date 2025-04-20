from typing import Tuple
import torch

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).

    # Compute inverse frequency for each dimension
    freq_seq = torch.arange(0, head_dim, 2, device=device)
    inv_freq = 1.0 / (theta ** (freq_seq.float() / head_dim))  # shape: (head_dim // 2)

    # Generate sequence position indices
    t = torch.arange(seqlen, device=device)  # (seqlen,)

    # Outer product gives us full frequencies
    freqs = torch.outer(t, inv_freq)  # (seqlen, head_dim // 2)

    # Convert to complex representation (e^{iθ} = cos(θ) + i sin(θ))
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seqlen, head_dim // 2)

    # Broadcast to match Q/K tensor shapes
    freqs_cis = reshape_for_broadcast(freqs_cis, query_real)  # (1, seqlen, 1, head_dim // 2)

    # Form complex tensors for Q and K
    query_complex = torch.complex(query_real, query_imag)
    key_complex = torch.complex(key_real, key_imag)

    # Apply element-wise rotation (complex multiplication)
    query_rotated = query_complex * freqs_cis
    key_rotated = key_complex * freqs_cis

    # Convert back to real representation
    query_out = torch.view_as_real(query_rotated).reshape_as(query)
    key_out = torch.view_as_real(key_rotated).reshape_as(key)

    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out