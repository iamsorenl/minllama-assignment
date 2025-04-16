import torch
from llama import RMSNorm  # adjust if it's in another path

# Sample input: batch of 2 sequences, each with 4 tokens
x = torch.randn(2, 4, 8)  # shape = (batch=2, seq_len=4, dim=8)

# Instantiate RMSNorm with the correct feature dimension
norm = RMSNorm(dim=8)

# Run the normalization
output = norm(x)

# Print output stats
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Check mean and norm of each token vector
for i in range(2):
    for j in range(4):
        print(f"\nToken vector {i}-{j}:")
        print("Before norm RMS:", torch.sqrt(torch.mean(x[i, j] ** 2)).item())
        print("After norm RMS: ", torch.sqrt(torch.mean(output[i, j] ** 2)).item())
