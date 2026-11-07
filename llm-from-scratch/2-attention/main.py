import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
)

query = inputs[1]

# Simple attention
# ================
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)
context_vecs = attn_weights @ inputs

print(context_vecs)

# Self-attention with trainable weights
# =====================================
d_in = inputs.shape[1] # d_in = 3
d_out = 2

# initialize 3 weight matrices
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
x_2 = inputs[1]

q_2 = x_2 @ W_query
k_2 = x_2 @ W_key
v_2 = x_2 @ W_value