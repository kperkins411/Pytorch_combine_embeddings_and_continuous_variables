import torch

# dtype = torch.float32
device = torch.device("cpu")
# torch.cuda.current_device()
# dtype = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in1,D_in2, H, D_out = 64, 100,800, 100, 10

# Create random input and output data
x1 = torch.randn(N, D_in1, device=device)
x2 = torch.randn(N, D_in2, device=device)

y = torch.randn(N, D_out, device=device)

#NO requiresgrad=True
w1 = torch.randn(D_in1 + D_in2, H, device=device)
w2 = torch.randn(H, D_out, device=device)

learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y using operations on Tensors. Since w1 and
    # w2 have requires_grad=True, operations involving these Tensors will cause
    # PyTorch to build a computational graph, allowing automatic computation of
    # gradients. Since we are no longer implementing the backward pass by hand we
    # don't need to keep references to intermediate values.
    xc = torch.cat((x1, x2), 1)
    y_pred = xc.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = xc.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
