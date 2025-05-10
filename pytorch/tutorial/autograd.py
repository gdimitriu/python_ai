import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
print(loss)
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# compute gradients
loss.backward()
print(w.grad)
print(b.grad)

# disable gradients tracking
# when we have trained the model and just want to apply it to some input data,
# i.e. we only want to do forward computations through the network.
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# same method using detach
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
