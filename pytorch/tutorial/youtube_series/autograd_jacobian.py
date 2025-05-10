import torch

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)  # stand-in for gradients
y.backward(v)

print(x.grad)


# high level API
def exp_adder(x, y):
    return 2 * x.exp() + 3 * y


inputs = (torch.rand(1), torch.rand(1))  # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)
inputs = (torch.rand(3), torch.rand(3))  # arguments for the function
print(inputs)
torch.autograd.functional.jacobian(exp_adder, inputs)


# vector-jacobian product
def do_some_doubling(x):
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    return y


inputs = torch.randn(3)
my_gradients = torch.tensor([0.1, 1.0, 0.0001])
torch.autograd.functional.vjp(do_some_doubling, inputs, v=my_gradients)
