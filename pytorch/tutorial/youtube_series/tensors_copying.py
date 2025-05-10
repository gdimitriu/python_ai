import torch

a = torch.ones(2, 2)
b = a

a[0][1] = 561  # we change a...
print(b)  # ...and b is also altered

a = torch.ones(2, 2)
b = a.clone()

assert b is not a  # different objects in memory...
print(torch.eq(a, b))  # ...but still with the same contents!

a[0][1] = 561  # a changes...
print(b)  # ...but b is still all ones

# autograd
a = torch.rand(2, 2, requires_grad=True)  # turn on autograd
print(a)

b = a.clone()
print(b)

c = a.detach().clone()
print(c)

print(a)
