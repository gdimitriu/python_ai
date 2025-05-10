import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math


a = torch.linspace(0., 2. * math.pi, steps=25, requires_grad=True)
print(a)

b = torch.sin(a)
plt.figure()
plt.title("input sin tensor")
plt.plot(a.detach(), b.detach())
print(b)

c = 2 * b
print(c)

d = c + 1
print(d)

out = d.sum()
print(out)

print('d:')
print(d.grad_fn)
print(d.grad_fn.next_functions)
print(d.grad_fn.next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions)
print(d.grad_fn.next_functions[0][0].next_functions[0][0].next_functions[0][0].next_functions)
print('\nc:')
print(c.grad_fn)
print('\nb:')
print(b.grad_fn)
print('\na:')
print(a.grad_fn)

# get the derivatives
out.backward()
print(a.grad)
plt.figure()
plt.title("derivate of the input")
plt.plot(a.detach(), a.grad.detach())
plt.show()
