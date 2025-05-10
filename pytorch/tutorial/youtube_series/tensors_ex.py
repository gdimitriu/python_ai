import torch
import math

x = torch.empty(3, 4)
print(type(x))
print(x)

zeros = torch.zeros(2, 3)
print(zeros)

ones = torch.ones(2, 3)
print(ones)

# random initialize the tensors
print("tensor randomize")
torch.manual_seed(1729)
random = torch.rand(2, 3)
print(random)

torch.manual_seed(1729)
random1 = torch.rand(2, 3)
print(random1)

random2 = torch.rand(2, 3)
print(random2)

torch.manual_seed(1729)
random3 = torch.rand(2, 3)
print(random3)

random4 = torch.rand(2, 3)
print(random4)

print("tensor shapes")
x = torch.empty(2, 2, 3)
print(x.shape)
print(x)

empty_like_x = torch.empty_like(x)
print(empty_like_x.shape)
print(empty_like_x)

zeros_like_x = torch.zeros_like(x)
print(zeros_like_x.shape)
print(zeros_like_x)

ones_like_x = torch.ones_like(x)
print(ones_like_x.shape)
print(ones_like_x)

rand_like_x = torch.rand_like(x)
print(rand_like_x.shape)
print(rand_like_x)

print("from python collections")
some_constants = torch.tensor([[3.1415926, 2.71828], [1.61803, 0.0072897]])
print(some_constants)

some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)

more_integers = torch.tensor(((2, 4, 6), [3, 6, 9]))
print(more_integers)

print("specify the datatypes")
a = torch.ones((2, 3), dtype=torch.int16)
print(a)

b = torch.rand((2, 3), dtype=torch.float64) * 20.
print(b)

c = b.to(torch.int32)
print(c)

print("arithmetic operations on tensors")
ones = torch.zeros(2, 2) + 1
twos = torch.ones(2, 2) * 2
threes = (torch.ones(2, 2) * 7 - 1) / 2
fours = twos ** 2
sqrt2s = twos ** 0.5

print(ones)
print(twos)
print(threes)
print(fours)
print(sqrt2s)

powers2 = twos ** torch.tensor([[1, 2], [3, 4]])
print(powers2)

fives = ones + fours
print(fives)

dozens = threes * fours
print(dozens)

print("tensor broadcasting")
rand = torch.rand(2, 4)
doubled = rand * (torch.ones(1, 4) * 2)

print(rand)
print(doubled)

a = torch.ones(4, 3, 2)

b = a * torch.rand(3, 2)  # 3rd & 2nd dims identical to a, dim 1 absent
print(b)

c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print(c)

d = a * torch.rand(1, 2)  # 3rd dim identical to a, 2nd dim = 1
print(d)

print("changing the tensor shapes")
a = torch.rand(3, 226, 226)
b = a.unsqueeze(0)

print(a.shape)
print(b.shape)

c = torch.rand(1, 1, 1, 1, 1)
print(c)

a = torch.rand(1, 20)
print(a.shape)
print(a)

b = a.squeeze(0)
print(b.shape)
print(b)

c = torch.rand(2, 2)
print(c.shape)

d = c.squeeze(0)
print(d.shape)

a = torch.ones(4, 3, 2)

c = a * torch.rand(3, 1)  # 3rd dim = 1, 2nd dim identical to a
print(c)

a = torch.ones(4, 3, 2)
b = torch.rand(3)  # trying to multiply a * b will give a runtime error
c = b.unsqueeze(1)  # change to a 2-dimensional tensor, adding new dim at the end
print(c.shape)
print(a * c)  # broadcasting works again!

# in place
batch_me = torch.rand(3, 226, 226)
print(batch_me.shape)
batch_me.unsqueeze_(0)
print(batch_me.shape)

output3d = torch.rand(6, 20, 20)
print(output3d.shape)

input1d = output3d.reshape(6 * 20 * 20)
print(input1d.shape)

# can also call it as a method on the torch module:
print(torch.reshape(output3d, (6 * 20 * 20,)).shape)

