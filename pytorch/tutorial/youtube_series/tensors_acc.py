import torch
if torch.accelerator.is_available():
    print('We have an accelerator!')
else:
    print('Sorry, CPU only.')

if torch.accelerator.is_available():
    gpu_rand = torch.rand(2, 2, device=torch.accelerator.current_accelerator())
    print(gpu_rand)
else:
    print('Sorry, CPU only.')

my_device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)

y = torch.rand(2, 2)
y = y.to(my_device)

x = torch.rand(2, 2)
y = torch.rand(2, 2, device='cuda')
z = x + y  # exception will be thrown
