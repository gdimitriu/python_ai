threshold = 1.5
inputs = [1, 0, 1, 0, 1]
weights = [0.7, 0.6, 0.5, 0.3, 0.4]
sum = 0
i = 0
while i < len(inputs):
    sum += inputs[i] * weights[i]
    i += 1
activate = (sum > threshold)
print(activate)

